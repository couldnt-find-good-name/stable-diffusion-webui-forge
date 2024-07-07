import torch
from modules import prompt_parser, devices, sd_samplers_common

from modules.shared import opts, state
import modules.shared as shared
from modules.script_callbacks import CFGDenoiserParams, cfg_denoiser_callback
from modules.script_callbacks import CFGDenoisedParams, cfg_denoised_callback
from modules.script_callbacks import AfterCFGCallbackParams, cfg_after_cfg_callback
from modules_forge import forge_sampler


def catenate_conds(conds):
    if not isinstance(conds[0], dict):
        return torch.cat(conds)

    return {key: torch.cat([x[key] for x in conds]) for key in conds[0].keys()}


def subscript_cond(cond, a, b):
    if not isinstance(cond, dict):
        return cond[a:b]

    return {key: vec[a:b] for key, vec in cond.items()}


def pad_cond(tensor, repeats, empty):
    if not isinstance(tensor, dict):
        return torch.cat([tensor, empty.repeat((tensor.shape[0], repeats, 1))], axis=1)

    tensor['crossattn'] = pad_cond(tensor['crossattn'], repeats, empty)
    return tensor


class CFGDenoiser(torch.nn.Module):
    def __init__(self, sampler):
        super().__init__()
        self.model_wrap = None
        self.mask = None
        self.nmask = None
        self.init_latent = None
        self.steps = None
        self.total_steps = None
        self.step = 0
        self.image_cfg_scale = None
        self.padded_cond_uncond = False
        self.padded_cond_uncond_v0 = False
        self.sampler = sampler
        self.model_wrap = None
        self.p = None
        self.mask_before_denoising = False
        self.classic_ddim_eps_estimation = False

    @property
    def inner_model(self):
        raise NotImplementedError()

    def combine_denoised(self, x_out, conds_list, uncond, cond_scale):
        denoised_uncond = x_out[-uncond.shape[0]:]
        denoised = torch.clone(denoised_uncond)

        for i, conds in enumerate(conds_list):
            for cond_index, weight in conds:
                denoised[i] += (x_out[cond_index] - denoised_uncond[i]) * (weight * cond_scale)

        return denoised

    def combine_denoised_for_edit_model(self, x_out, cond_scale):
        out_cond, out_img_cond, out_uncond = x_out.chunk(3)
        denoised = out_uncond + cond_scale * (out_cond - out_img_cond) + self.image_cfg_scale * (out_img_cond - out_uncond)

        return denoised

    def get_pred_x0(self, x_in, x_out, sigma):
        return x_out

    def update_inner_model(self):
        self.model_wrap = None

        c, uc = self.p.get_conds()
        self.sampler.sampler_extra_args['cond'] = c
        self.sampler.sampler_extra_args['uncond'] = uc

    def pad_cond_uncond(self, cond, uncond):
        empty = shared.sd_model.cond_stage_model_empty_prompt
        num_repeats = (cond.shape[1] - uncond.shape[1]) // empty.shape[1]

        if num_repeats < 0:
            cond = pad_cond(cond, -num_repeats, empty)
            self.padded_cond_uncond = True
        elif num_repeats > 0:
            uncond = pad_cond(uncond, num_repeats, empty)
            self.padded_cond_uncond = True

        return cond, uncond

    def pad_cond_uncond_v0(self, cond, uncond):
        is_dict_cond = isinstance(uncond, dict)
        uncond_vec = uncond['crossattn'] if is_dict_cond else uncond

        if uncond_vec.shape[1] < cond.shape[1]:
            last_vector = uncond_vec[:, -1:]
            last_vector_repeated = last_vector.repeat([1, cond.shape[1] - uncond_vec.shape[1], 1])
            uncond_vec = torch.hstack([uncond_vec, last_vector_repeated])
            self.padded_cond_uncond_v0 = True
        elif uncond_vec.shape[1] > cond.shape[1]:
            uncond_vec = uncond_vec[:, :cond.shape[1]]
            self.padded_cond_uncond_v0 = True

        if is_dict_cond:
            uncond['crossattn'] = uncond_vec
        else:
            uncond = uncond_vec

        return cond, uncond

    def apply_blend(self, current_latent, sigma):
        blended_latent = current_latent * self.nmask + self.init_latent * self.mask

        if self.p.scripts is not None:
            from modules import scripts
            mba = scripts.MaskBlendArgs(current_latent, self.nmask, self.init_latent, self.mask, blended_latent, denoiser=self, sigma=sigma)
            self.p.scripts.on_mask_blend(self.p, mba)
            blended_latent = mba.blended_latent

        return blended_latent

    def forward(self, x, sigma, uncond, cond, cond_scale, s_min_uncond, image_cond):
        if state.interrupted or state.skipped:
            raise sd_samplers_common.InterruptedException

        original_x_device = x.device
        original_x_dtype = x.dtype

        if self.classic_ddim_eps_estimation:
            acd = self.inner_model.inner_model.alphas_cumprod
            fake_sigmas = ((1 - acd) / acd) ** 0.5
            real_sigma = fake_sigmas[sigma.round().long().clip(0, int(fake_sigmas.shape[0]))]
            real_sigma_data = 1.0
            x = x * (((real_sigma ** 2.0 + real_sigma_data ** 2.0) ** 0.5)[:, None, None, None])
            sigma = real_sigma

        if sd_samplers_common.apply_refiner(self, sigma):
            cond = self.sampler.sampler_extra_args['cond']
            uncond = self.sampler.sampler_extra_args['uncond']

        is_edit_model = shared.sd_model.cond_stage_key == "edit" and self.image_cfg_scale is not None and self.image_cfg_scale != 1.0

        cond_composition, cond = prompt_parser.reconstruct_multicond_batch(cond, self.step)
        uncond = prompt_parser.reconstruct_cond_batch(uncond, self.step)

        assert not is_edit_model or all(len(conds) == 1 for conds in cond_composition), "AND is not supported for InstructPix2Pix checkpoint (unless using Image CFG scale = 1.0)"

        # Blend in the original latents (before)
        if self.mask_before_denoising and self.mask is not None:
            x = self.apply_blend(x, sigma)

        batch_size = len(cond_composition)

        denoiser_params = CFGDenoiserParams(x, image_cond, sigma, state.sampling_step, state.sampling_steps, cond, uncond, self)
        cfg_denoiser_callback(denoiser_params)

        x = denoiser_params.x
        image_cond = denoiser_params.image_cond
        sigma = denoiser_params.sigma
        cond = denoiser_params.text_cond
        uncond = denoiser_params.text_uncond

        # Handle skip_uncond logic
        if (shared.opts.skip_early_cond != 0. and self.step / self.total_steps <= shared.opts.skip_early_cond) or \
        ((self.step % 2 or shared.opts.s_min_uncond_all) and s_min_uncond > 0 and sigma[0] < s_min_uncond and not is_edit_model):
            # Modify cond_composition to skip uncond
            cond_composition = [conds for conds in cond_composition if any(weight != 0 for _, weight in conds)]
            if not cond_composition:
                cond_composition = [[(0, 1.0)]]  # Fallback to using the first condition if all were skipped

        denoised = forge_sampler.forge_sample(self, denoiser_params=denoiser_params,
                                            cond_scale=cond_scale, cond_composition=cond_composition)

        # Blend in the original latents (after)
        if not self.mask_before_denoising and self.mask is not None:
            denoised = self.apply_blend(denoised, sigma)

        self.sampler.last_latent = self.get_pred_x0(x, denoised, sigma)

        if opts.live_preview_content == "Prompt":
            preview = self.sampler.last_latent
        elif opts.live_preview_content == "Negative prompt":
            preview = self.get_pred_x0(x, denoised, sigma)
        else:
            preview = self.get_pred_x0(x, denoised, sigma)

        sd_samplers_common.store_latent(preview)

        after_cfg_callback_params = AfterCFGCallbackParams(denoised, state.sampling_step, state.sampling_steps)
        cfg_after_cfg_callback(after_cfg_callback_params)
        denoised = after_cfg_callback_params.x

        self.step += 1

        if self.classic_ddim_eps_estimation:
            eps = (x - denoised) / sigma[:, None, None, None]
            return eps

        return denoised.to(device=original_x_device, dtype=original_x_dtype)
    