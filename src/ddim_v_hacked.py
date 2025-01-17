"""SAMPLING ONLY."""

# CrossAttn precision handling
import os
import math
import einops
import numpy as np
import torch
from tqdm.notebook import tqdm
from einops import rearrange

from ControlNet.ldm.modules.diffusionmodules.util import (
    extract_into_tensor, make_ddim_sampling_parameters, make_ddim_timesteps,
    noise_like)

_ATTN_PRECISION = os.environ.get('ATTN_PRECISION', 'fp32')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def register_attention_control(model, controller=None):

    def ca_forward(self, place_in_unet):

        def forward(x, context=None, mask=None):
            h = self.heads
            self.attn_in = x.clone().detach()
            if controller.cur_frame_x is not None:
                x = controller.cur_frame_x
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            context = controller(context, is_cross, place_in_unet)

            k = self.to_k(context)
            v = self.to_v(context)

            q, k, v = map(
                lambda t: einops.rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                (q, k, v))

            # force cast to fp32 to avoid overflowing
            if _ATTN_PRECISION == 'fp32':
                with torch.autocast(enabled=False, device_type=device):
                    q, k = q.float(), k.float()
                    sim = torch.einsum('b i d, b j d -> b i j', q,
                                       k) * self.scale
            else:
                sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

            del q, k

            if mask is not None:
                mask = einops.rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = einops.repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of

            # randon_mask_attn
            if controller.random_noise:
                i, j = controller.random_pos_1
                k, l = controller.random_pos_2
                n = sim.shape[-2]
                # random_mask = torch.rand_like(sim[..., 0].unsqueeze(-1)).repeat(1, 1, sim.shape[-1])
                # sim = torch.where(random_mask <= 0.1, torch.randn_like(sim), sim)
                if controller.block_idx < controller.random_th:

                    times = int(math.sqrt((512 * 576) / (n)))
                    w = 576 // times
                    h_ = 512 // times
                    sim = rearrange(sim, 'b (h w) c -> b c h w', h=h_, w=w).contiguous()  # 转化回 b c h w
                    i = int(i * w)
                    j = int(j * h_)
                    k = int(k * w)
                    l = int(l * h_)
                    sim[...,j:l,i:k] = torch.rand_like(sim[...,j:l,i:k])
                    sim = rearrange(sim, 'b c h w -> b (h w) c').contiguous()
                    # sim = torch.flip(sim, dims=[-2])
            sim = sim.softmax(dim=-1)
            self.tmp_sim = sim
            # active plug
            if controller.attn is not None:
                sim = controller.attn


            out = torch.einsum('b i j, b j d -> b i d', sim, v)
            out = einops.rearrange(out, '(b h) n d -> b n (h d)', h=h)
            self.attn_out = out.clone().detach()
            return self.to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.cur_step = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                register_recr(net__, place_in_unet)

    sub_nets = model.named_children()
    for net in sub_nets:
        if 'input_blocks' in net[0]:
            register_recr(net[1], 'down')
        elif 'output_blocks' in net[0]:
            register_recr(net[1], 'up')
        elif 'middle_block' in net[0]:
            register_recr(net[1], 'mid')


class DDIMVSampler(object):

    def __init__(self, model, schedule='linear', **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device(device):
                attr = attr.to(torch.device(device))
        setattr(self, name, attr)

    def make_schedule(self,
                      ddim_num_steps,
                      ddim_discretize='uniform',
                      ddim_eta=0.,
                      verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, \
            'alphas have to be defined for each timestep'

        def to_torch(x):
            return x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = \
            make_ddim_sampling_parameters(
                alphacums=alphas_cumprod.cpu(),
                ddim_timesteps=self.ddim_timesteps,
                eta=ddim_eta,
                verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas',
                             np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) *
            (1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps',
                             sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               xtrg=None,
               noise_rescale=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               dynamic_threshold=None,
               ucg_schedule=None,
               controller=None,
               strength=0.0,
               **kwargs):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list):
                    ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f'Warning: Got {cbs} conditionings'
                          f'but batch-size is {batch_size}')

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f'Warning: Got {cbs} conditionings'
                              f'but batch-size is {batch_size}')

            else:
                if conditioning.shape[0] != batch_size:
                    print(f'Warning: Got {conditioning.shape[0]}'
                          f'conditionings but batch-size is {batch_size}')

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        # print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(
            conditioning,
            size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            xtrg=xtrg,
            noise_rescale=noise_rescale,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            dynamic_threshold=dynamic_threshold,
            ucg_schedule=ucg_schedule,
            controller=controller,
            strength=strength,
        )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self,
                      cond,
                      shape,
                      x_T=None,
                      ddim_use_original_steps=False,
                      callback=None,
                      timesteps=None,
                      quantize_denoised=False,
                      mask=None,
                      x0=None,
                      xtrg=None,
                      noise_rescale=None,
                      img_callback=None,
                      log_every_t=100,
                      temperature=1.,
                      noise_dropout=0.,
                      score_corrector=None,
                      corrector_kwargs=None,
                      unconditional_guidance_scale=1.,
                      unconditional_conditioning=None,
                      dynamic_threshold=None,
                      ucg_schedule=None,
                      controller=None,
                      strength=0.0):

        if strength == 1 and x0 is not None:
            return x0, None

        register_attention_control(self.model.model.diffusion_model,
                                   controller)

        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps \
                else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(
                min(timesteps / self.ddim_timesteps.shape[0], 1) *
                self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(
            0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps \
            else timesteps.shape[0]
        # print(f'Running DDIM Sampling with {total_steps} timesteps')

        # iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        iterator = time_range
        if controller is not None:
            controller.set_total_step(total_steps)
        if mask is None:
            mask = [None] * total_steps

        dir_xt = 0
        for i, step in enumerate(iterator):
            if controller is not None:
                controller.set_step(i)
            index = total_steps - i - 1
            ts = torch.full((b, ), step, device=device, dtype=torch.long)

            if strength > 1:
                if i == int(total_steps * (strength - 1)):
                    img = x0
            else:
                if strength >= 0 and i == int(
                        total_steps * strength) and x0 is not None:
                    img = self.model.q_sample(x0, ts)
            if mask is not None and xtrg is not None:
                # TODO: deterministic forward pass?
                if type(mask) == list:
                    weight = mask[i]
                else:
                    weight = mask
                if weight is not None:
                    rescale = torch.maximum(1. - weight, (1 - weight**2)**0.5 *
                                            controller.inner_strength)
                    if noise_rescale is not None:
                        rescale = (1. - weight) * (
                            1 - noise_rescale) + rescale * noise_rescale
                    img_ref = self.model.q_sample(xtrg, ts)
                    img = img_ref * weight + (1. - weight) * (
                        img - dir_xt) + rescale * dir_xt

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]

            outs = self.p_sample_ddim(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                dynamic_threshold=dynamic_threshold,
                controller=controller,
                return_dir=True,
                img_callback=img_callback,
                i=i,)
            img, pred_x0, dir_xt = outs
            if callback:
                callback(i)
            # if img_callback:
            #     img_callback(i, )

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self,
                      x,
                      c,
                      t,
                      index,
                      repeat_noise=False,
                      use_original_steps=False,
                      quantize_denoised=False,
                      temperature=1.,
                      noise_dropout=0.,
                      score_corrector=None,
                      corrector_kwargs=None,
                      unconditional_guidance_scale=1.,
                      unconditional_conditioning=None,
                      dynamic_threshold=None,
                      controller=None,
                      return_dir=False,
                      img_callback=None,
                      i=None,):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or \
                unconditional_guidance_scale == 1.:
            model_output = self.model.apply_model(x, t, c)
        else:
            controller.block_idx = 3
            model_t = self.model.apply_model(x, t, c)
            if img_callback is not None:
                img_callback(i)
            controller.block_idx = 3
            model_uncond = self.model.apply_model(x, t,
                                                  unconditional_conditioning)
            model_output = model_uncond + unconditional_guidance_scale * (
                model_t - model_uncond)

        if self.model.parameterization == 'v':
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == 'eps', 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c,
                                               **corrector_kwargs)

        if use_original_steps:
            alphas = self.model.alphas_cumprod
            alphas_prev = self.model.alphas_cumprod_prev
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod
            sigmas = self.model.ddim_sigmas_for_original_num_steps
        else:
            alphas = self.ddim_alphas
            alphas_prev = self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
            sigmas = self.ddim_sigmas

        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1),
                                       sqrt_one_minus_alphas[index],
                                       device=device)

        # current prediction for x_0
        if self.model.parameterization != 'v':
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()
        '''
        if mask is not None and xtrg is not None:
            pred_x0 = xtrg * mask + (1. - mask) * pred_x0
        '''

        if controller is not None:
            pred_x0 = controller.update_x0(pred_x0)

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device,
                                     repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        if return_dir:
            return x_prev, pred_x0, dir_xt
        return x_prev, pred_x0

    @torch.no_grad()
    def single_sample(self, x, c, t, index, unconditional_guidance_scale=1.,
                      unconditional_conditioning=None, repeat_noise=False, temperature=1.):
        # temperature 始终是1
        b, *_, device = *x.shape, x.device
        model_t = self.model.apply_model(x, t, c)
        model_uncond = self.model.apply_model(x, t,
                                              unconditional_conditioning)
        model_output = model_uncond + unconditional_guidance_scale * (
                model_t - model_uncond)

        e_t = model_output

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)  # sigma_t is 0
        sqrt_one_minus_at = torch.full((b, 1, 1, 1),
                                       sqrt_one_minus_alphas[index],
                                       device=device)

        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device,
                                     repeat_noise) * temperature  # noise is 0
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0


    @torch.no_grad()
    def encode_ddim(self, img, num_steps,conditioning, unconditional_conditioning=None, unconditional_guidance_scale=1.,
                    controller=None, img_callback=None):
        if controller is not None:
            register_attention_control(self.model.model.diffusion_model,
                                       controller)
        T = 1000
        c = T // num_steps
        # iterator = tqdm(range(1, T - c, c), desc='DDIM Inversion', total=num_steps)
        iterator = range(1, T - c, c)
        steps = list(range(1, T, c))
        x_inversion = img
        x_inversion_lst = []
        x0_inversion_lst = []
        for i, t in enumerate(iterator):
            x0_inversion, x_inversion = self.reverse_ddim(x_inversion, t, t_next=steps[i+1], c=conditioning,
                                                          unconditional_conditioning=unconditional_conditioning,
                                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                                          img_callback=img_callback, i=i)
            break
            # x_inversion_lst.append(x_inversion.detach())
            # x0_inversion_lst.append(x0_inversion.detach())

        return x0_inversion_lst, x_inversion_lst

    @torch.no_grad()
    def reverse_ddim(self, x, t, t_next, c=None, quantize_denoised=False, unconditional_guidance_scale=1.,
                     unconditional_conditioning=None, img_callback=None, i=None):
        b, *_, device = *x.shape, x.device

        t_tensor = torch.full((b,), t, device=device, dtype=torch.long)
        if c is None:
            e_t = self.model.apply_model(x, t_tensor, unconditional_conditioning)
        elif unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            c['c_concat'] = None
            e_t = self.model.apply_model(x, t_tensor, c)
            if img_callback:
                img_callback(i)
        else:
            # x_in = torch.cat([x] * 2)
            # t_in = torch.cat([t_tensor] * 2)
            # c_in = {'c_crossattn': [torch.stack(unconditional_conditioning['c_crossattn'], 0),
            #                         torch.stack(c['c_crossattn'], 0)],
            #         'c_concat': None}
            # e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            # e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            x_in = x
            t_in = t_tensor
            c_in = {'c_crossattn': c['c_crossattn'], 'c_concat': None}
            c_un_in = {'c_crossattn': unconditional_conditioning['c_crossattn'], 'c_concat': None}
            e_t = self.model.apply_model(x_in, t_in, c_in)
            e_t_uncond = self.model.apply_model(x_in, t_in, c_un_in)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        alphas = self.model.alphas_cumprod
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod

        a_t = torch.full((b,1,1,1), alphas[t], device=device)
        a_next = torch.full((b,1,1,1), alphas[t_next], device=device)
        sqrt_one_minus_at = torch.full((b,1,1,1), sqrt_one_minus_alphas[t], device=device)

        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        dir_xt = (1. - a_next).sqrt() * e_t
        x_next = a_next.sqrt() * pred_x0 + dir_xt
        return pred_x0, x_next


    @torch.no_grad()
    def encode(self,
               x0,
               c,
               t_enc,
               use_original_steps=False,
               return_intermediates=None,
               unconditional_guidance_scale=1.0,
               unconditional_conditioning=None,
               callback=None):
        timesteps = np.arange(self.ddpm_num_timesteps
                              ) if use_original_steps else self.ddim_timesteps
        num_reference_steps = timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0], ),
                           timesteps[i],
                           device=self.model.device,
                           dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                noise_pred = self.model.apply_model(x_next, t, c)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(
                        torch.cat((x_next, x_next)), torch.cat((t, t)),
                        torch.cat((unconditional_conditioning, c))), 2)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (
                    noise_pred - e_t_uncond)
            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                (1 / alphas_next[i] - 1).sqrt() -
                (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (num_steps // return_intermediates
                                             ) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback:
                callback(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        if t >= len(sqrt_alphas_cumprod):
            return noise
        return (
            extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
            extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) *
            noise)

    @torch.no_grad()
    def decode(self,
               x_latent,
               cond,
               t_start,
               unconditional_guidance_scale=1.0,
               unconditional_conditioning=None,
               use_original_steps=False,
               callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps
                              ) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        # print(f'Running DDIM Sampling with {total_steps} timesteps')

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0], ),
                            step,
                            device=x_latent.device,
                            dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(
                x_dec,
                cond,
                ts,
                index=index,
                use_original_steps=use_original_steps,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning)
            if callback:
                callback(i)
        return x_dec


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat -
                       content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)
