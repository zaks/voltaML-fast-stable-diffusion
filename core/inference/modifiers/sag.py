from contextlib import ExitStack

import torch
import torch.nn.functional as F

from ..step import DiffusionModifier, DiffusionData


class SAGStep(DiffusionModifier):
    """Self Attention Guidance reimplemented as a step in the diffusion pipeline."""

    def __init__(self, self_attention_scale: float) -> None:
        super().__init__("Self-Attention Guidance")
        self.self_attention_scale = self_attention_scale

    _map_size = None
    _store_processor = None

    def _get_map_size(self, _, __, output):
        global _map_size
        _map_size = output.sample.shape[-2:]

    def enter_context(self, context: ExitStack, data: DiffusionData) -> DiffusionData:
        if hasattr(data.pipeline, "unet"):
            if hasattr(data.pipeline.unet, "unet"):
                global _store_processor
                _store_processor = CrossAttnStoreProcessor()
                data.pipeline.unet.unet.mid_block.attentions[0].transformer_blocks[0].attn1.processor = _store_processor  # type: ignore

                context.enter_context(
                    data.pipeline.unet.unet.mid_block.attentions[
                        0
                    ].register_forward_hook(self._get_map_size)
                )
        return data

    def post_unet(self, data: DiffusionData) -> DiffusionData:
        classifier_free_guidance = data.cfg_scale > 1.0
        if classifier_free_guidance:
            pred = self.pred_x0(
                data.pipeline.scheduler,
                data.latents,
                data.noise_pred_uncond,
                data.timesteps,
            )
            uncond_attn, cond_attn = _store_processor.attention_probs.chunk(2)
            degraded_latents = self.sag_masking(
                data.pipeline.unet.unet,
                data.pipeline.scheduler,
                pred,
                uncond_attn,
                data.timesteps,
                self.pred_epsilon(
                    data.pipeline.scheduler,
                    data.latents,
                    data.noise_pred_uncond,
                    data.timesteps,
                ),
            )
            uncond_emb, _ = data.encoder_hidden_states.chunk(2)
            degraded_prep = data.pipeline.unet.infer(
                latent_model_input=degraded_latents,
                timestep=data.timesteps,
                encoder_hidden_states=uncond_emb,
            )
            data.noise_pred += self.self_attention_scale * (
                data.noise_pred_uncond - degraded_prep
            )
        else:
            pred = self.pred_x0(
                data.pipeline.scheduler, data.latents, data.noise_pred, data.timesteps
            )
            cond_attn = _store_processor.attention_probs
            degraded_latents = self.sag_masking(
                data.pipeline.unet.unet,
                data.pipeline.scheduler,
                pred,
                cond_attn,
                data.timesteps,
                self.pred_epsilon(
                    data.pipeline.scheduler,
                    data.latents,
                    data.noise_pred,
                    data.timesteps,
                ),
            )
            degraded_prep = data.pipeline.unet.infer(
                latent_model_input=degraded_latents,
                timestep=data.timesteps,
                encoder_hidden_states=data.encoder_hidden_states,
            )
            data.noise_pred += self.self_attention_scale * (
                data.noise_pred - degraded_prep
            )
        return data

    def pred_x0(self, scheduler, sample, model_output, timestep):
        """
        Modified from diffusers.schedulers.scheduling_ddim.DDIMScheduler.step
        Note: there are some schedulers that clip or do not return x_0 (PNDMScheduler, DDIMScheduler, etc.)
        """
        alpha_prod_t = scheduler.alphas_cumprod[
            timestep.to(scheduler.alphas_cumprod.device, dtype=torch.int64)
        ]

        beta_prod_t = 1 - alpha_prod_t
        if scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
        elif scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output
            # predict V
            model_output = (alpha_prod_t**0.5) * model_output + (
                beta_prod_t**0.5
            ) * sample
        else:
            raise ValueError(
                f"prediction_type given as {scheduler.config.prediction_type} must be one of `epsilon`, `sample`,"
                " or `v_prediction`"
            )

        return pred_original_sample

    def sag_masking(self, unet, scheduler, original_latents, attn_map, t, eps):
        "sag_masking"
        # Same masking process as in SAG paper: https://arxiv.org/pdf/2210.00939.pdf
        global _map_size
        _, hw1, hw2 = attn_map.shape
        b, latent_channel, latent_h, latent_w = original_latents.shape
        h = unet.config.attention_head_dim
        if isinstance(h, list):
            h = h[-1]

        # Produce attention mask
        attn_map = attn_map.reshape(b, h, hw1, hw2)
        attn_mask = attn_map.mean(1, keepdim=False).sum(1, keepdim=False) > 1.0
        attn_mask = (
            attn_mask.reshape(b, _map_size[0], _map_size[1])
            .unsqueeze(1)
            .repeat(1, latent_channel, 1, 1)
            .type(attn_map.dtype)
        )
        attn_mask = F.interpolate(attn_mask, (latent_h, latent_w))

        # Blur according to the self-attention mask
        degraded_latents = self.gaussian_blur_2d(
            original_latents, kernel_size=9, sigma=1.0
        )
        degraded_latents = degraded_latents * attn_mask + original_latents * (
            1 - attn_mask
        )

        # Noise it again to match the noise level
        degraded_latents = scheduler.add_noise(
            degraded_latents, noise=eps, timesteps=torch.tensor([t])
        )

        return degraded_latents

    def pred_epsilon(self, scheduler, sample, model_output, timestep):
        "pred_epsilon"
        alpha_prod_t = scheduler.alphas_cumprod[
            timestep.to(scheduler.alphas_cumprod.device, dtype=torch.int64)
        ]

        beta_prod_t = 1 - alpha_prod_t
        if scheduler.config.prediction_type == "epsilon":
            pred_eps = model_output
        elif scheduler.config.prediction_type == "sample":
            pred_eps = (sample - (alpha_prod_t**0.5) * model_output) / (
                beta_prod_t**0.5
            )
        elif scheduler.config.prediction_type == "v_prediction":
            pred_eps = (beta_prod_t**0.5) * sample + (
                alpha_prod_t**0.5
            ) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {scheduler.config.prediction_type} must be one of `epsilon`, `sample`,"
                " or `v_prediction`"
            )

        return pred_eps

    def gaussian_blur_2d(self, img, kernel_size, sigma):
        "Blurs an image with gaussian blur."
        ksize_half = (kernel_size - 1) * 0.5

        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

        pdf = torch.exp(-0.5 * (x / sigma).pow(2))

        x_kernel = pdf / pdf.sum()
        x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

        kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
        kernel2d = kernel2d.expand(
            img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1]
        )

        padding = [
            kernel_size // 2,
            kernel_size // 2,
            kernel_size // 2,
            kernel_size // 2,
        ]

        img = F.pad(img, padding, mode="reflect")
        img = F.conv2d(img, kernel2d, groups=img.shape[-3])

        return img


class CrossAttnStoreProcessor:
    "Modified Cross Attention Processor with capabilities to store probabilities."

    def __init__(self):
        self.attention_probs = None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        self.attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(self.attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
