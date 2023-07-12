from contextlib import ExitStack
from pathlib import Path
from typing import Any, Optional, List, Callable, Literal, Tuple, Union
import inspect

import numpy as np
from diffusers import ControlNetModel, MultiControlNetModel, SchedulerMixin
from diffusers.utils import PIL_INTERPOLATION
from PIL.Image import Image, fromarray
import torch
from transformers import CLIPTokenizer

from .backends import BaseUNetBackend, BaseCLIPBackend, BaseVAEBackend
from ..config import config
from .functions import progress_bar_style
from .modifier import DiffusionModifier, DiffusionData
from .pytorch.lwp import get_weighted_text_embeddings


class UnifiedPipeline:
    scheduler: SchedulerMixin
    tokenizer: CLIPTokenizer
    unet: BaseUNetBackend
    clip: BaseCLIPBackend
    vae: BaseVAEBackend

    controlnet: Union[ControlNetModel, MultiControlNetModel]

    def load_textual_inversion(self, textual_inversion: Path) -> bool:
        "Load a textual inversion."

    def unload_lora(self, lora_name: str) -> bool:
        "Unload a lora from this model."
        return False

    def load_lora(self, lora_path: Path, lora_name: str) -> bool:
        "Load lora into this model."
        return False

    def _maybe_convert_prompt(
        self, prompt: str, tokenizer: CLIPTokenizer
    ) -> torch.Tensor:
        # TODO: implement this -> textual inversion
        return torch.tensor(0)

    def _encode_prompt(
        self,
        prompt: str,
        images_per_prompt: int,
        classifier_free_guidance: bool,
        negative_prompt: Optional[str] = None,
        max_embeddings_multiples: int = 3,
    ) -> torch.Tensor:
        batch_size = 1
        prompt = self._maybe_convert_prompt(prompt, self.tokenizer)
        negative_prompt = self._maybe_convert_prompt(negative_prompt, self.tokenizer)

        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        else:
            negative_prompt = [negative_prompt] * batch_size
        text_embeddings, uncond_embeddings = get_weighted_text_embeddings(
            pipe=self,
            prompt=prompt,
            uncond_prompt=negative_prompt if classifier_free_guidance else None,
            max_embeddings_multiples=max_embeddings_multiples,
        )
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, images_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * images_per_prompt, seq_len, -1
        )

        if classifier_free_guidance:
            bs_embed, seq_len, _ = uncond_embeddings.shape
            uncond_embeddings = uncond_embeddings.repeat(1, images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                bs_embed * images_per_prompt, seq_len, -1
            )
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def _prepare_latents(
        self,
        image: Optional[torch.Tensor],
        timestep: torch.FloatTensor,
        batch_size: int,
        height: Optional[int],
        width: Optional[int],
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator],
        latents=None,
    ):
        vae_scale_factor = 2 ** (
            len(self.vae.backend_object.config.block_out_channels) - 1
        )
        if image is None:
            shape = (
                batch_size,
                self.unet.backend_object.config.in_channels,  # type: ignore
                height // vae_scale_factor,
                width // vae_scale_factor,
            )

            if latents is None:
                latents = torch.randn(
                    shape,
                    generator=generator,
                    device="cpu" if generator.device != device else generator.device,
                    dtype=dtype,
                ).to(device)
            else:
                if latents.shape != shape:
                    raise ValueError(
                        f"Unexpected latents shape, got {latents.shape}, expected {shape}"
                    )
                latents = latents.to(device)

            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma  # type: ignore
            return latents, None, None
        else:
            if image.shape[1] != 4:
                init_latent_dist = self.vae.encode(image.to(config.api.device)).latent_dist  # type: ignore
                init_latents = init_latent_dist.sample(generator=generator)
                init_latents = 0.18215 * init_latents
                init_latents = torch.cat([init_latents] * batch_size, dim=0)
            else:
                init_latents = image

            init_latents.to(device)
            init_latents_orig = init_latents
            shape = init_latents.shape

            # add noise to latents using the timesteps
            noise = torch.randn(
                shape,
                generator=generator,
                device="cpu" if generator.device != device else generator.device,
                dtype=dtype,
            ).to(device)
            latents = self.scheduler.add_noise(init_latents, noise, timestep.to(device))  # type: ignore
            return latents, init_latents_orig, noise

    def _decode_latents(
        self, data: DiffusionData, modifiers: Optional[List[DiffusionModifier]]
    ) -> np.ndarray:
        if modifiers is not None:
            for modifier in modifiers:
                modifier.pre_vae(data)
        data.latents = 1 / 0.18215 * data.latents
        image: torch.Tensor = self.vae.decode(data.latents)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if modifiers is not None:
            for modifier in modifiers:
                modifier.post_vae(data)
        return image

    def _get_timesteps(
        self,
        inference_steps: int,
        strength: float,
        device: torch.device,
        is_txt2img: bool,
    ) -> Tuple[torch.Tensor, int]:
        if is_txt2img:
            return self.scheduler.timesteps.to(device), inference_steps
        else:
            offset = self.scheduler.config.get("steps_offset", 0)  # type: ignore
            init_timestep = int(inference_steps * strength) + offset
            init_timestep = min(init_timestep, inference_steps)

            t_start = max(inference_steps - init_timestep + offset, 0)
            timesteps = self.scheduler.timesteps[t_start:].to(device)
            return timesteps, inference_steps - t_start

    def _extra_kwargs(self, generator: Optional[torch.Generator], eta: Optional[float]):
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def _prepare_image(
        self,
        image: Image,
        width: int,
        height: int,
        batch_size: int,
        img_per_prompt: int,
        device: torch.device,
        dtype: torch.dtype,
        scale_factor: int = 8,
        classifier_free: bool = False,
        guess_mode: bool = False,
    ) -> torch.Tensor:
        image = image.convert("RGB")
        width, height = (x - x % scale_factor for x in (width, height))
        image = image.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32)
        repeat_by = batch_size if image.shape[0] == 1 else img_per_prompt
        image = image.repeat_interleave(repeat_by, dim=0)
        image = image.to(device=device, dtype=dtype)

        if classifier_free and not guess_mode:
            image = torch.cat([image] * 2)
        return image

    def _preprocess_image(self, image: Image):
        w, h = image.size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.0 * image - 1.0

    def _preprocess_mask(self, mask, scale_factor=8):
        mask = mask.convert("L")
        w, h = mask.size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        mask = mask.resize(
            (w // scale_factor, h // scale_factor),
            resample=PIL_INTERPOLATION["nearest"],
        )
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = np.tile(mask, (4, 1, 1))
        mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
        mask = 1 - mask  # repaint white, keep black
        mask = torch.from_numpy(mask)
        return mask

    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        image: Optional[Image] = None,
        control_image: Optional[Image] = None,
        mask_image: Optional[Image] = None,
        width: int = 512,
        height: int = 512,
        steps: int = 50,
        images_per_prompt: int = 1,
        strength: float = 0.8,
        guidance_scale: float = 0.0,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        max_embeddings_multiples: int = 3,
        output_type: Literal["pil", "latent"] = "pil",
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        modifiers: Optional[List[DiffusionModifier]] = None,
        # Controlnet-specific stuff
        controlnet_conditioning_scale: Optional[float] = 1.0,
        guess_mode: Optional[bool] = False,
        controlnet_guidance_start: Optional[float] = 0.0,
        controlnet_guidance_end: Optional[float] = 1.0,
    ) -> Any:
        # Add modifier steps to the total
        total = (
            4
            + (mask_image is not None)  # do we do inpaint
            + (control_image is not None)  # do we do controlnet
            + (image is not None)  # do we do img2img
        )

        classifier_free_guidance = guidance_scale > 1.0
        batch_size = 1
        device = self.unet.backend_object.device

        if self.controlnet is not None:
            mult = (
                len(self.controlnet.nets)
                if isinstance(self.controlnet, MultiControlNetModel)
                else 1
            )
            controlnet_guidance_start, controlnet_guidance_end = mult * [
                controlnet_guidance_start
            ], mult * [controlnet_guidance_end]
            if isinstance(self.controlnet, MultiControlNetModel) and isinstance(
                controlnet_conditioning_scale, float
            ):
                controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(
                    self.controlnet.nets
                )
            global_pool_conditions = (
                self.controlnet.config.global_pool_conditions
                if isinstance(self.controlnet, ControlNetModel)
                else self.controlnet.nets[0].config.global_pool_conditions
            )
            guess_mode = guess_mode or global_pool_conditions

        with progress_bar_style as progress_bar:
            task = progress_bar.add_task("[red]Generating in progress...", total=total)
            progress_task = lambda x: progress_bar.advance(task) and setattr(
                progress_bar.tasks[task], "description", x
            )
            with ExitStack() as gs:
                data = DiffusionData(
                    self, width=width, height=height, cfg_scale=guidance_scale
                )
                if modifiers is not None:
                    for modifier in modifiers:
                        data = modifier.setup(gs, data)
                progress_task("[red]CLIP inference in progress...")
                data.encoder_hidden_states = self._encode_prompt(
                    prompt,
                    images_per_prompt,
                    classifier_free_guidance,
                    negative_prompt,
                    max_embeddings_multiples,
                )
                dtype = data.encoder_hidden_states.dtype
                if image is not None:
                    progress_task("[red]Preprocessing image")
                    image: torch.Tensor = self._preprocess_image(image)
                if control_image is not None:
                    progress_task("[red]Preprocessing controlnet image")
                    control_image: torch.Tensor = self._prepare_image(
                        control_image,
                        width,
                        height,
                        batch_size * images_per_prompt,
                        images_per_prompt,
                        device,
                        dtype,
                        classifier_free=classifier_free_guidance,
                        guess_mode=guess_mode,
                    )
                if mask_image is not None:
                    progress_task("[red]Preprocessing mask")
                    mask_image: torch.Tensor = self._preprocess_mask(
                        mask_image,
                        2 ** len(self.vae.backend_object.config.block_out_channels) - 1,
                    )
                    mask = mask_image.to(device=device, dtype=dtype)
                    mask = torch.cat([mask] * batch_size * images_per_prompt)
                else:
                    mask = None
                self.scheduler.set_timesteps(steps, device=device)
                data.timesteps, steps = self._get_timesteps(
                    steps, strength, device, image is None
                )
                latent_timestep = data.timesteps[:1].repeat(
                    batch_size * images_per_prompt
                )

                data.latents, init_latents_orig, noise = self._prepare_latents(
                    image,
                    latent_timestep,
                    batch_size * images_per_prompt,
                    width,
                    height,
                    dtype,
                    device,
                    generator,
                    latents,
                )
                if modifiers is not None:
                    for modifier in modifiers:
                        modifier.enter_context(gs, data)

                extra_kwargs = self._extra_kwargs(generator, eta)

                if self.controlnet is not None:
                    controlnet_keep = []
                    for i in range(len(data.timesteps)):
                        keeps = [
                            1.0
                            - float(
                                i / len(data.timesteps) < s
                                or (i + 1) / len(data.timesteps) > e
                            )
                            for s, e in zip(
                                controlnet_guidance_start, controlnet_guidance_end
                            )
                        ]
                        controlnet_keep.append(keeps[0] if len(keeps) == 1 else keeps)

                unet_task = progress_bar.add_task(
                    "[blue]UNet inference...", total=steps
                )
                for i, t in enumerate(data.timesteps):
                    data.latent_model_input = torch.cat(
                        [data.latents] * 2 if classifier_free_guidance else data.latents
                    )
                    data.latent_model_input = self.scheduler.scale_model_input(
                        data.latent_model_input, t
                    )
                    if modifiers is not None:
                        for modifier in modifiers:
                            data = modifier.pre_unet(data)
                    if self.controlnet is None:
                        data.noise_pred = self.unet(
                            latent_model_input=data.latent_model_input,
                            timesteps=t,
                            encoder_hidden_states=data.encoder_hidden_states,
                            width=width,
                            height=height,
                        )
                    elif image is not None:
                        if guess_mode and classifier_free_guidance:
                            control_model_input = data.latents
                            control_model_input = self.scheduler.scale_model_input(
                                control_model_input, t
                            )
                            controlnet_prompt_embeds = data.encoder_hidden_states.chunk(
                                2
                            )[1]
                        else:
                            control_model_input = data.latent_model_input
                            controlnet_prompt_embeds = data.encoder_hidden_states

                        if isinstance(controlnet_keep[i], list):
                            cond_scale = [
                                c * s
                                for c, s in zip(
                                    controlnet_conditioning_scale, controlnet_keep[i]
                                )
                            ]
                        else:
                            cond_scale = (
                                controlnet_conditioning_scale * controlnet_keep[i]
                            )

                        down_block_res_samples, mid_block_res_sample = self.controlnet(
                            control_model_input,
                            t,
                            encoder_hidden_states=controlnet_prompt_embeds,
                            controlnet_cond=image,
                            conditioning_scale=cond_scale,
                            guess_mode=guess_mode,
                            return_dict=False,
                        )

                        if guess_mode and classifier_free_guidance:
                            down_block_res_samples = [
                                torch.cat(
                                    [torch.zeros_like(d), d]
                                    for d in down_block_res_samples
                                )
                            ]
                            mid_block_res_sample = torch.cat(
                                [
                                    torch.zeros_like(mid_block_res_sample),
                                    mid_block_res_sample,
                                ]
                            )

                        data.noise_pred = self.unet(
                            latent_model_input=data.latent_model_input,
                            timesteps=t,
                            encoder_hidden_states=data.encoder_hidden_states,
                            width=width,
                            height=height,
                            down_block_res_samples=down_block_res_samples,
                            mid_block_res_sample=mid_block_res_sample,
                        )
                    else:
                        raise ValueError("Invalid state")

                    # perform guidance
                    if classifier_free_guidance:
                        (
                            data.noise_pred_uncond,
                            data.noise_pred_text,
                        ) = data.noise_pred.chunk(2)
                        data.noise_pred = data.noise_pred_uncond + guidance_scale * (
                            data.noise_pred_text - data.noise_pred_uncond
                        )

                    progress_bar.advance(unet_task)
                    if modifiers is not None:
                        for modifier in modifiers:
                            data = modifier.post_unet(data)
                    data.latents = self.scheduler.step(
                        data.noise_pred, data.timesteps, data.latents, **extra_kwargs
                    ).prev_sample
                    if mask is not None:
                        init_latents_proper = self.scheduler.add_noise(
                            init_latents_orig, noise, torch.tensor([data.timesteps])
                        )
                        data.latents = (init_latents_proper * mask) + (
                            data.latents * (1 - mask)
                        )

                    if i % callback_steps == 0 and callback is not None:
                        callback(i, t, data.latents)
                if output_type == "latent":
                    return data.latents, False

                image = self._decode_latents(data, modifiers)

                del data

                image = fromarray(image)

                if hasattr(self, "final_offload_hook"):
                    self.final_offload_hook.offload()  # type: ignore

                return image
