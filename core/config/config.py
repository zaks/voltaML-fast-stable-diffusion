import logging
import multiprocessing
from dataclasses import Field, dataclass, field, fields
from typing import Dict, List, Literal, Union

from dataclasses_json import CatchAll, DataClassJsonMixin, Undefined, dataclass_json
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers

logger = logging.getLogger(__name__)


@dataclass
class Txt2ImgConfig:
    "Configuration for the text to image pipeline"

    width: int = 512
    height: int = 512
    seed: int = -1
    cfg_scale: int = 7
    sampler: int = KarrasDiffusionSchedulers.DPMSolverSinglestepScheduler.value
    prompt: str = ""
    negative_prompt: str = ""
    steps: int = 40
    batch_count: int = 1
    batch_size: int = 1
    self_attention_scale: float = 0.0


@dataclass
class Img2ImgConfig:
    "Configuration for the image to image pipeline"

    width: int = 512
    height: int = 512
    seed: int = -1
    cfg_scale: int = 7
    sampler: int = KarrasDiffusionSchedulers.DPMSolverSinglestepScheduler.value
    prompt: str = ""
    negative_prompt: str = ""
    steps: int = 40
    batch_count: int = 1
    batch_size: int = 1
    resize_method: int = 0
    denoising_strength: float = 0.6
    self_attention_scale: float = 0.0


@dataclass
class InpaintingConfig:
    "Configuration for the inpainting pipeline"

    prompt: str = ""
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 40
    cfg_scale: int = 7
    seed: int = -1
    batch_count: int = 1
    batch_size: int = 1
    sampler: int = KarrasDiffusionSchedulers.DPMSolverSinglestepScheduler.value
    self_attention_scale: float = 0.0


@dataclass
class ControlNetConfig:
    "Configuration for the inpainting pipeline"

    prompt: str = ""
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    seed: int = -1
    cfg_scale: int = 7
    steps: int = 40
    batch_count: int = 1
    batch_size: int = 1
    sampler: int = KarrasDiffusionSchedulers.DPMSolverSinglestepScheduler.value
    controlnet: str = "lllyasviel/sd-controlnet-canny"
    controlnet_conditioning_scale: float = 1.0
    detection_resolution: int = 512
    is_preprocessed: bool = False
    save_preprocessed: bool = False
    return_preprocessed: bool = True


@dataclass
class UpscaleConfig:
    "Configuration for the RealESRGAN upscaler"

    model: str = "RealESRGAN_x4plus_anime_6B"
    scale_factor: int = 4
    tile_size: int = field(default=128)
    tile_padding: int = field(default=10)


@dataclass
class APIConfig:
    "Configuration for the API"

    # Websockets and intervals
    websocket_sync_interval: float = 0.02
    websocket_perf_interval: float = 1.0

    # TomeSD
    use_tomesd: bool = False  # really extreme, probably will have to wait around until tome improves a bit
    tomesd_ratio: float = 0.25  # had to tone this down, 0.4 is too big of a context loss even on short prompts
    tomesd_downsample_layers: Literal[1, 2, 4, 8] = 1

    image_preview_delay: float = 2.0

    # General optimizations
    attention_processor: Literal["xformers", "spda", "cross_attention"] = "xformers"
    attention_slicing: Union[int, Literal["auto", "disabled"]] = "disabled"
    channels_last: bool = True
    vae_slicing: bool = True
    trace_model: bool = False
    clear_memory_policy: Literal["always", "after_disconnect", "never"] = "always"
    offload: Literal["module", "model", "disabled"] = "disabled"
    use_fp32: bool = False

    # CPU specific optimizations
    quantize_to_int8: bool = False  # preferably will also be able to port this over to gpu, but cpu only for now

    # CUDA specific optimizations
    reduced_precision: bool = False
    cudnn_benchmark: bool = False
    deterministic_generation: bool = False

    # Device settings
    device_id: int = 0
    device_type: Literal[
        "cpu", "cuda", "mps", "directml", "intel", "vulkan", "iree"
    ] = "cuda"
    iree_target: Literal["cuda", "vulkan", "llvm", "interpreted"] = "vulkan"

    # Lora
    lora_text_encoder_weight: float = 0.5
    lora_unet_weight: float = 0.5

    # Critical
    enable_shutdown: bool = True

    # Autoload
    autoloaded_loras: Dict[str, Dict] = field(default_factory=dict)
    autoloaded_textual_inversions: List[str] = field(default_factory=list)

    @property
    def device(self):
        "Return the device string"

        if self.device_type == "intel":
            from core.inference.functions import is_ipex_available

            if is_ipex_available():
                return "xpu"
            return "cpu"

        if self.device_type == "cpu":
            return "cpu"
        if self.device_type == "vulkan":
            return "vulkan"
        if self.device_type == "directml":
            import torch_directml  # pylint: disable=import-error

            return torch_directml.device()

        return f"{self.device_type}:{self.device_id}"


@dataclass
class AITemplateConfig:
    "Configuration for model inference and acceleration"

    num_threads: int = field(default=min(multiprocessing.cpu_count() - 1, 8))


@dataclass
class BotConfig:
    "Configuration for the bot"

    default_scheduler: KarrasDiffusionSchedulers = (
        KarrasDiffusionSchedulers.DPMSolverSinglestepScheduler
    )
    verbose: bool = False
    use_default_negative_prompt: bool = True


@dataclass
class InterrogatorConfig:
    "Configuration for interrogation models"

    # set to "Salesforce/blip-image-captioning-base" for an extra vram
    caption_model: str = "Salesforce/blip-image-captioning-large"
    visualizer_model: str = "ViT-L-14/openai"

    offload_captioner: bool = (
        False  # should net a very big vram save for minimal performance cost
    )
    offload_visualizer: bool = False  # should net a somewhat big vram save for a bigger performance cost compared to captioner

    chunk_size: int = 2048  # set to 1024 for lower vram usage
    flavor_intermediate_count: int = 2048  # set to 1024 for lower vram usage

    flamingo_model: str = "dhansmair/flamingo-mini"

    caption_max_length: int = 32


@dataclass
class FrontendConfig:
    "Configuration for the frontend"

    theme: Literal["dark", "light"] = "dark"


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class Configuration(DataClassJsonMixin):
    "Main configuration class for the application"

    # default_factory= instead of default= so we're python3.11 compatible

    txt2img: Txt2ImgConfig = field(default_factory=Txt2ImgConfig)
    img2img: Img2ImgConfig = field(default_factory=Img2ImgConfig)
    inpainting: InpaintingConfig = field(default_factory=InpaintingConfig)
    controlnet: ControlNetConfig = field(default_factory=ControlNetConfig)
    upscale: UpscaleConfig = field(default_factory=UpscaleConfig)
    api: APIConfig = field(default_factory=APIConfig)
    interrogator: InterrogatorConfig = field(default_factory=InterrogatorConfig)
    aitemplate: AITemplateConfig = field(default_factory=AITemplateConfig)
    bot: BotConfig = field(default_factory=BotConfig)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    extra: CatchAll = field(default_factory=dict)


def save_config(config: Configuration):
    "Save the configuration to a file"

    logger.info("Saving configuration to data/settings.json")

    with open("data/settings.json", "w", encoding="utf-8") as f:
        f.write(config.to_json(ensure_ascii=False, indent=4))


def update_config(config: Configuration, new_config: Configuration):
    "Update the configuration with new values instead of overwriting the pointer"

    for cls_field in fields(new_config):
        assert isinstance(cls_field, Field)
        setattr(config, cls_field.name, getattr(new_config, cls_field.name))


def load_config():
    "Load the configuration from a file"

    logger.info("Loading configuration from data/settings.json")

    try:
        with open("data/settings.json", "r", encoding="utf-8") as f:
            config = Configuration.from_json(f.read())
            logger.info("Configuration loaded from data/settings.json")
            return config

    except FileNotFoundError:
        logger.info("data/settings.json not found, creating a new one")
        config = Configuration()
        save_config(config)
        logger.info("Configuration saved to data/settings.json")
        return config
