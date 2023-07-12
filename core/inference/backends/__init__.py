from ..functions import is_aitemplate_available, is_onnx_available

from .base_backend import BaseVAEBackend, BaseUNetBackend, BaseCLIPBackend
from .pytorch import (
    DynamicPyTorchVAEBackend,
    DynamicPyTorchUNetBackend,
    DynamicPyTorchCLIPBackend,
)

if is_aitemplate_available():
    from .aitemplate.static import (
        StaticAITemplateVAEBackend,
        StaticAITemplateUNetBackend,
        StaticAITemplateCLIPBackend,
    )
    from .aitemplate.dynamic import (
        DynamicAITemplateVAEBackend,
        DynamicAITemplateUNetBackend,
        DynamicAITemplateCLIPBackend,
    )

if is_onnx_available():
    from .onnx import (
        DynamicOnnxVAEBackend,
        DynamicOnnxUNetBackend,
        DynamicOnnxCLIPBackend,
    )
