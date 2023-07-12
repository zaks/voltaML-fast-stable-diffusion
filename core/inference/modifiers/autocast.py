from contextlib import ExitStack

from ...config import config
from ...optimizations.autocast_utils import autocast
from ..modifier import DiffusionModifier, DiffusionData


class AutocastStep(DiffusionModifier):
    def __init__(self) -> None:
        super().__init__("Autocast")

    def setup(self, context: ExitStack, data: DiffusionData) -> DiffusionData:
        if config.api.autocast:
            context.enter_context(autocast(dtype=config.api.dtype))
        return data
