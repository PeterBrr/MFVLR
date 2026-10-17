"""External visual/language orchestration only; no heads, losses or training loop.

Plain caller object does not duplicate ownership of MVE, VD or FLT parameters.
The image-only route deliberately has no token/metadata dependency.
"""
from dataclasses import dataclass

from torch import Tensor

from .flt import FineGrainedLanguageTransformer, FLTOutput
from .mve import MVE, MVEOutput
from .vision_decoder import VisionDecoder, VisionDecoderOutput


@dataclass(frozen=True)
class VisualLanguageOutput:
    visual: MVEOutput
    decoded: VisionDecoderOutput
    language: FLTOutput | None


class VisualLanguageOrchestrator:
    def __init__(self,mve: MVE,decoder: VisionDecoder,flt: FineGrainedLanguageTransformer | None = None):
        self.mve=mve
        self.decoder=decoder
        self.flt=flt

    def visual_only(self,image: Tensor) -> VisualLanguageOutput:
        appearance=self.mve.encode_appearance(image)
        decoded=self.decoder.decode_appearance(appearance)
        visual=self.mve(image,decoded.reconstruction,appearance_encoding=appearance)
        return VisualLanguageOutput(visual,decoded,None)

    def with_language(self,image: Tensor,ids: Tensor,*,return_logits: bool=True) -> VisualLanguageOutput:
        if self.flt is None:
            raise RuntimeError('with_language requires an explicitly supplied FLT')
        result=self.visual_only(image)
        language=self.flt(ids,result.visual.fused_global,return_logits=return_logits)
        return VisualLanguageOutput(result.visual,result.decoded,language)
