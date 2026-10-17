"""Full scratch FLT under approved L04--L13 engineering baseline.

MFVLR III-D pp. 6--7, IV-A p. 8 and tied vocabulary Eq. (24).
No loss, adapter, classifier, dataset or optimizer is implemented.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .language_encoder import (LanguageEncoderStage,finite,validate_hidden,
                               initialize_attention,initialize_linear)
from .language_decoder import LanguageDecoder


@dataclass(frozen=True)
class FLTOutput:
    low: Tensor
    encoder_hidden: Tensor
    global_language: Tensor
    decoder_hidden: Tensor
    logits: Tensor | None


class FineGrainedLanguageTransformer(LanguageEncoderStage):
    def __init__(self, *, seed: int = 0) -> None:
        # Build once, initialize once in the exact Wvoc,Pe,Pd,LE,LD draw order.
        nn.Module.__init__(self)
        self._validate_seed(seed)
        with torch.random.fork_rng(devices=[]):
            self._build_encoder_owner()
            self.decoder = LanguageDecoder()
            generator = torch.Generator(device="cpu").manual_seed(seed)
            self._initialize_encoder_owner(generator)
            for block in self.decoder.blocks:
                initialize_attention(block.self_attention,generator)
                initialize_attention(block.memory_attention,generator)
                initialize_linear(block.ff.fc1,generator)
                initialize_linear(block.ff.fc2,generator)
                initialize_attention(block.vim,generator)

    @classmethod
    def from_config(cls, config):
        config.require_approval()
        return cls(seed=config.data["initialization"]["seed"])

    @property
    def vocabulary_output_weight(self) -> nn.Parameter:
        """Same object as lookup; property registers no second parameter."""
        return self.vocabulary.weight

    def shift_low(self, low: Tensor) -> Tensor:
        validate_hidden(low,"low",self.vocabulary.weight)
        bos = self.vocabulary.weight[49406].view(1,1,512).expand(low.shape[0],1,512)
        return torch.cat((bos,low[:,:-1,:]),dim=1)

    def decode(self, low: Tensor, encoder_hidden: Tensor, visual: Tensor) -> Tensor:
        validate_hidden(low,"low",self.vocabulary.weight)
        validate_hidden(encoder_hidden,"encoder_hidden",self.vocabulary.weight)
        validate_hidden(visual,"visual",self.vocabulary.weight,length=1)
        if low.shape[0] != encoder_hidden.shape[0] or low.shape[0] != visual.shape[0]:
            raise ValueError("decoder input batches must match")
        shifted = finite(self.shift_low(low)+self.decoder_position,"LD.shift_position_sum")
        return self.decoder(shifted,encoder_hidden,visual)

    def project_vocabulary(self, hidden: Tensor) -> Tensor:
        validate_hidden(hidden,"decoder_hidden",self.vocabulary.weight)
        # Eq. (24) raw logits. No output bias, copied weight, norm or softmax.
        return finite(F.linear(hidden,self.vocabulary_output_weight,bias=None),"vocabulary.logits")

    def forward(self, ids: Tensor, visual: Tensor, *, return_logits: bool = True) -> FLTOutput:
        if type(return_logits) is not bool:
            raise TypeError("return_logits must be boolean")
        encoded = self.encode(ids,visual)
        decoded = self.decode(encoded.low,encoded.encoder_hidden,visual)
        logits = self.project_vocabulary(decoded) if return_logits else None
        return FLTOutput(encoded.low,encoded.encoder_hidden,encoded.global_language,decoded,logits)


def parameter_report(model: FineGrainedLanguageTransformer) -> dict:
    count=lambda module:sum(p.numel() for p in module.parameters())
    report={"vocabulary":model.vocabulary.weight.numel(),"encoder_position":model.encoder_position.numel(),
            "decoder_position":model.decoder_position.numel(),
            "encoder_blocks":[count(b) for b in model.encoder.blocks],
            "decoder_blocks":[count(b) for b in model.decoder.blocks],
            "vim_instances":len(model.encoder.blocks)+len(model.decoder.blocks),
            "vim_subtotal":sum(count(b.vim) for b in [*model.encoder.blocks,*model.decoder.blocks])}
    report["vim_qk_subtotal"]=sum(b.vim.q.weight.numel()+b.vim.k.weight.numel() for b in
                                 [*model.encoder.blocks,*model.decoder.blocks])
    parameters=list(model.parameters())
    report["parameter_tensors"]=len(parameters)
    report["total_unique"]=sum(p.numel() for p in parameters)
    report["unique_storage_elements"]=sum(p.numel() for p in
           {p.data_ptr():p for p in parameters}.values())
    return report
