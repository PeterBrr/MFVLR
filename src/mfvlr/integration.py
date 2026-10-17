"""Approved P2D04/P2D16 single-owner graph (MFVLR Fig.4, III-B--E).

Engineering baseline only. No optimizer, dataset or cached graph/state outputs.
All accepted modules and mathematical formulations are reused unchanged.
"""
from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import Tensor,nn

from .mve import MVE,MVEOutput
from .vision_decoder import VisionDecoder,VisionDecoderOutput
from .flt import FineGrainedLanguageTransformer,FLTOutput
from .heads import DetectionHeads
from .objectives import SixObjectives,LossValues,floating,indices,finite


def pairing(sample_ids:Sequence[str],batch:int)->tuple[str,...]:
    if isinstance(sample_ids,(str,bytes)) or not isinstance(sample_ids,Sequence):
        raise TypeError('sample_ids must be an ordered sequence')
    values=tuple(sample_ids)
    if len(values)!=batch or any(type(x) is not str or not x.strip() for x in values) or len(set(values))!=batch:
        raise ValueError('sample_ids must be nonempty, unique and match the image batch')
    return values


@dataclass(frozen=True)
class VisualPredictions:
    visual:MVEOutput
    decoded:VisionDecoderOutput
    detection_logits:Tensor


@dataclass(frozen=True)
class LanguagePredictions:
    image:Tensor
    sample_ids:tuple[str,...]
    token_ids:Tensor
    predictions:VisualPredictions
    language:FLTOutput
    predicted_language:Tensor


@dataclass(frozen=True)
class SuppliedTargets:
    sample_ids:tuple[str,...]
    token_ids:Tensor
    labels:Tensor
    masks:Tensor
    sample_kinds:tuple[str,...]


def visual_route(mve,decoder,detector,image):
    floating(image,'image',(3,224,224))
    if image.min().item()<0 or image.max().item()>1:raise ValueError('image must be in [0,1]')
    appearance=mve.encode_appearance(image)
    decoded=decoder.decode_appearance(appearance)
    visual=mve(image,decoded.reconstruction,appearance_encoding=appearance)
    finite(visual.fused_global,'Iv')
    logits=finite(detector(visual.fused_global[:,0,:]),'detection_logits')
    return VisualPredictions(visual,decoded,logits)


class MFVLRIntegration(nn.Module):
    """One owner each: MVE, VD, head bundle, FLT, objective bundle.

`forward`/`visual_only` never call language/adapter/objectives. The explicitly
selected `with_language` route may be evaluated diagnostically in eval mode.
Records are per-call caller-owned tensors, not immutable tensor snapshots.
"""
    def __init__(self):
        super().__init__()
        self.mve=MVE(seed=0)
        self.decoder=VisionDecoder(skip_policy='bottleneck_only',seed=0)
        self.heads=DetectionHeads(seed=0)
        self.flt=FineGrainedLanguageTransformer(seed=0)
        self.objectives=SixObjectives()

    @classmethod
    def from_config(cls,path):
        from .integration_config import load_integration_config
        load_integration_config(path)
        return cls()

    def visual_only(self,image:Tensor)->VisualPredictions:
        return visual_route(self.mve,self.decoder,self.heads.detector,image)

    def forward(self,image:Tensor)->VisualPredictions:
        return self.visual_only(image)

    def with_language(self,image:Tensor,token_ids:Tensor,*,sample_ids:Sequence[str])->LanguagePredictions:
        floating(image,'image',(3,224,224))
        ordered=pairing(sample_ids,image.shape[0])
        indices(token_ids,image,'token_ids',(image.shape[0],308),49408)
        predictions=self.visual_only(image)
        iv=predictions.visual.fused_global
        language=self.flt(token_ids,iv,return_logits=True)
        adapted=self.heads.align_language(iv)
        return LanguagePredictions(image,ordered,token_ids,predictions,language,adapted)

    def compute_objectives(self,output:LanguagePredictions,targets:SuppliedTargets)->LossValues:
        if not isinstance(output,LanguagePredictions) or not isinstance(targets,SuppliedTargets):
            raise TypeError('objectives require language predictions and supplied paired targets')
        batch=output.image.shape[0]
        if pairing(targets.sample_ids,batch)!=output.sample_ids:
            raise ValueError('target sample ordering must match predictions')
        indices(targets.token_ids,output.image,'token_ids',(batch,308),49408)
        if not torch.equal(targets.token_ids,output.token_ids):raise ValueError('target token IDs must be original unshifted IDs')
        if output.language.logits is None:raise ValueError('full vocabulary logits required')
        return self.objectives(detection_logits=output.predictions.detection_logits,labels=targets.labels,
            language_logits=output.language.logits,token_ids=targets.token_ids,
            visual=output.predictions.visual.fused_global,global_language=output.language.global_language,
            mask_logits=output.predictions.decoded.mask_logits,masks=targets.masks,
            image=output.image,reconstruction=output.predictions.decoded.reconstruction,
            predicted_language=output.predicted_language,sample_kinds=targets.sample_kinds)


class ImageOnlyMFVLR(nn.Module):
    """Physical deployment owner; no FLT, adapter or objective is instantiated.

Detector-only initialization reproduces exactly the first accepted head draw.
This is the P2D04 deployment boundary, not a new architectural head choice.
"""
    def __init__(self):
        super().__init__()
        self.mve=MVE(seed=0)
        self.decoder=VisionDecoder(skip_policy='bottleneck_only',seed=0)
        with torch.random.fork_rng(devices=[]):
            self.detector=nn.Linear(512,2,bias=True,device='cpu',dtype=torch.float32)
            nn.init.xavier_uniform_(self.detector.weight,gain=1.,generator=torch.Generator(device='cpu').manual_seed(0))
            nn.init.zeros_(self.detector.bias)

    def forward(self,image:Tensor)->VisualPredictions:
        return visual_route(self.mve,self.decoder,self.detector,image)


def ownership_report(root:nn.Module)->dict:
    # remove_duplicate=False is intentional: detect registration aliases as well
    # as storage aliases, which default parameter enumeration hides.
    entries=list(root.named_parameters(remove_duplicate=False))
    names=[n for n,p in entries]
    if len({id(p) for n,p in entries})!=len(entries) or len({p.data_ptr() for n,p in entries})!=len(entries):
        raise AssertionError('duplicate registered parameter or storage owner')
    counts={name:sum(p.numel() for p in child.parameters()) for name,child in root.named_children()}
    total=sum(p.numel() for n,p in entries)
    expected=154156232 if isinstance(root,MFVLRIntegration) else 41135303
    if total!=expected:raise AssertionError('unique ownership/count mismatch')
    return {'total_unique':total,'parameter_tensors':len(entries),'registered_names':names,'child_counts':counts,
            'independent_registered_parameter_and_storage_owners':True,
            'vocabulary_tied':root.flt.vocabulary_output_weight is root.flt.vocabulary.weight if isinstance(root,MFVLRIntegration) else None,
            'classification':'DERIVED counts under approved ASSUMED engineering topology'}
