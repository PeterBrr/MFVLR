"""G2 synthetic oracle masks, not dataset mask preprocessing (P2D15)."""
from pathlib import Path
from dataclasses import dataclass
import hashlib
import json
import torch
from mfvlr.integration import SuppliedTargets

ROOT=Path(__file__).resolve().parents[1]
CASES={'real':(('real',17,0),),'efs':(('efs',412,1),),'local':(('local',997,5),),
       'real_efs':(('real',17,0),('efs',412,1)),
       'real_local':(('real',17,0),('local',997,5)),
       'local_duplicate':(('local',997,5),('local',412,5))}


def tensor_hash(tensor):return hashlib.sha256(memoryview(tensor.detach().contiguous().numpy()).cast('B')).hexdigest()


@dataclass(frozen=True)
class SyntheticFixture:
    image:torch.Tensor
    targets:SuppliedTargets
    provenance:dict


def fixture(name):
    golden_path=ROOT/'tests/foundation/fixtures/golden_tokens.json'
    golden=json.loads(golden_path.read_text())['prompt_cases']
    rows=CASES[name];images=[];masks=[];labels=[];ids=[];descriptions=[];sample_ids=[]
    for index,(kind,seed,case) in enumerate(rows):
        image=.1+.8*torch.rand(3,224,224,generator=torch.Generator().manual_seed(seed))
        mask=torch.zeros(224,224,dtype=torch.int64)
        if kind=='efs':mask.fill_(1)
        if kind=='local':
            image[:,64:160,80:176]=1-image[:,64:160,80:176];mask[64:160,80:176]=1
        sample_id=f'{name}:{index}:{kind}:{seed}'
        images.append(image);masks.append(mask);labels.append(0 if kind=='real' else 1)
        ids.append(golden[case]['ids']);sample_ids.append(sample_id)
        descriptions.append({'sample_id':sample_id,'kind':kind,'seed':seed,'golden_case':case,
                             'prompt_metadata':golden[case]['metadata'],'prompts':golden[case]['prompts'],
                             'image_sha256':tensor_hash(image),'mask_sha256':tensor_hash(mask)})
    image=torch.stack(images);token_ids=torch.tensor(ids,dtype=torch.int64);mask=torch.stack(masks)
    targets=SuppliedTargets(tuple(sample_ids),token_ids,torch.tensor(labels,dtype=torch.int64),mask,tuple(r[0] for r in rows))
    provenance={'name':name,'batch':len(rows),'samples':descriptions,'image_sha256':tensor_hash(image),
                'token_ids_sha256':tensor_hash(token_ids),'mask_sha256':tensor_hash(mask),'token_ids':ids,
                'golden_sha256':hashlib.sha256(golden_path.read_bytes()).hexdigest(),
                'rectangle':[64,160,80,176],'mask_is_synthetic_oracle':True,'dataset_preprocessing':False,
                'base_RGB_range':[.1,.9],'local_operation':'1-I inside rectangle',
                'local_duplicate_resolution':'second local uses seed 412; distinct image perturbation, same golden DiffFace IDs',
                'prompt_unique_count':len({tuple(row) for row in ids})}
    return SyntheticFixture(image,targets,provenance)
