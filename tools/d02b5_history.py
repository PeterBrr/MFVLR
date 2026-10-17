"""Read-only historical A0 F2 evidence comparison; no numerical execution."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mfvlr_research.d02b5.policy import write
OUT=ROOT/'artifacts/phase2d2_d02b5'
rows=[]
for precision in ('float32','float64'):
    historic=ROOT/f'artifacts/phase2d2_n31/n2/fixture2_{precision}_threads4.json'
    h=json.loads(historic.read_text())
    folder=OUT/'cells'/f'D02B5-F2-A0-{precision}'/'attempt_01'
    if not (folder/'result.json').exists():
        rows.append(dict(precision=precision,status='NOT RUN'));continue
    comparisons=[]
    for original in h['cross_batch']:
        live=json.loads((folder/f"cross_batch_sample{original['sample']}.json").read_text())
        for name,metrics in original['stages'].items():
            differences={k:dict(historical=v,current=live[name].get(k)) for k,v in metrics.items() if live[name].get(k)!=v}
            comparisons.append(dict(sample=original['sample'],stage=name,exact_historical_metrics=not differences,differences=differences))
    activations=json.loads((folder/'activations.json').read_text())
    rows.append(dict(precision=precision,historical_path=historic.relative_to(ROOT).as_posix(),comparisons=comparisons,activations_exact=activations==h['activations'],basis='Read-only comparison of submitted historical and current actual records'))
write(OUT/'a0_historical_comparison.json',rows)
