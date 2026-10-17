"""Read-only verification/packaging of the one completed R2 attempt."""
import json,hashlib,zipfile,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/phase2d2_d02b7_r2';A=OUT/'attempt_01'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(name,v):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
result=read(OUT/'execution_record.json');analysis=read(A/'additivity.json');before=read(A/'inventory_before.json');after=read(A/'inventory_after.json')
assert result['execution']=='COMPLETE' and result['forwards']==result['backwards']==7
assert result['numerical']==analysis['status']=='FAIL' and before==after and len(before)==508
assert analysis['elementwise']['atol']==1e-6 and analysis['elementwise']['rtol']==1e-5
names={r['name'] for r in before};assert names==set(analysis['parameters'])
assert sum(__import__('math').prod(r['shape']) for r in before)==154156232
pins=read(OUT/'driver_manifest.json');assert all(sha(ROOT/p)==h for p,h in pins.items())
old=read(OUT/'prior_attempts_before.json');assert all(sha(ROOT/p)==h for p,h in old.items())
write('preservation_after.json',{'status':'PASS','source_pins_checked':len(pins),'prior_attempt_files_checked':len(old),'mismatches':[]})
wrapper=ROOT/'tools/d02b7_r2_run.py';raw=wrapper.read_bytes();scope=read(OUT/'retry_scope.json')
assert hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==scope['wrapper_sha256']
write('wrapper_hash_clarification.json',{'original_record_preserved':True,'original_digest_kind':'SHA256 of LF source text, not Windows on-disk bytes','original_text_sha256':scope['wrapper_sha256'],'actual_file_sha256':sha(wrapper),'normalized_text_matches':True,'difference':'CRLF serialization only; no code changed','execution_adapter':'unchanged D02B7 coordinator with exclusive OUT directory override'})
passes=('fd','lr','cmc','fl','ar','kl','total');counts={p:dict(allocated=0,none=0,allocated_zero=0,nonzero=0,vim_qk_allocated=0,vim_qk_none=0) for p in passes};binary={}
for name,p in analysis['parameters'].items():
 assert len(p['loss_pairs'])==15 and len(p['direction_actual'])==len(p['direction_reference'])==4
 for loss,g in p['passes'].items():
  c=counts[loss]
  c['allocated' if g['allocated'] else 'none']+=1
  if g['allocated']:
   assert g['finite'];path=A/'gradient_stream'/g['path'];assert path.stat().st_size==g['bytes'] and sha(path)==g['sha256']
   binary[path.relative_to(ROOT).as_posix()]=dict(sha256=g['sha256'],bytes=g['bytes'],kind='original FP32 gradient',parameter=name,loss=loss)
   c['allocated_zero' if g['nonzero']==0 else 'nonzero']+=1
  if '.vim.q.' in name or '.vim.k.' in name:
   c['vim_qk_allocated' if g['allocated'] else 'vim_qk_none']+=1
   if g['allocated']:assert g['nonzero']==0
 path=A/'gradient_stream'/p['sum_path'];assert sha(path)==p['sum_sha256']
 binary[path.relative_to(ROOT).as_posix()]=dict(sha256=p['sum_sha256'],bytes=path.stat().st_size,kind='ordered FP64 gradient sum',parameter=name)
for loss in passes:
 assert counts[loss]['allocated']+counts[loss]['none']==508
 assert len(read(A/f'{loss}_activations.json'))==63
 assert read(A/f'{loss}_pass.json')['state_unchanged']
 seal=read(A/f'{loss}_seal.json');assert seal['source_hashes']==pins
write('gradient_participation.json',counts)
write('excluded_binary_inventory.json',{'files':binary,'reason':'Original gradient and sum shards preserved on disk; omitted only from compact ZIP','total_bytes':sum(v['bytes'] for v in binary.values())})
losses=read(A/'losses.json');hist=read(ROOT/'artifacts/phase2d2_n31/g2/final_real/summary.json')
assert losses==hist['losses'];arithmetic=read(A/'live_arithmetic.json');assert arithmetic['all_exact']
write('live_evidence_audit.json',{'status':'PASS','basis':'Read-only verification of actual worker records and binary shards; no new model execution','parameter_tensors':508,'gradient_pass_records':508*7,'pair_records':508*15,'direction_pairs':508*4,'binary_files_verified':len(binary),'failed_parameters':sum(p['outside_tolerance_count']>0 for p in analysis['parameters'].values()),'outside_coordinates':sum(p['outside_tolerance_count'] for p in analysis['parameters'].values()),'historical_losses_exact':True,'arithmetic_exact':True,'source_and_prior_attempts_preserved':True})
suite=ET.Element('testsuite',name='D02B7-R2 recorded live additivity',tests='508',failures=str(sum(p['outside_tolerance_count']>0 for p in analysis['parameters'].values())),errors='0',skipped='0')
for name,p in analysis['parameters'].items():
 node=ET.SubElement(suite,'testcase',name=name,classname='D02B7.R2.actual_additivity')
 if p['outside_tolerance_count']:ET.SubElement(node,'failure',message=f"{p['outside_tolerance_count']} coordinates outside original tolerance").text='Execution COMPLETE; numerical FAIL; dependent work BLOCKED. No waiver.'
ET.ElementTree(suite).write(OUT/'live_additivity.junit.xml',encoding='utf-8',xml_declaration=True)
print(json.dumps({'counts':counts,'excluded_bytes':sum(v['bytes'] for v in binary.values()),'peak_rss_GiB':result['sampled_peak_rss_bytes']/1024**3}))
