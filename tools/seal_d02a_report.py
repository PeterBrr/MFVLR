"""Editorial cleanup and consolidated actual JUnit evidence; no test execution."""
import copy
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d2_d02a'
p=ROOT/'docs/phase2d2_d02a_numerical_pilot_report.md'
s=p.read_text(encoding='utf-8')
# Fix joined number/word typography only in the final new report.
for a,b in {'All414':'All 414','original40':'original 40','interrupted407':'interrupted 407',
            'all1946':'all 1,946','all426':'all 426','and426':'and 426','separate407':'separate 407',
            'Phase2D.2':'Phase 2D.2','job407':'job 407','exit1073807364':'exit 1073807364',
            'All40':'All 40','only407':'only 407','between407':'between 407','and408':'and 408',
            'cut7':'cut 7','scale1024':'scale 1024'}.items():s=s.replace(a,b)
p.write_text(s,encoding='utf-8')
root=ET.Element('testsuites');sources=[];count=0
for label in ('oracles','driver_tests','summary_tests','ram_guard_tests','ram_guard_v2_tests'):
    source=OUT/label/'junit.xml'
    original=ET.parse(source).getroot()
    suites=list(original) if original.tag=='testsuites' else [original]
    for suite in suites:
        assert suite.tag=='testsuite'
        c=copy.deepcopy(suite);c.set('name',label+':'+c.get('name','pytest'))
        assert not c.findall('.//failure') and not c.findall('.//error') and not c.findall('.//skipped')
        count+=len(c.findall('testcase'));root.append(c)
    sources.append({'path':source.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
assert count==34
root.set('tests',str(count));root.set('failures','0');root.set('errors','0');root.set('skipped','0')
target=OUT/'verification_final/combined_recorded_junit.xml'
assert not target.exists()
ET.ElementTree(root).write(target,encoding='utf-8',xml_declaration=True)
with (target.with_suffix('.provenance.json')).open('x') as h:json.dump({
    'description':'Consolidated original JUnit testcases, not a new pytest run',
    'source_reports':sources,'cases':count,'earlier_repeated_v1_preflight_cases_not_double_counted':4,
    'original_tests_not_modified':True,'numerical_FAIL_comparisons_are_separate_from_helper_pytest':True},h,indent=2)
print(json.dumps({'final_report_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'consolidated_recorded_cases':count}))
