"""Append-only progress report aware of stopped coordinators and partial jobs."""
from collections import Counter
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
import psutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d2_d02a'


def sha(p):
    with p.open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()


def write(p,x):
    with p.open('x',encoding='utf-8') as h:json.dump(x,h,indent=2,allow_nan=False)


def main():
    now=datetime.now(timezone(timedelta(hours=7)))
    dest=OUT/'progress_snapshots'/now.strftime('%Y%m%d_%H%M%S');dest.mkdir(exist_ok=False)
    protocol=json.loads((OUT/'pilot_protocol.json').read_text());jobs=protocol['jobs']
    processes=[]
    for v in psutil.process_iter(['pid','cmdline']):
        if any(str(a).replace('\\','/').endswith(('tools/d02a/run.py','tools/continue_d02a_registered.py','tools/d02a_ram_guard_v2.py')) for a in v.info['cmdline'] or []):processes.append(v.info)
    rows=[];totals=Counter();groups=Counter();partial=[]
    for index,job in enumerate(jobs):
        p=OUT/'jobs'/job['id'];summary=p/'summary.json';fail=None;records=None
        if summary.exists():
            x=json.loads(summary.read_text());status='COMPLETED' if x['status']=='COMPLETED_LOCAL_DIAGNOSTIC' else 'HARD_STOP'
            fail=x['failed_comparisons'];records=x['records'];totals['numerical_FAIL']+=fail;totals['records']+=records
            groups[(job['family'],job['variant'],'jobs')]+=1;groups[(job['family'],job['variant'],'FAIL')]+=fail
        elif p.exists():
            status='IN_PROGRESS' if processes else 'INTERRUPTED';count=failed=0;bad=[]
            with (p/'comparisons.jsonl').open() as h:
                for number,line in enumerate(h,1):
                    try:r=json.loads(line)
                    except json.JSONDecodeError:bad.append(number);continue
                    count+=1;failed+=r.get('status')=='FAIL'
            partial.append({'index':index,'id':job['id'],'complete_partial_rows':count,'partial_FAILs':failed,'truncated_lines':bad,
                'original_files_kept_in_place':True,'files':{f.relative_to(ROOT).as_posix():{'sha256':sha(f),'bytes':f.stat().st_size} for f in p.rglob('*') if f.is_file()}})
        else:status='NOT_RUN'
        totals[status]+=1
        rows.append({'index':index,**job,'execution_status':status,'numerical_FAIL_comparisons':fail,'records':records})
    driver=json.loads((OUT/'driver_manifest_v2.json').read_text());driver_ok=all(sha(ROOT/n)==h for n,h in driver.items())
    tests=[]
    for name in ('oracles','driver_tests','summary_tests','ram_guard_tests','ram_guard_v2_tests'):
        cases=ET.parse(OUT/name/'junit.xml').getroot().findall('.//testcase')
        tests.append({'group':name,'cases':len(cases),'failures':sum(c.find('failure') is not None or c.find('error') is not None for c in cases),'skips':sum(c.find('skipped') is not None for c in cases),'exit_code':json.loads((OUT/name/'execution.json').read_text())['exit_code']})
    stopped=json.loads((OUT/'continuation_0122/stopped.json').read_text())
    coordinator=json.loads((OUT/'continuation_0122/registered_local.execution.json').read_text())
    events=json.loads((OUT/'history/poweroff_0407_20261013_inspection/system_events.json').read_text(encoding='utf-8-sig'))
    guard=[json.loads(s) for s in (OUT/'resource_guard_v2/events.jsonl').read_text().splitlines()]
    gates=[e for e in guard if e['event']=='RAM_GATE_PASS']
    assert all(e['available_RAM']>=e['minimum_RAM']==3*2**30 for e in gates)
    ram=psutil.virtual_memory().available
    snapshot={'time':now.isoformat(),'phase_status':'BLOCKED','registered':414,'counts':dict(totals),'live_coordinator_worker_guard_processes':processes,
        'RAM_available_bytes':ram,'minimum_RAM_before_launch_bytes':3*2**30,'RAM_gate_currently_satisfied':ram>=3*2**30,
        'source_driver_hashes_match':driver_ok,'partial_attempts':partial,'controller_exit':coordinator,'controller_stopped_record':stopped,
        'system_events':events,'cause_evidence':'System event1074 user-initiated power-off followed by6006 immediately after last partial row; no OOM diagnosis inferred',
        'latest_guard_event':guard[-1],'verified_recorded_gate_passes':len(gates),'tests':tests,
        'tests_read_from_existing_logs_not_rerun':True,'prior_replay_verified_interruptions':[84,121],
        'not_run':['MVE-T10','VD-T12','A1-D02B','full-model G2/NG2','G3/G4','I1','R2','optimizer','training','GPU','B8'],
        'remaining':['Preserve and reconcile partial407 before any replay','Require at least3GiB available RAM before each new worker','Finish registered407..413 without repeating completed jobs','Validate all414 identities, required outputs, source/input/state hashes','Final paired aggregation, report, preservation manifest and review ZIP']}
    write(dest/'progress.json',snapshot)
    with (dest/'job_coverage.csv').open('x',newline='',encoding='utf-8') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    p407=next(p for p in partial if p['index']==407)
    lines=['# Tiến độ MFVLR A1-D02A', '',f"Chốt lúc **{now.strftime('%d/%m/%Y %H:%M:%S')} — Asia/Saigon (UTC+07:00)**.",'',
        f"**{totals['COMPLETED']}/414 job đã hoàn tất ({totals['COMPLETED']/414:.2%}). Pilot hiện gián đoạn; không có coordinator, worker hoặc RAM guard đang chạy. Phase 2D.2 vẫn BLOCKED.**",'',
        '## Trạng thái đã xác minh','',
        f"- Job gián đoạn: **407**; chưa có summary hoặc mã thoát riêng của worker.",
        f"- Job chưa chạy: **{totals['NOT_RUN']}**, chỉ số **408–413**.",
        f"- Hard-stop số học được ghi trong summary: **{totals['HARD_STOP']}**; điều này không chứng minh các bước chưa chạy sẽ PASS.",
        f"- Các job hoàn tất có **{totals['records']:,}** dòng kết quả và **{totals['numerical_FAIL']:,}** so sánh FAIL.",
        f"- Riêng phần dang dở của 407 có **{p407['complete_partial_rows']:,}** dòng hoàn chỉnh, **{p407['partial_FAILs']:,}** FAIL; được bảo toàn và không cộng lặp vào tổng job hoàn tất.",
        '- Mã thoát 0 của worker không phải chấp nhận số học. Không thay đổi hoặc miễn trừ ngưỡng.', '',
        '| Nhóm | Biến thể | Job hoàn tất | So sánh FAIL |','|---|---|---:|---:|']
    for f in ('NLD','NATT'):
        for v in ('A0','P3','Q3'):lines.append(f"| {f} | {v} | {groups[(f,v,'jobs')]} | {groups[(f,v,'FAIL')]:,} |")
    lines+=['','## Gián đoạn và RAM','',
        'Windows ghi nhận sự kiện 1074 yêu cầu tắt máy và 6006 dừng dịch vụ log ngay sau bản ghi cuối của job 407. JSON đi kèm giữ thời điểm sự kiện. Bộ điều phối ghi mã thoát **1073807364**; không suy diễn đây là mã thoát riêng của worker hoặc OOM.',
        f"RAM khả dụng lúc chốt: **{ram/2**30:.3f} GiB**; mức tối thiểu trước mỗi worker mới: **3 GiB**. Không khởi chạy phép tính mới để tạo báo cáo này.",
        'Cổng RAM tự động bên ngoài driver được áp dụng từ job 277. Các lần khởi chạy trước đó chỉ có kiểm tra tại mốc tiếp tục thủ công; không mô tả hồi tố là đã được kiểm tra từng worker. Lỗi khởi tạo guard v1, bản sửa v2 và log từng lần cấp phép đều được bảo toàn.',
        'Hai gián đoạn trước tại job 84 và 121 đã được lưu riêng và replay đối chiếu chính xác. Job 407 chưa replay; tệp gốc vẫn nguyên vị trí, có SHA-256 trong progress.json.', '',
        '## Kiểm tra đã có bằng chứng','',
        f"Hash sáu tệp driver đã khóa vừa được kiểm tra: **{'PASS' if driver_ok else 'FAIL'}**. Không chạy lại regression hoặc mô hình để tạo bản tiến độ này.", '',
        '| Nhóm pytest | Cases | Failures | Skips | Exit code |','|---|---:|---:|---:|---:|']
    lines += [f"| {t['group']} | {t['cases']} | {t['failures']} | {t['skips']} | {t['exit_code']} |" for t in tests]
    lines += ['', 'Tổng **34 case duy nhất** trong các JUnit cuối của từng nhóm. Có thêm một lần preflight 4 case guard v1 đã PASS rồi được chạy lại sau chỉnh sửa bookkeeping; tổng 38 lượt case qua các lần chạy đó. Đây là kết quả đọc từ log, không phải vừa chạy lại, và không phải 276 regression case mới.',
        'MVE-T10 và VD-T12 không chạy trong pilot. Sự cố SGD VD-T12 lịch sử không được phê duyệt hồi tố. Tám thất bại A0 ban đầu và mọi N1/N2/N3/N3.1 policy vẫn giữ nguyên.', '',
        '## Việc còn lại','',
        '1. Hoàn tất bảo toàn và đối soát lượt 407 gián đoạn; chỉ replay theo quy trình phục hồi đã khóa khi đủ RAM.',
        '2. Hoàn tất 407–413, không lặp các job đã hoàn tất.',
        '3. Đối chiếu 414 ID và mọi output/hash bắt buộc; tổng hợp đủ block, cut, scale, precision và backend.',
        '4. Hoàn thiện báo cáo cuối, manifest bảo toàn và review ZIP. Chưa có bộ review cuối hoàn chỉnh.',
        '5. Dừng để người dùng review. Không mở full-model sweeps, optimizer, training, dữ liệu hoặc benchmark.', '',
        '## Cách theo dõi','',
        '- `job_coverage.csv`: đủ 414 ID; lọc cột `execution_status` để xem COMPLETED / INTERRUPTED / NOT_RUN.',
        '- `progress.json`: số liệu, hash phần dang dở, sự kiện tắt máy, gate RAM và trạng thái bộ điều phối.',
        '- `manifest.json` và `package_record.json`: SHA-256 của bộ file tiến độ.',
        '- Đây là bản chụp có thời điểm chốt, không phải bảng tự cập nhật. Các bản cũ vẫn nằm trong thư mục `progress_snapshots/`.', '',]
    (dest/'progress_report.md').open('x',encoding='utf-8').write('\n'.join(lines))
    files={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in dest.iterdir() if p.is_file()}
    write(dest/'manifest.json',{'time':now.isoformat(),'files':files,'generator_sha256':sha(Path(__file__))})
    archive=dest/'progress_review.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dest.iterdir()):
            if p!=archive:z.write(p,p.name)
    with zipfile.ZipFile(archive) as z:
        for name,r in files.items():assert hashlib.sha256(z.read(name)).hexdigest()==r['sha256']
    write(dest/'package_record.json',{'sha256':sha(archive),'bytes':archive.stat().st_size,'archive_verification':'PASS'})
    print(json.dumps({'directory':str(dest),'completed':totals['COMPLETED'],'numerical_FAILs':totals['numerical_FAIL'],'partial407_rows':p407['complete_partial_rows'],'partial407_FAILs':p407['partial_FAILs'],'RAM_GiB':ram/2**30,'ZIP_SHA256':sha(archive)},indent=2))


if __name__=='__main__':main()
