"""Read-only progress snapshot; does not construct models or run numerical jobs."""
from collections import Counter
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/phase2d2_d02a'


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def main():
    now = datetime.now(timezone(timedelta(hours=7)))
    dest = OUT / 'progress_snapshots' / now.strftime('%Y%m%d_%H%M%S')
    dest.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((OUT / 'pilot_protocol.json').read_text())
    # Freeze the membership before reading to avoid mixing subsequently completed jobs.
    completed = {p.parent.name: p for p in (OUT / 'jobs').glob('*/summary.json')}
    started = {p.name for p in (OUT / 'jobs').iterdir() if p.is_dir()}
    coverage = []
    summaries = []
    groups = Counter()
    failures = Counter()
    for job in protocol['jobs']:
        row = dict(job)
        if job['id'] in completed:
            result = json.loads(completed[job['id']].read_text())
            summaries.append(result)
            state = 'COMPLETED' if result['status'] == 'COMPLETED_LOCAL_DIAGNOSTIC' else 'HARD_STOP'
            row.update(execution=state, numerical_FAIL_comparisons=result['failed_comparisons'], records=result['records'])
            groups[(job['family'], job['variant'], state)] += 1
            failures[(job['family'], job['variant'])] += result['failed_comparisons']
        else:
            row.update(execution='IN_PROGRESS_OR_INTERRUPTED' if job['id'] in started else 'NOT_RUN', numerical_FAIL_comparisons=None, records=None)
        coverage.append(row)
    tests = []
    for name in ('oracles', 'driver_tests', 'summary_tests'):
        cases = ET.parse(OUT / name / 'junit.xml').getroot().findall('.//testcase')
        execution = json.loads((OUT / name / 'execution.json').read_text())
        tests.append(dict(group=name, cases=len(cases), failures=sum(c.find('failure') is not None or c.find('error') is not None for c in cases), skips=sum(c.find('skipped') is not None for c in cases), exit_code=execution['exit_code'], verification='Read previously recorded JUnit/logs; not rerun for this snapshot'))
    recoveries = []
    for name in ('poweroff_20261013', 'interruption_0121_20261013'):
        folder = OUT / 'history' / name
        recoveries.append(dict(folder=name, recovery=json.loads((folder / 'recovery_record.json').read_text()), replay=json.loads((folder / 'replay_comparison.json').read_text()), archive=json.loads((folder / 'archive_verification.json').read_text())))
    drivers = json.loads((OUT / 'driver_manifest_v2.json').read_text())
    driver_match = all(sha(ROOT / name) == digest for name, digest in drivers.items())
    processes = []
    for proc in psutil.process_iter(['pid', 'cmdline']):
        cmd = proc.info['cmdline'] or []
        if any(str(v).replace('\\', '/').endswith(('tools/d02a/run.py', 'tools/continue_d02a_registered.py')) for v in cmd):
            processes.append(proc.info)
    count = sum(r['execution'] == 'COMPLETED' for r in coverage)
    hard = sum(r['execution'] == 'HARD_STOP' for r in coverage)
    active = [r['id'] for r in coverage if r['execution'] == 'IN_PROGRESS_OR_INTERRUPTED']
    nfail = sum(r['failed_comparisons'] for r in summaries)
    resources = [r['resources'] for r in summaries]
    state = dict(snapshot_time=now.isoformat(), timezone='Asia/Saigon / UTC+07:00',
        phase='A1-D02A local numerical pilot', phase_2D2_status='BLOCKED',
        registered=414, completed=count, hard_stops=hard, active_directories=active,
        not_started=sum(r['execution'] == 'NOT_RUN' for r in coverage),
        numerical_failed_comparisons=nfail, recorded_rows=sum(r['records'] for r in summaries),
        current_available_RAM_bytes=psutil.virtual_memory().available,
        peak_sampled_worker_RSS_bytes=max(r['sampled_peak_rss_bytes'] for r in resources),
        summed_completed_worker_seconds=sum(r['elapsed_seconds'] for r in resources),
        running_processes=processes, locked_driver_hashes_match=driver_match,
        protocol_sha256=sha(OUT/'pilot_protocol.json'), driver_manifest_sha256=sha(OUT/'driver_manifest_v2.json'),
        pytest_records=tests, recoveries=recoveries,
        historical_regression_cases_freshly_run=0, legacy_MVE_T10='NOT RUN', legacy_VD_T12='NOT RUN',
        full_model_sweeps_training_GPU_B8='NOT RUN',
        note='Partial snapshot only. Completed execution does not imply numerical acceptance. Partial interrupted-attempt FAILs are additional historical evidence, not double-counted in canonical totals.')
    write(dest/'progress.json', state)
    with (dest/'job_coverage.csv').open('x', newline='', encoding='utf-8') as handle:
        writer=csv.DictWriter(handle, fieldnames=list(coverage[0])); writer.writeheader(); writer.writerows(coverage)
    lines=[
        '# Tiến độ MFVLR A1-D02A — pilot số học decoder', '',
        f"Thời điểm chốt: **{now.strftime('%d/%m/%Y %H:%M:%S')} (Asia/Saigon, UTC+07:00)**.", '',
        f"**Đã hoàn tất {count}/414 job local. Phase 2D.2 vẫn BLOCKED.** Đây là báo cáo giữa chừng, chưa phải báo cáo chấp nhận hoặc kết luận chọn biến thể.", '',
        '## Tình trạng thực thi', '',
        f"- Job có hard-stop đã ghi nhận: **{hard}**.",
        f"- Thư mục job đang dang dở: `{', '.join(active) or 'không có tại thời điểm chốt'}`.",
        f"- Chưa khởi chạy: **{state['not_started']}** job.",
        f"- Đã ghi **{state['recorded_rows']:,}** dòng kết quả trong các job hoàn tất, gồm **{nfail:,}** so sánh số học FAIL.",
        '- FAIL số học được giữ nguyên. Mã thoát 0 chỉ cho biết worker hoàn thành các phép đo và không vi phạm điều kiện hard-stop; không biến các so sánh FAIL thành PASS.', '',
        '| Nhóm | Biến thể | Job hoàn tất | So sánh FAIL |', '|---|---|---:|---:|',
    ]
    for family in ('NLD','NATT'):
        for variant in ('A0','P3','Q3'):
            lines.append(f"| {family} | {variant} | {groups[(family,variant,'COMPLETED')]} | {failures[(family,variant)]:,} |")
    lines += ['', '## Kiểm tra và bảo toàn', '',
        f"Manifest sáu tệp driver v2 vừa được kiểm tra lại: **{'PASS' if driver_match else 'FAIL'}**. Toàn bộ 1.594 tệp nền được xác minh không đổi trong lần khôi phục gần nhất; báo cáo này không tuyên bố vừa chạy lại toàn bộ kiểm tra 1.594 tệp.", '',
        'Các kết quả pytest dưới đây được đọc từ JUnit/log đã ghi trong pilot; không chạy lại để tạo bản chụp tiến độ:', '',
        '| Nhóm | Cases | Failures | Skips | Exit code |', '|---|---:|---:|---:|---:|']
    lines += [f"| {t['group']} | {t['cases']} | {t['failures']} | {t['skips']} | {t['exit_code']} |" for t in tests]
    lines += ['', '**26 case pytest đã có bằng chứng PASS; không phải 276 regression case mới chạy.** MVE-T10 và VD-T12 không được thực thi trong pilot này. Tám thất bại A0 gốc và sự cố SGD VD-T12 lịch sử vẫn được giữ nguyên, không được miễn trừ hoặc phê duyệt hồi tố.', '',
        '## Các lượt chạy gián đoạn đã bảo toàn', '',
        '| Job | Dòng trước gián đoạn | FAIL trước gián đoạn | Sai khác khi replay | Nguyên nhân |', '|---|---:|---:|---:|---|']
    for r in recoveries:
        a=r['recovery']; b=r['replay']
        cause='Windows ghi nhận tắt máy' if a['index']==84 else 'Chưa xác định; không suy diễn OOM hoặc lỗi số học'
        lines.append(f"| {a['index']} | {a['complete_partial_rows']} | {a['partial_attempt_failed_comparisons']} | {len(b['differing_records'])} | {cause} |")
    lines += ['', 'Replay giữ nguyên input/state/provenance và so sánh mọi trường toán học; chỉ bỏ qua thời gian thực thi. Tệp gốc được lưu theo byte và kiểm tra SHA-256. Các FAIL của lượt dở dang vẫn có trong kho lịch sử, không bị xóa hoặc cộng lặp vào tổng job chuẩn.', '',
        '## Tham số và tài nguyên', '',
        '| Biến thể | LD | FLT | Toàn mô hình (suy ra) |', '|---|---:|---:|---:|',
        '| A0 | 36.746.752 | 112.758.272 | 154.156.232 |',
        '| P3 | 36.768.256 | 112.779.776 | 154.177.736 |',
        '| Q3 | 36.768.256 | 112.779.776 | 154.177.736 |', '',
        'Các số trên thuộc inventory chuẩn bị của pilot; số toàn mô hình cộng thêm phần visual/heads/objectives đã chấp nhận, không phải phép chạy six-loss toàn mô hình mới.', '',
        f"- Peak RSS worker đã lấy mẫu: **{state['peak_sampled_worker_RSS_bytes']/2**30:.3f} GiB** (lấy mẫu 50 ms; không bảo đảm bắt được đỉnh tức thời).",
        f"- RAM khả dụng lúc chốt: **{state['current_available_RAM_bytes']/2**30:.3f} GiB**.",
        f"- Tổng thời gian worker đã hoàn tất: **{state['summed_completed_worker_seconds']/3600:.2f} giờ**; không phải thời gian đồng hồ của riêng lượt tiếp tục này.",
        '- Máy dùng chung; không suy ra xếp hạng hiệu năng giữa các biến thể từ các thời gian này. GPU và B8: NOT RUN.', '',
        '## Việc còn lại', '',
        '1. Hoàn tất các ô NLD/NATT và đối chứng 1 luồng còn lại trong lịch đã khóa; không chạy lại ô hoàn tất.',
        '2. Tổng hợp đủ bảy LD block, tất cả cut/scale/backend, bao gồm trường hợp yếu và gradient gần 0.',
        '3. Kiểm tra báo cáo cuối, SHA-256 bảo toàn và đóng gói review ZIP hoàn chỉnh.',
        '4. Dừng để review; không mở A1-D02B, G2/NG2, G3/G4, optimizer hoặc training.', '',
        'Bản JSON đi kèm chứa provenance, trạng thái tiến trình và chi tiết khôi phục; CSV liệt kê đủ 414 ô. Các bản interim_report_01/02 là kết quả tổng hợp cũ hơn, không đại diện cho toàn bộ lịch hiện tại.', '',
        'Báo cáo này chỉ đọc bằng chứng và không thay đổi mô hình, fixture, ngưỡng hay numerical policy. Lịch chạy nền tiếp tục sau thời điểm chốt.', '',
    ]
    (dest/'progress_report.md').open('x', encoding='utf-8').write('\n'.join(lines))
    entries={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(dest.iterdir()) if p.is_file()}
    write(dest/'manifest.json', {'snapshot':now.isoformat(),'files':entries,'snapshot_generator_sha256':sha(Path(__file__))})
    archive=dest/'progress_review.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dest.iterdir()):
            if p!=archive:z.write(p,p.name)
    with zipfile.ZipFile(archive) as z:
        for name,entry in entries.items():
            assert hashlib.sha256(z.read(name)).hexdigest()==entry['sha256']
    write(dest/'package_record.json', {'sha256':sha(archive),'bytes':archive.stat().st_size,'archive_entry_verification':'PASS'})
    print(json.dumps({'directory':str(dest),'completed':count,'registered':414,'failed_comparisons':nfail,'hard_stops':hard,'active':active,'archive_sha256':sha(archive)},indent=2))


if __name__=='__main__':
    main()
