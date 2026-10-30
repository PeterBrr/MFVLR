# Khởi động pilot Diffae — 17/10/2026

**Cập nhật:** Đã bổ sung `train.py`, `evaluate.py`, loader file-backed, checkpoint/resume và config thực thi riêng `configs/diffae_pilot.json`. Xem [lệnh chạy hiện tại](diffae_pilot_run.md). Phần dưới giữ lại trạng thái khi chuẩn bị dữ liệu ban đầu; mục trainer chưa triển khai đã được thay thế bởi pipeline mới. Chưa chạy full-model training để xác nhận bộ nhớ/hội tụ.

Người dùng đã yêu cầu bắt đầu pilot theo generator và xác nhận `conda activate my_env`. Không có metadata nguồn/danh tính riêng cho từng generator. Đã thực hiện chuẩn bị dữ liệu; **chưa có model forward/backward, optimizer update hoặc checkpoint đã train**.

## Đã thực hiện

- Tìm đúng môi trường: `C:\Users\NHA\miniconda3\envs\my_env\python.exe`, PyTorch `2.7.1+cu118`, CUDA khả dụng.
- Pin tokenizer cũ trong môi trường không khớp: regex 2026.7.19 và wcwidth 0.8.2. Đã cài regex 2026.6.28 và wcwidth 0.2.13 vào `.pilot_deps` của repository, không thay đổi các package trong my_env. `verify_assets()` đã pass với PYTHONPATH `.pilot_deps;src`.
- Sửa cách kiểm kê: chỉ lấy real từ `images`, mask từ `masks`; bỏ staging ẩn, file rỗng và đường dẫn ảnh/mask không duy nhất. Không dùng code train của `D:\MFVLR` vì cấu hình ở đó thay đổi tokenizer và objective so với repository đang mở.
- Giữ membership của ba manifest phát hành, chọn ngẫu nhiên xác định seed 0 trong các dòng có ảnh/mask duy nhất. Chưa tự chia lại dataset hoặc tạo mask.

| Split | Real khả dụng theo tên/cặp | Diffae khả dụng theo tên/cặp | Real đã chọn | Diffae đã chọn |
|---|---:|---:|---:|---:|
| train | 8,080 | 44,074 | 2,500 | 2,500 |
| val | 1,030 | 5,551 | 200 | 200 |
| test | 891 | 5,541 | 200 | 200 |

Loại vì basename real khớp nhiều đường dẫn: train 16,103; val 1,963; test 1,932. Test còn 1 dòng thiếu ảnh. Không chọn ngẫu nhiên một đường dẫn để lấp chỗ trống. Những mẫu được chọn chưa đại diện toàn bộ phân phối real của bộ gốc.

**Kết quả decode hoàn tất:** 5,800/5,800 cặp đọc được, kích thước ảnh/mask khớp và đạt kiểm tra giá trị mask; không có lỗi decode. Phát hiện một cặp real train có nội dung RGB giống hệt: `CelebA_HQ/7384.jpg` và `CelebA_HQ/7385.jpg`. Không thấy exact decoded RGB trùng xuyên split trong tập chọn. Đã tạo `unique_checked_candidates.jsonl` loại bản thứ hai, còn **4,999 train (2,499 real + 2,500 fake), 400 val, 400 test**. `dedup_report.json` ghi việc loại; chưa bù mẫu hoặc chốt manifest train. Kiểm tra này không phát hiện hết near-duplicates hay rò rỉ danh tính. Mục tokenizer environment trong danh sách pending của decode report đã được xử lý ở `environment.json`.

## Artifact cục bộ

Trong `artifacts/diffae_pilot/`:

- `environment.json`: phiên bản runtime, kiểm tra tokenizer và RAM tại thời điểm kiểm tra.
- `preparation/report.json`: số lượng và SHA-256 các candidate manifest.
- `preparation/{train,val,test}.candidates.jsonl`: đường dẫn ảnh/mask gốc, nhãn manifest, generator, seed chọn mẫu; `subject_group=null`, trạng thái `CANDIDATE_ONLY`.
- `preparation/rejected.jsonl`: các dòng bị loại và nguyên nhân.
- `preparation/checked_candidates.jsonl`, `decode_report.json`: kết quả decode, dimensions, giá trị mask, SHA-256 file và nội dung RGB; kiểm tra exact duplicates trong tập chọn.
- `preparation/possible_source_overlap.json`: kiểm tra heuristic suffix số của tên Diffae, không phải identity mapping đã xác minh. Không thấy suffix số xuất hiện xuyên split trong các ứng viên được chọn; điều này không chứng minh identity-disjoint.

## Các điều kiện chưa đạt để train

1. RAM máy 15.71 GiB tổng, chỉ còn khoảng 0.55–0.91 GiB khả dụng trong các lần đo. Chưa cấp phát full A0 hay Adam; cần giải phóng RAM và profile batch 2 trước. Không tự đóng ứng dụng của người dùng.
2. Chưa có source/identity mapping và provenance mask. Không suy ra identity đã xác minh từ filename. Có thể tiếp tục chuẩn bị một pilot kỹ thuật theo split phát hành, nhưng không tuyên bố đánh giá subject-disjoint hay kết quả cross-generator đáng tin cậy.
3. Cần chốt phép chuyển ảnh/mask từ kích thước gốc về 224×224 và ánh xạ annotation chính thức. Kiểm tra mask binary không chứng minh đó là mask pixel manipulation đúng theo paper.
4. G01 gradient-additivity FAIL lịch sử vẫn còn; không sửa trạng thái acceptance bằng yêu cầu chạy pilot. Cần review giới hạn lỗi và đo resource/numerical gate trước optimizer.
5. Loader file-backed, trainer và checkpoint resume thực tế chưa triển khai. `configs/diffae_pilot_proposal.json` là proposal, không phải CLI chạy được. Không bật `training_mock.json` để giả lập đã có trainer.

## Kiểm chứng công cụ

`python tests/test_diffae_preparation.py` pass: chọn cặp duy nhất, bỏ staging, loại basename mơ hồ, chặn output trong dataset và annotation trùng xuyên split. Một fixture riêng kiểm tra công cụ decode từ chối real mask có foreground đã pass. Không chạy bộ test lịch sử có optimizer.

Chạy lại kiểm kê từ project root:

```powershell
conda activate my_env
$env:PYTHONPATH = "$PWD\.pilot_deps;$PWD\src"
python tools/prepare_diffae_pilot.py --root D:\BaiduNetdiskDownload --output artifacts/diffae_pilot/preparation
python tools/check_diffae_candidates.py artifacts/diffae_pilot/preparation
```

Các lệnh chỉ đọc ảnh gốc và ghi báo cáo/candidate trong workspace; không train hoặc resize dữ liệu gốc. Báo cáo sẽ được ghi lại nếu chạy cùng output.
