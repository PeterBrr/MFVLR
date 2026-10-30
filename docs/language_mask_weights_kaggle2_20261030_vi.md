# Chuyển tiếp đối chứng mask sang Kaggle2 — 30/10/2026

Yêu cầu: dùng quota Kaggle2 để tiếp tục hai lượt FW2/FW1 đang dở. MCP Kaggle2 và CLI dùng KAGGLE_API_TOKEN_2 đều liệt kê owner v5nhdf; tài khoản này không được đọc dataset riêng tư của nha4von. API quota: đã dùng94.008,2898821/108.000giây, còn13.991,7101179giây≈3,89giờ. Làm mới07:00ngày31/10/2026giờViệt Nam.

Đã chuẩn bị gói local4.360.714.707byte≈4,36GB tại `artifacts/language_mask_weights_kaggle2_20261030/package_001/upload/`, dataset đích dự kiến RIÊNG TƯ `v5nhdf/mfvlr-fw-resume-20261030`:

| File | Byte | Nội dung |
|---|---:|---|
| genface5000_data.bin | 625.716.293 | Gói train5.000/val400 và maskGT đã dùng; không cótest |
| project_fw2.bin | 1.482.424 | Source/assets đóng băng, configFW2 |
| project_fw1.bin | 1.482.424 | Source/assets đóng băng, configFW1 |
| resume_fw2.bin | 1.866.143.727 | Model/optimizer/RNG/schedule/history bước14.600 |
| resume_fw1.bin | 1.865.889.839 | Model/optimizer/RNG/schedule/history bước12.700 |

Đã đọc checkpoint thật, kiểm schema/step/config, hash toàn bộ source và dữ liệu. Source dựa bản đóng băng đang dùng, chỉ đổi time-budget8600→6300giâyFW2 và6600giâyFW1; chuẩn hóa key đường dẫn Windows→Linux trong provenance. Giữ tensor model/optimizer, RNG, lịchbatch/history và config không đổi. Hash input/output và source cũ/mới trong `migration_audit.json`. Không có token/credentials hoặc dữ liệu dự án khác trong gói.

Notebook chuẩn bị:
- `v5nhdf/mfvlr-fw2-resume-20261030-train`
- `v5nhdf/mfvlr-fw1-resume-20261030-train`

Mục tiêu15epoch, nhưng quota hiện tại có thể chỉ đủ tiến thêm một phần. Tổng ngân sách runner12.900giây, chừa khoảng1.092giây cho bootstrap và lưucheckpoint. FP32/TF32off, recipe và split giữ nguyên; không augmentation/test mới. Máy local không còn supervisor8236/worker24340 tại lúc kiểm tra; checkpointFW2 mới nhất14.600, log sau đó chỉ tới14.699 chưa lưu. Không tự chạy lại local khi đang chuẩn bị chuyển nền tảng.

## Trạng thái phê duyệt

Đã chuẩn bị đầy đủ gói và notebook, CHƯA upload/launch. Automatic approval review từ chối upload với lý do: gói private4,36GB là sensitive egress, quyền sở hữu đíchKaggle2 chưa được xác lập bằng bằng chứng tin cậy theo bộduyệt và chưa có xác nhận explicit của người dùng cho đúngpayload/đích. Không bypass bằng tool/token khác. Đã gửi câu hỏi xác nhận v5nhdf thuộc người dùng, đồng ý nội dung/đích riêng tư và chạy tiếp. Chờ câu trả lời trước bước phụ thuộc.

Script: `tools/migrate_mask_weights_kaggle2.py --action prepare|upload|launch`. Không chạy lại prepare vào package đã tồn tại; upload/launch có record chống trùng. Chưa có upload_record/launch_record, không báo đã chạy Kaggle2.

## Đã có xác nhận và bắt đầu upload

Người dùng trả lời: **"Xác nhận tài khoản và đồng ý tải lên, train tiếp"** cho câu hỏi nêu rõ v5nhdf, gói4,36GB, đích riêng tư và nội dung checkpoint/dữ liệu. Sau xác nhận, automatic approval chấp nhận upload; không còn chờ phê duyệt này.

Đã bắt đầu upload nhưng mạng dao động khoảng0,3–1,5MB/s trong đoạn đầu, không bảo đảm ETA. Chuyển sang pipeline nền có log, dừng uploader tiền cảnh cũ sau khi xác minh đúng process để tránh upload trùng. PipelinePID27032, lệnh `--action upload-and-launch`:
1. Kiểm hash từng file đúng migration_audit.
2. Tạo datasetprivate/upload bằng KAGGLE_API_TOKEN_2, xác minh ownerCLI v5nhdf.
3. Chờ datasetready, kiểm lại is_private=true và quota còn tối thiểu13.200giây.
4. Tự push hai notebookprivateGPU, mỗi lượt từ checkpoint đúng weight; ghi launch_record. Nếuquota thay đổi không đủ thì báoFAILED thay vì âm thầm chạy vượt ngân sách.

Theo dõi `pipeline_status.json` (phase), `pipeline.log` (sự kiện), `upload_progress.log` (thanh upload) trong artifact gốc. PhaseUPLOADING/WAITING_DATASET_READY chưa phải train; SUBMITTED chỉ xác nhận notebook đã gửi, cần logGPU/resume/train để xác nhận thực chạy. Không đóng máy trong khi gói local đang upload; sau khi notebook thực sự chạy trênKaggle thì không cần giữ client để tính toán trênserver.
