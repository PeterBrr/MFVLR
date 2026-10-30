# Mở rộng đối chứng FD: train1000 / validation400 — 26/10/2026

## Thiết kế chốt trước khi chạy

- Chỉ GenFace real + Diffae:1000 train cân bằng500/500,400 validation cân bằng200/200.
- Dùng đúng các split đã đóng băng; test không đóng gói/không dùng.
- Tập16 ảnh thí nghiệm trước nằm trong tập1000 train mới.
- So lambda_FD0,1 và1; lambda_AR=lambda_FL=1, class weight pixel[1,4].
- Ba seed0/1/2, mỗi seed có hai lượt cùng khởi tạo và lịch batch; giữa các seed
  thay cả khởi tạo CNN/transformer/decoder/detector và thứ tự batch.
- 5 epoch,250 cập nhật/epoch,1250 cập nhật/lượt,6 lượt nối tiếp: tổng7500 cập nhật.
- Microbatch2/effective batch4; Adam LR1e-4, weight decay0, FP32 deterministic,
  TF32 tắt, activation checkpointing, không augmentation.
- Kiến trúc giữ như thí nghiệm loss: skip concat, nearest resize-conv, AD tuyến
  tính, detach B trước RE; language tắt. Không đổi decoder hoặc thêm loss mới.
- Tái chạy cả hai lambda trên cùng môi trường Kaggle để so sánh, không dùng
  checkpoint16 ảnh hoặc kết quả local làm đối chứng thay thế.

## Đánh giá

- Metric train và validation riêng ở cuối từng epoch;4 ảnh preview cố định.
- MSE AD, AUC/accuracy/F1 detection, IoU/Dice foreground macro trên fake,
  false-positive area trên real, precision/recall pixel và global2-class mIoU.
- Detection threshold0,5 và mask argmax cố định; không chọn ngưỡng theo test.
- Kết quả epoch5 là chính, không chọn epoch đẹp nhất sau khi nhìn validation.
- Tổng hợp từng seed, trung bình và độ lệch chuẩn mẫu; ưu tiên chênh lệch
  theo cặp cùng seed. Ba seed vẫn là số lượng ít, không khẳng định ý nghĩa
  thống kê chỉ từ dấu chênh lệch hoặc giá trị trung bình.
- Probe gradient AR/FL/FD đầu/cuối trên cùng4 cặp trong16 ảnh train cố định.
  Không dùng gradient trên validation để cập nhật. Phép đo không bao phủ
  mọi batch hoặc toàn bộ thời gian train.
- Lưu final checkpoint riêng từng lượt, schema `mfvlr-fd-validation-pilot-v1`.

## Dữ liệu và kiểm tra đã chuẩn bị

Gói local: `artifacts/genface_fd_validation_kaggle_20261026/package_001/`.

Đã decode/kiểm tra hash ảnh và GT qua reader hiện tại. RGB float32 chia255
được đóng gói uint8 và phục hồi đúng bit; GT binary giữ nguyên. Không trùng
sample ID hoặc nội dung RGB sau resize giữa train và validation. Mapping danh
tính/subject vẫn chưa xác minh; đây là pilot nội bộ, không claim subject-disjoint.

Gói binary162.591.522 byte gồm1400 ảnh đã resize/GT và source cần thiết; không
có credentials, dữ liệu test hoặc đường dẫn cá nhân của máy. Bundle SHA256:
`9bc64eb4e32c1dea5d15a08fe34de6df9a7048c510e3dd84d17e8295e928f073`.

Đã kiểm tra factory seed: seed0 khớp model hiện tại, lặp cùng seed khớp đúng
hash, seed0/1/2 tạo ra ba khởi tạo khác nhau. Script không chỉ đổi torch.manual_seed
vì các module hiện tại còn có generator seed riêng.

Dataset đích **riêng tư mới**: `nha4von/mfvlr-genface-fd-pilot-20261026`.
Notebook đích **riêng tư mới**: `nha4von/mfvlr-genface-fd-pilot-20261026-001`.
GPU T4, Internet tắt, timeout3 giờ; không sửa dataset/notebook của thí nghiệm16 ảnh.

## Tệp thực thi

- `tools/run_fd_validation_pilot.py`: train, validation và gradient,6 lượt.
- `tools/prepare_fd_validation_pilot.py`: cố định tensor và audit nguồn/split.
- `tools/launch_fd_validation_pilot.py`: prepare/upload/launch qua CLI chính thức.
- `tools/watch_kaggle_fd_logs.py`: stream log qua SSE và ghi file local.

Lệnh theo dõi sau khi notebook được khởi chạy:

```powershell
& 'C:\Users\NHA\miniconda3\python.exe' -u tools/watch_kaggle_fd_logs.py --kernel nha4von/mfvlr-genface-fd-pilot-20261026-001 --output artifacts/genface_fd_validation_kaggle_20261026/remote_live.log
```

Trạng thái: đã đóng gói và tạo notebook local. Kiểm tra CPU một cập nhật cho
cả FD0,1/FD1 với train/validation riêng đã hoàn tất; hash cặp khởi tạo và source
đều được kiểm tra. Factory seed0/1/2 đã được kiểm tra độc lập.

Lệnh upload bị bộ duyệt tự động từ chối: đồng ý mở rộng thí nghiệm chưa được
coi là xác nhận rõ payload mới1400 ảnh/GT/source162,6MB và đích dataset mới.
Đã hỏi người dùng xác nhận đúng gói, đích riêng tư và6 lượt train. Người dùng
đã trả lời **"Đồng ý tải lên và chạy"**. Lệnh upload được chạy lại theo đúng
xác nhận và tải lên thành công. Dataset ID12871469, version1, Ready,
`is_private=true`, đúng162.591.522 byte.

Notebook đã khởi chạy version1, kernel ID140997159, được xác minh private=true,
GPU T4 bật, Internet=false, đính kèm đúng dataset mới. Log thực tế xác nhận
PyTorch2.11.0+cu128/Tesla T4, source/data hashes hợp lệ, train_count1000,
val_count400, test_used=false. Lượt seed0/FD0,1 đã bắt đầu cập nhật CUDA.
Chưa có kết quả cuối; không coi smoke CPU là kết quả pilot.

Đã quan sát mốc đầu tiên: seed0/FD0,1, epoch1, step250, metric train count1000
và validation count400 tách biệt. Validation AUC0,7144, IoU fake0,00403115,
MSE0,00807955. Đây là mốc sớm của lượt đầu, không dùng làm kết luận cuối.

Notebook:
https://www.kaggle.com/code/nha4von/mfvlr-genface-fd-pilot-20261026-001

Log local đang ghi liên tục:
`artifacts/genface_fd_validation_kaggle_20261026/remote_live.log`.

```powershell
Get-Content "D:\New folder (2)\MFVLR\artifacts\genface_fd_validation_kaggle_20261026\remote_live.log" -Tail 20 -Wait
```

`tools/read_fd_validation_status.py` tạo snapshot tiến độ từ log thành
`live_status.json`. Khi chạy xong, `tools/collect_fd_validation_results.py`
tải report/preview về và tổng hợp epoch5 theo seed cùng chênh lệch theo cặp;
không chạy collector khi report chưa hoàn tất.
