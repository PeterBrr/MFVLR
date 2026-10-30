# GenFace 5.000 ảnh, 30 epoch trên Kaggle

Theo yêu cầu mở rộng: seed 0 trước, 5.000 train (2.500 thật + 2.500 Diffae), giữ nguyên 400 validation (200/200), 30 epoch × 1.250 cập nhật = 37.500 cập nhật. Không dùng test. Không chạy seed 1/2 trong job này.

Giữ baseline: Adam LR=1e-4, weight decay=0, effective batch4/microbatch2, FP32 deterministic, không augmentation; FD0,1 + AR1 + FL1; trọng số pixel nền/giả=[1,4]; skip concat, nearest resize-conv, AD tuyến tính, detach trước RE, không ngôn ngữ. Khởi tạo mới seed0, không warm-start từ checkpoint cũ. Dữ liệu và tổng cập nhật cùng tăng, nên không thể quy riêng cải thiện cho mỗi yếu tố.

Manifest train cũ có 4.999 ảnh sau loại trùng (2.499 thật, 2.500 giả). Bổ sung một ảnh thật hợp lệ từ `train.txt` gốc, kiểm tra ánh xạ tên không mơ hồ, mask rỗng/cùng kích thước, decoded hash không trùng train/val/test. Giữ nguyên các ảnh cũ, chứa tập train1.000 trước đó. Đóng gói kiểm tra lại nội dung sau resize và so sánh thứ tự/hash validation với gói cũ. Không đóng gói ảnh test. Subject-disjoint vẫn chưa xác minh.

Dataset riêng tư dự kiến: `nha4von/mfvlr-genface5000-epoch30-20261027`.
Notebook riêng tư: `nha4von/mfvlr-genface5000-epoch30-20261027-train`.
Artifact: `artifacts/genface5000_epoch30_kaggle_20261027/`.

Mỗi epoch đánh giá đầy đủ train/validation, xuất preview và ghi checkpoint `last.pt` bằng file tạm rồi thay thế atomically. Checkpoint chứa model, optimizer, bước/epoch, RNG torch CPU/CUDA, lịch batch, history/evaluations và hash dữ liệu/source. Chỉ giữ checkpoint đầy đủ mới nhất để tiết kiệm dung lượng; kết thúc thêm final.pt model-only. Đây là trạng thái lưu trên đĩa phiên Kaggle, không phải cam kết đã sao lưu ra ngoài phiên nếu máy chủ bị hủy đột ngột. Runner hiện chưa có tùy chọn CLI resume tự động; cần bộ nạp đúng schema nếu phải mở phiên tiếp theo.

Kiểm tra CPU một cập nhật với tùy chọn checkpoint đã hoàn thành; đã nạp lại checkpoint và kiểm tra optimizer/RNG/schedule. Không coi smoke là kết quả chất lượng. Không thay kiến trúc hoặc công thức loss.

Timeout dự kiến 10 giờ, ước lượng 6–8 giờ từ tốc độ cũ nhưng sẽ đo lại trên 5.000 ảnh. Quota kiểm tra trước launch còn khoảng 25,9 giờ GPU. Epoch30 là mốc báo cáo chính; không chọn epoch theo test.

Đóng gói đã hoàn tất: 625.716.293 byte; SHA256 bundle `1a372662af9f3e9f3208baef47fea5fda0eb9d218ea403dad6c57771c21bdc1b`.
NPZ train SHA256 `e234984a1a30a932e08ec636a938af94e8ae5b99d0dcb71773512583ffd66534`; validation SHA256 giữ nguyên `16f60fd9c3764c464822499e46ec1ae9596378c78aee03045c7d1dca8064f073`.
Ảnh thật bổ sung: CelebA_HQ/3692.jpg từ train.txt gốc. Train mới chứa đủ 1.000 ảnh cũ. Không trùng ID hoặc RGB sau resize trong train và giữa train/validation.

Upload thành công: dataset ID12888170 version1, READY, is_private=true, đúng 625.716.293 byte. Kaggle không cho notebook dùng cùng slug đã có dataset, nên lần tạo ban đầu trả Conflict; đã đổi tên notebook có hậu tố `-train`, không thay dữ liệu/cấu hình.

Notebook khởi chạy version1, kernel ID141122832, private=true, T4, Internet=false, gắn đúng dataset. MCP xác nhận RUNNING; log bootstrap xác minh source/data và GPU Tesla T4, torch2.11.0+cu128. Mã nguồn đã đóng băng trong bundle; không sửa bundle đang chạy.

Link: https://www.kaggle.com/code/nha4von/mfvlr-genface5000-epoch30-20261027-train

Đã quan sát `data_ready` đúng 5.000 train/400 validation, seed0/FD0,1 bắt đầu train và các cập nhật 1–29/37.500 hữu hạn. Tốc độ ban đầu khoảng 0,52 giây/cập nhật, chưa bao gồm đánh giá toàn bộ train/validation và ghi checkpoint mỗi epoch; chưa có metric epoch đầu hoặc kết quả chất lượng mới.

## Kết quả cuối, xác minh ngày 28/10/2026

Kaggle COMPLETE; đã tải report và xác minh COMPLETED, source_unchanged=true, smoke=false, 5.000 train/400 validation, đủ 30 lần đánh giá epoch và 37.500 cập nhật. Chưa dùng test. Chỉ seed0, không coi là trung bình nhiều seed.

| Metric | Train | Validation |
|---|---:|---:|
| AUC | 100% | 99,97% |
| Accuracy ngưỡng 0,5 | 100% | 99,00% |
| IoU vùng giả macro | 48,72% | 36,41% |
| Dice vùng giả macro | 65,24% | 53,01% |
| Precision pixel giả | 54,87% | 45,34% |
| Recall pixel giả | 85,84% | 63,88% |
| Diện tích báo giả trên ảnh thật | 0,0147% | 0,4084% |
| MSE AD | 0,00029015 | 0,00027809 |

Detection validation: TP197/TN199/FP1/FN3. Confusion pixel toàn bộ validation: TP881.936/FP1.063.334/FN498.746/TN17.626.384. Precision/recall tính từ confusion pixel toàn split; IoU/Dice macro chỉ lấy trung bình các ảnh giả, không trộn hai cách tổng hợp.

IoU validation theo epoch 5/10/15/20/25/30: 32,15% / 31,69% / 33,57% / 32,90% / 36,07% / 36,41%. Epoch30 vẫn là mốc báo cáo chính định trước, không chọn lại theo epoch đẹp nhất.

Đối chiếu cùng seed0 với lượt 1.000 ảnh × 15 epoch: IoU 28,20% → 36,41% (+8,20 điểm phần trăm). Dữ liệu và số cập nhật cùng tăng nên chưa quy riêng cải thiện cho từng yếu tố. Khoảng cách IoU train/validation cuối 12,31 điểm phần trăm, lớn hơn lượt cũ; gợi ý hạn chế tổng quát hóa mask và không ủng hộ tăng epoch một cách mặc định. Precision còn 45,34%, cho thấy vẫn nhiều FP; recall 63,88% còn bỏ sót GT. Không kết luận chỉ một nguyên nhân từ metric.

Report local: `artifacts/genface5000_epoch30_kaggle_20261027/downloaded/fd_validation_results/report.json`.
Model và checkpoint đầy đủ vẫn ở output notebook; bước báo cáo này mới tải report, chưa tải checkpoint/preview về máy.

Theo dõi:

```powershell
Get-Content artifacts/genface5000_epoch30_kaggle_20261027/remote_live.log -Tail 20 -Wait
```
