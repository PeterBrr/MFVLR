# GenFace: tăng thời gian train lên 15 epoch

Theo yêu cầu tiếp tục sau đối chứng 5 epoch, chạy λFD=0,1 với seed 0/1/2, mỗi lượt 15 epoch (3.750 cập nhật; tổng 11.250 cập nhật). Dùng lại dataset Kaggle riêng tư đã được cho phép; không tải thêm ảnh.

Giữ nguyên 1.000 ảnh train, 400 ảnh validation, λAR=λFL=1, trọng số pixel [1,4], Adam LR=1e-4, effective batch=4, FP32, decoder skip/resize-conv, AD tuyến tính và detach trước RE. Không bật nhánh ngôn ngữ. Không dùng test.

Train lại từ cùng khởi tạo và lịch shuffle theo seed như lượt 5 epoch để so sánh các mốc epoch 5/10/15. Không nạp checkpoint cũ. Việc này cũng cho phép kiểm tra tính lặp lại của mốc epoch 5; chỉ công bố so sánh sau khi kiểm tra report thực tế. Không thay kiến trúc hay công thức loss.

Runner thêm tùy chọn `--fd-weights` (mặc định vẫn 0,1 và 1 cho giao thức cũ). Notebook xác minh bundle/source gốc, sau đó cập nhật riêng runner đã ghi hash để chọn FD0,1. Report lưu hash runner thực sự chạy; `paired_initialization_verified=false` ở lượt mới vì mỗi seed chỉ chạy một trọng số FD, không còn đối chứng FD trong cùng job.

Theo dõi AUC, IoU vùng giả, precision/recall mask, MSE trên cả train/validation sau từng epoch. Epoch 15 là mốc kết quả chính; dùng các epoch trước để xem đường học, không lựa epoch đẹp nhất. Đo gradient ban đầu/cuối như lượt cũ.

Notebook riêng tư: https://www.kaggle.com/code/nha4von/mfvlr-genface-fd01-epoch15-20261027

Đã khởi chạy version 1, kernel ID 141089542; Kaggle xác nhận RUNNING, private, GPU T4, Internet tắt và gắn đúng dataset. Log bootstrap xác nhận hash dữ liệu/source gốc hợp lệ, runner mới đã áp dụng và CUDA/Tesla T4 sẵn sàng. CPU smoke hoàn thành một cập nhật; chưa có kết quả cuối của lượt 15 epoch.

Artifact local: `artifacts/genface_fd_epoch15_kaggle_20261027/`.

Lệnh đọc log local:

```powershell
Get-Content artifacts/genface_fd_epoch15_kaggle_20261027/remote_live.log -Tail 20 -Wait
```

CPU smoke một cập nhật chỉ kiểm tra luồng chọn FD/ghi report, không dùng làm đánh giá chất lượng.

## Kết quả cuối đã xác minh

Đã hoàn thành 3/3 seed, 11.250 cập nhật, khoảng 131 phút (runner 7.859,46 giây). Kaggle COMPLETE, report COMPLETED, source không đổi. Mốc epoch 5 của cả ba seed tái lập chính xác toàn bộ metric train/validation của thí nghiệm trước; hash dữ liệu và khởi tạo cũng trùng. Không dùng test.

Trung bình trên 400 ảnh validation qua ba seed:

| Metric | Epoch 5 | Epoch 10 | Epoch 15 |
|---|---:|---:|---:|
| AUC | 96,91% | 99,04% | 99,42% |
| Accuracy (ngưỡng 0,5) | 84,17% | 89,50% | 94,92% |
| IoU vùng giả macro | 13,77% | 26,39% | 26,20% |
| MSE AD | 0,002303 | 0,001867 | 0,001717 |
| Precision pixel giả | 27,17% | 27,66% | 32,15% |
| Recall pixel giả | 18,05% | 49,61% | 54,01% |
| Diện tích báo giả trên ảnh thật | 3,62% | 7,95% | 2,84% |
| IoU vùng giả trên train | 13,98% | 27,17% | 28,03% |

Độ lệch chuẩn mẫu epoch 15: AUC 0,21 điểm phần trăm; IoU 2,22 điểm phần trăm; MSE 0,000209. Các precision/recall ở bảng là trung bình metric từng seed, không cộng confusion của các model thành một model.

| Seed | AUC epoch 15 | IoU fake epoch 5 → 15 | MSE epoch 15 |
|---|---:|---:|---:|
| 0 | 99,52% | 10,87% → 28,20% | 0,001526 |
| 1 | 99,55% | 19,25% → 23,81% | 0,001686 |
| 2 | 99,18% | 11,19% → 26,57% | 0,001940 |

Tăng thời gian từ 5 lên 15 epoch cải thiện detection, tái tạo và IoU ở cả ba seed. Tuy nhiên IoU trung bình gần như không tăng từ epoch 10 lên 15; seed 1 giảm từ 28,58% xuống 23,81%. Điều này cho thấy tăng epoch có ích giai đoạn đầu, chưa đủ giải quyết độ chính xác vùng mask. Precision pixel giả 32,15% vẫn thấp; còn dự đoán dư vùng giả, trong khi recall mới đạt 54,01%.

Khoảng cách IoU train/validation ở epoch 15 khoảng 1,83 điểm phần trăm; chưa thấy khoảng cách tổng quát hóa lớn theo metric này, nhưng không loại trừ các dạng overfit khác hay rò rỉ danh tính chưa kiểm chứng. Không kết luận nguyên nhân duy nhất từ metric.

Đề xuất giữ FD0,1 và 15 epoch làm baseline tiếp theo. Trước khi tăng epoch tiếp, xem preview, phân tích FP/FN và gradient AR–FL để chọn một thay đổi có kiểm soát. Không chọn lại epoch chỉ vì IoU validation đẹp hơn; epoch 15 vẫn là mốc báo cáo chính đã định trước. Thí nghiệm vẫn tắt nhánh ngôn ngữ và AD không sigmoid.

Report đầy đủ: `artifacts/genface_fd_epoch15_kaggle_20261027/downloaded/fd_validation_results/report.json`.

Tổng hợp đã kiểm tra: `artifacts/genface_fd_epoch15_kaggle_20261027/comparison.json`.
