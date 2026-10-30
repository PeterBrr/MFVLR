# Test checkpoint GenFace 5.000 ảnh × 30 epoch

Ngày 28/10/2026, đã đánh giá model cuối epoch30, seed0, step37.500 trên 400 ảnh test cố định (200 thật, 200 Diffae). Không train thêm, không chỉnh threshold hoặc chọn checkpoint bằng test. Detection threshold0,5, mask argmax như validation. Nhánh ngôn ngữ tắt.

## Kiểm tra trước test

- Tải `final.pt` từ output Kaggle; SHA256 `d012a65b73158f013100847b72c10d922b72fa872188967981359e2dc9f8e407`.
- Khôi phục source snapshot từ bundle train và kiểm tra toàn bộ hash với checkpoint/report. Không dùng adapter đang sửa trong source workspace.
- Nạp strict toàn bộ trọng số model, không đổi kiến trúc.
- Kiểm tra bytes ảnh/mask so với manifest khi đọc; giải mã, resize theo reader gốc; không trùng ID hoặc RGB sau resize trong test và giữa test với train5.000/validation400.
- Chạy lại validation local: AUC và IoU khớp chính xác Kaggle; MSE chênh khoảng 3,2e-12. Môi trường local torch2.7.1+cu118, RTX4050 Laptop; Kaggle torch2.11.0+cu128/T4.
- Không có metadata xác nhận identity-disjoint. Tập test này từng được dùng cho các pilot cũ nên đây là kiểm tra nội bộ trên split test có sẵn, không tuyên bố holdout hoàn toàn chưa từng xem trong toàn bộ quá trình phát triển.

## Kết quả

| Metric | Validation | Test |
|---|---:|---:|
| AUC | 99,97% | 99,935% |
| Accuracy | 99,00% | 99,00% |
| F1 detection | 98,995% | 98,995% |
| IoU vùng giả macro | 36,41% | 37,28% |
| Dice vùng giả macro | 53,01% | 53,88% |
| Precision pixel giả | 45,34% | 46,26% |
| Recall pixel giả | 63,88% | 63,04% |
| Diện tích báo giả trên ảnh thật | 0,408% | 0,483% |
| MSE AD | 0,00027809 | 0,00028363 |

Detection test: TP197/TN199/FP1/FN3. Pixel test: TP913.551/FP1.061.205/FN535.699/TN17.559.945. IoU/Dice macro trên 200 ảnh giả; precision/recall từ confusion cộng toàn bộ400 ảnh. Global pixel mIoU hai lớp64,03%, không nhầm với IoU vùng giả37,28%.

Test và validation cho kết quả gần nhau, chưa thấy suy giảm lớn giữa hai split này. Detection tốt trong phạm vi GenFace/Diffae đã xét; mask vẫn có nhiều FP và bỏ sót, chưa giải quyết hoàn toàn localization. Chỉ một seed, một generator, identity split chưa xác minh nên không suy rộng thành khả năng tổng quát hóa sang generator hoặc danh tính khác.

## File

- Metric: `artifacts/genface5000_epoch30_kaggle_20261027/test_001/test_metrics.json`.
- Preview bốn ảnh cố định: `artifacts/genface5000_epoch30_kaggle_20261027/test_001/preview.png` và `preview.npz`.
- Source đã xác minh: `test_001/frozen_source/` trong artifact trên.
- Checkpoint local: `downloaded/fd_validation_results/seed_0_fd_0p1/final.pt` trong artifact trên.
- Script: `tools/test_genface5000_checkpoint.py`.
