# Đối chứng augmentation theo DeepfakeBench (đăng ký trước 31/10/2026)

Viết trước khi chạy. Mục "Kết quả" được bổ sung sau khi chạy xong.

## Câu hỏi và lý do

**Augmentation theo DeepfakeBench có tăng IoU của nhánh ảnh không?** Model hiện đánh dấu thừa nhiều (precision thấp), và pilot chỉ có 1.000 ảnh train nên có thể học thuộc dữ liệu. Paper không mô tả augmentation, nên đây là thử nghiệm cải tiến của dự án, không phải tái hiện paper.

## Thiết lập

- Nơi chạy: Kaggle T4, tài khoản `nha4von`, notebook riêng tư `mfvlr-augmentation-20261031`, mỗi GPU một lượt, mỗi lượt một process. Lượt local đầu tiên bị dừng ở bước 550 của `aug_deepfakebench` seed 0 khi chuyển nền tảng; số liệu của lượt đó chưa được xem và không được dùng.
- Baseline: `residual_mask_none` được **chạy lại trên Kaggle**, vì baseline cũ chạy trên RTX 4050 local và không nên so ghép cặp giữa hai nền tảng. Kết quả local cũ (IoU 27,66) chỉ để tham khảo. Trên máy local, nhánh `none` của code mới đã cho kết quả giống hệt từng bit với code cũ (617 tensor, 30 cập nhật).
- Dữ liệu lấy từ dataset riêng tư `nha4von/mfvlr-genface-fd-pilot-20261026`, cùng hash với bản dùng ở các đối chứng trước.
- Mọi thiết lập khác giữ nguyên: 1.000 train / 400 validation, không language, 15 epoch, seed 0, 1, 2, cùng khởi tạo và cùng lịch batch, không dùng test. Validation không augmentation.
- Ảnh và mask biến đổi hình học đồng bộ. AR tái tạo đúng ảnh đã augmentation.

| Nhánh | Config | Biến đổi |
|---|---|---|
| Flip | `aug_flip` | Lật ngang p 0,5 |
| DeepfakeBench | `aug_deepfakebench` | Recipe UCF, theo thứ tự bên dưới |

Recipe UCF (commit `f188b1c`, `ucf.yaml` và `init_data_aug_method()`):
1. Lật ngang, p 0,5.
2. Xoay ±10 độ, p 0,5; ảnh nội suy tuyến tính, mask nearest.
3. Gaussian blur, kernel 3, 5 hoặc 7, p 0,5.
4. Chọn một trong brightness/contrast ±0,1, FancyPCA, HueSaturationValue, p 0,5.
5. Nén JPEG chất lượng 40–100, p 0,5.

Khác với DeepfakeBench:
- Viết lại bằng OpenCV, không dùng Albumentations, nên cùng tham số nhưng không cùng giá trị ngẫu nhiên.
- Ảnh 224×224 và không align lại; không dùng normalize mean/std 0,5.
- Bỏ IsotropicResize, vì ảnh đã vuông đúng kích thước nên phép này không đổi gì.
- JPEG mã hóa đúng thứ tự kênh màu.

## Tiêu chí

Chỉ số chính: IoU fake macro trên validation, trung bình 3 epoch cuối của mỗi seed, rồi lấy chênh lệch ghép cặp so với baseline.

**Mỗi nhánh được coi là đáng xác nhận trên 5.000 ảnh khi đồng thời:**
1. Tốt hơn baseline ở cả 3/3 seed.
2. Chênh lệch trung bình ít nhất +2 điểm IoU.
3. FP area trên ảnh real không tăng quá 0,5 điểm và AUC không giảm quá 0,5 điểm.
4. Train không có lỗi số học.

Hai nhánh được đánh giá độc lập; nếu cả hai đạt, chọn nhánh có IoU trung bình cao hơn. Precision, recall và Dice vẫn được báo cáo.

## Giới hạn đã biết trước

- Augmentation thường cần train lâu hơn, nên 15 epoch có thể bất lợi cho nhánh DeepfakeBench.
- Blur và JPEG có thể làm mất dấu vết giả mà nhánh residual dựa vào, và validation không có các suy giảm này, nên thử nghiệm không đo độ bền trước ảnh nén.
- 3 seed và 400 ảnh validation không đủ để kết luận về ý nghĩa thống kê.
- Thử nghiệm không kiểm tra khả năng tổng quát sang generator khác, vốn là một mục đích chính của augmentation trong DeepfakeBench.

## Kết quả

Chạy trên Kaggle T4 (torch 2.11.0+cu128), 4 giờ 5 phút cho 9 lượt. Kết quả trong `artifacts/ablation_augmentation_kaggle_20261031/downloaded/results`. Mọi lượt chạy đủ 3.750 cập nhật, không phải chạy lại, không có lỗi số học, cùng dữ liệu và cùng lịch batch trong mỗi seed.

**Không nhánh nào đạt tiêu chí; không chuyển sang xác nhận trên 5.000 ảnh.**

| Tiêu chí | Ngưỡng | `aug_flip` | `aug_deepfakebench` |
|---|---|---|---|
| 1. Tốt hơn baseline | 3/3 seed | Đạt (+1,26 / +3,05 / +0,90) | Không đạt, 1/3 (−0,69 / +0,35 / −0,39) |
| 2. IoU tăng trung bình | ≥ +2 điểm | Không đạt (+1,74) | Không đạt (−0,24) |
| 3a. FP area real tăng | ≤ 0,5 điểm | Đạt (−0,31) | Đạt (−0,39) |
| 3b. AUC giảm | ≤ 0,5 điểm | Đạt (+0,15) | Đạt (−0,20) |
| 4. Không lỗi số học | | Đạt | Đạt |

Trung bình 3 epoch cuối của 3 seed:

| Chỉ số (%) | Baseline | `aug_flip` | `aug_deepfakebench` |
|---|---|---|---|
| IoU | 26,68 | 28,42 | 26,44 |
| Dice | 41,60 | 43,74 | 41,25 |
| AUC | 99,31 | 99,46 | 99,11 |
| Accuracy | 93,50 | 94,81 | 90,56 |
| FP area trên ảnh real | 3,41 | 3,10 | 3,01 |
| Precision | 32,49 | 33,93 | 33,96 |
| Recall | 50,89 | 52,80 | 47,27 |

Nhận xét:
- Chỉ lật ngang tăng cả precision lẫn recall trung bình, nhưng precision giảm ở seed 2 (−1,66).
- Recipe đầy đủ làm recall giảm 3,62 điểm và accuracy giảm 2,94 điểm. Thử nghiệm không tách riêng được phép nào gây ra, và 15 epoch có thể chưa đủ cho recipe này.
- Baseline trên Kaggle (26,68) thấp hơn baseline local (27,66) khoảng 1 điểm ở cả 3 seed, nên không so sánh chéo hai nền tảng. Mức này cùng cỡ với hiệu ứng đang đo.
- IoU vẫn dao động mạnh: ở 5 epoch cuối, baseline seed 0 chênh tới 13,41 điểm.

Quyết định: không đổi ngưỡng sau khi đã thấy kết quả. Recipe đầy đủ không được dùng. Lật ngang chưa được xác nhận, chỉ được giữ làm một nhánh trong đối chứng tiếp theo.
