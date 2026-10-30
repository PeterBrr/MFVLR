# Kết quả đối chứng trọng số FD trên GenFace

Đã hoàn thành 6/6 lượt trên Kaggle, mỗi lượt 5 epoch (1.250 cập nhật), tổng thời gian runner 5.309,83 giây, khoảng 88,5 phút. Kaggle xác nhận COMPLETE; report xác nhận COMPLETED. Tổng 7.500 cập nhật.

## Thiết lập và kiểm tra

- 1.000 ảnh train cân bằng thật/Diffae; 400 ảnh validation cân bằng, không dùng test.
- So sánh λFD=0,1 và 1 theo cặp seed 0/1/2; λAR=λFL=1, trọng số pixel nền/giả=[1,4].
- Decoder có skip và resize-conv; AD tuyến tính, không sigmoid; detach reconstruction trước RE; không bật loss ngôn ngữ.
- Lấy epoch 5 đã định trước, không chọn epoch đẹp nhất.
- Report xác nhận source không đổi, khởi tạo giống nhau trong từng cặp và khác nhau giữa ba seed. Chưa xác minh identity-disjoint.

## Validation epoch 5

| Seed | λFD | AUC (%) | IoU vùng giả macro (%) | AR/MSE |
|---|---:|---:|---:|---:|
| 0 | 0,1 | 97,30 | 10,87 | 0,002115 |
| 0 | 1 | 89,26 | 6,10 | 0,002868 |
| 1 | 0,1 | 96,22 | 19,25 | 0,002104 |
| 1 | 1 | 91,48 | 13,27 | 0,002207 |
| 2 | 0,1 | 97,20 | 11,19 | 0,002691 |
| 2 | 1 | 76,91 | 16,10 | 0,003636 |

Trung bình ± độ lệch chuẩn mẫu qua 3 seed:

| λFD | AUC (%) | IoU vùng giả macro (%) | AR/MSE |
|---:|---:|---:|---:|
| 0,1 | 96,91 ± 0,60 | 13,77 ± 4,75 | 0,002303 ± 0,000336 |
| 1 | 85,88 ± 7,85 | 11,82 ± 5,16 | 0,002904 ± 0,000715 |

## Nhận xét và hướng tiếp theo

Giảm λFD về 0,1 cải thiện AUC và MSE ở cả ba seed. AUC cũng dao động ít hơn trong ba lượt đã chạy. Với mask, IoU tăng ở seed 0/1 nhưng giảm ở seed 2; mức tăng trung bình 1,94 điểm phần trăm chưa đủ để kết luận tác động luôn nhất quán hoặc có ý nghĩa thống kê.

IoU vùng giả vẫn thấp; kết quả ghi nhớ trên 16 ảnh trước đó không chuyển thành chất lượng localization tương đương trên validation. Chưa xác định nguyên nhân duy nhất: không thể chỉ từ các metric này quy hết cho xung đột gradient hoặc độ khó dataset.

Đề xuất dùng λFD=0,1 làm cấu hình đối chứng tiếp theo và giữ nguyên các thành phần khác. Ưu tiên kiểm tra mask train/validation, đường học FL và gradient trước khi thay tiếp trọng số pixel hoặc kiến trúc. Không thay nhiều yếu tố đồng thời. Chưa chạy test để chọn cấu hình.

## File kết quả

- `artifacts/genface_fd_validation_kaggle_20261026/remote_live.log`
- `artifacts/genface_fd_validation_kaggle_20261026/downloaded/fd_validation_results/report.json`
- Preview từng lượt: `downloaded/fd_validation_results/seed_<seed>_fd_<0p1 hoặc 1p0>/val/preview.png` trong thư mục artifact trên.
- Notebook: https://www.kaggle.com/code/nha4von/mfvlr-genface-fd-pilot-20261026-001

Đây là pilot nội bộ trên GenFace/Diffae, không phải kết quả tái hiện đầy đủ paper.
