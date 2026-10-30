# Phân tích lỗi mask và gradient sau 15 epoch

Phân tích report đầy đủ của ba seed và preview cố định bốn ảnh validation (hai thật, hai Diffae). Không train thêm, không đổi model, không dùng test. Metric toàn tập dưới đây tính trên 400 ảnh; nhận xét hình chỉ áp dụng trực tiếp cho bốn ảnh preview.

## 1. Mask vừa dư vùng, vừa sai vị trí

GT foreground chiếm 6,879% tổng pixel validation, giống nhau giữa các seed.

| Seed | Diện tích dự đoán / GT | FP / FN | Precision | Recall | IoU foreground global |
|---|---:|---:|---:|---:|---:|
| 0 | 1,85 lần | 3,00 | 31,14% | 57,57% | 25,32% |
| 1 | 2,53 lần | 5,17 | 25,08% | 63,40% | 21,91% |
| 2 | 1,02 lần | 1,03 | 40,24% | 41,05% | 25,50% |

Đây là confusion cộng trên cả 400 ảnh, gồm ảnh thật. IoU foreground global không phải IoU fake macro 28,20% / 23,81% / 26,57% đã báo cáo; hai cách tổng hợp khác nhau.

Seed 0/1 có dự đoán dư rõ ràng. Seed 2 có tổng diện tích gần đúng nhưng overlap vẫn thấp: diện tích đúng không đồng nghĩa vị trí đúng. Vì vậy chỉ thu nhỏ mask không thể giải quyết toàn bộ lỗi. Nếu giảm trọng số foreground hoặc tăng ngưỡng, có thể giảm FP nhưng cũng làm mất TP, nhất là seed 2.

Trong preview, FP đỏ tập trung ở một số đường tóc, viền mặt và chi tiết nền; các chi tiết tương tự cũng bị dự đoán giả trên ảnh thật GT rỗng. Trên hai ảnh Diffae, model bắt được một phần cấu trúc GT nhưng vừa mở rộng sang đường nét khác vừa bỏ sót GT mảnh. Điều này gợi ý model còn nhạy với đường biên/texture thông thường. Chưa chứng minh shortcut này trên toàn bộ dataset; cần thống kê theo ảnh để xác nhận.

Không đánh giá mask bằng pixel accuracy: chỉ dự đoán toàn nền đã đạt 93,12% pixel accuracy trên tập này nhưng không định vị được pixel giả nào.

## 2. Tăng epoch đã giúp, nhưng IoU chững sau epoch 10

IoU validation trung bình 13,77% → 26,39% → 26,20% ở epoch 5/10/15. IoU train tương ứng 13,98% → 27,17% → 28,03%. Khoảng cách train/validation cuối chỉ 1,83 điểm phần trăm, nên chưa có dấu hiệu khoảng cách tổng quát hóa lớn theo IoU này.

Seed 1 từ epoch 10 đến 15 tăng recall 56,92% → 63,40%, giảm precision 29,50% → 25,08%, IoU giảm 28,58% → 23,81%. Model tìm thêm pixel giả nhưng thêm nhiều FP. Cùng lúc diện tích báo giả trên ảnh thật giảm 5,52% → 3,41%, chứng tỏ một metric trên ảnh thật không mô tả hết lỗi trên ảnh giả. Không nên xem việc giảm FL hoặc tăng recall riêng lẻ là đảm bảo IoU tăng.

## 3. Detach hoạt động; chưa thấy AR lấn át FL ở cuối lượt

Ở cả initial/final, bốn cặp mẫu train cố định của mỗi seed, gradient FD không nối vào trunk hoặc hai head decoder. Đây là kiểm tra kết nối autograd, không chỉ nhìn norm bằng 0. Detach trước RE đã chặn đường trực tiếp FD → AD. FD vẫn cập nhật encoder chung, nên ảnh hưởng gián tiếp lên decoder còn tồn tại.

Gradient AR–FL tại decoder trunk cuối lượt:

| Seed | Cosine trung bình 4 cặp | Số cặp cosine âm | Trung vị norm FL / AR |
|---|---:|---:|---:|
| 0 | +0,0051 | 2/4 | 16,85 |
| 1 | +0,0211 | 0/4 | 15,01 |
| 2 | −0,0010 | 2/4 | 3,64 |

Cosine gần 0 biểu thị gần vuông góc ở các phép đo này, không phải thuận mạnh hoặc đối nghịch mạnh. FL có norm lớn hơn AR trên trunk, nên dữ liệu cuối lượt không ủng hộ giả thuyết AR đang lấn át FL ở đây. Chưa có cơ sở tăng λFL chỉ vì IoU thấp.

Ở CNN encoder, cosine AR–FL cuối theo seed là +0,0915 / +0,1159 / −0,1729; số cặp âm là 1/4, 1/4, 4/4. Có xung đột tại encoder ở seed 2 nhưng không đồng nhất giữa seed hoặc tầng. Đầu lượt trunk có 8/12 cặp âm, cuối lượt còn 4/12; đây là hai snapshot, không phải tỉ lệ xung đột xuyên suốt quá trình train.

Norm và cosine chỉ đo bốn cặp ảnh train ở đầu/cuối; chúng không đại diện toàn bộ batch, epoch hoặc toàn bộ encoder Transformer. Norm gradient thô cũng không bằng độ lớn cập nhật Adam do có moment và chuẩn hóa theo tham số. Các tỷ số so với FD ở một số probe cuối rất lớn vì FD gần bão hòa, không dùng chúng để khẳng định FD luôn yếu trong train.

## 4. Trọng số pixel 4 là giả thuyết cần kiểm tra

Code hiện tính FL = sum(w_y * CE_pixel) / sum(w_y), w_nền=1, w_giả=4. Đây khác với tăng λFL: trọng số pixel đổi ưu tiên giữa FP/FN, còn λFL đổi tương quan cả nhiệm vụ mask với AR/FD.

Để hiểu xu hướng, xét một pixel có xác suất nhãn giả thật sự là η và prediction p. Expected weighted CE là −4η log(p) − (1−η)log(1−p). Trong mô hình lý tưởng với mẫu số reduction cố định, nghiệm p*=4η/(1+3η). Chọn p≥0,5 tương đương η≥0,2. Đây chỉ là minh họa hướng thiên về foreground, không chứng minh output thực tế đã được hiệu chuẩn hay dự đoán đúng tác động định lượng của weighted-mean theo batch.

Do hiện có cả dư vùng lẫn sai vị trí, không nên tăng foreground weight ngay. Chưa thể kết luận weight=4 là nguyên nhân duy nhất; kiến trúc, tối ưu và cấu trúc GT còn liên quan.

## 5. Bước tiếp theo đề xuất

1. Giữ FD0,1, AR=FL=1, 15 epoch và cùng split/seed. Đối chứng duy nhất foreground weight 1 / 2 / 4, ưu tiên xem precision–recall, IoU fake macro và global, FP trên ảnh thật. Không thay decoder hoặc bật ngôn ngữ cùng lượt. Có thể chạy seed 0 trước để sàng lọc, sau đó xác nhận ba seed; không xem một seed là kết luận.
2. Trước khi thêm Dice/đổi kiến trúc, xuất probability trên đủ 400 validation để xem đường precision–recall và độ nhạy theo ngưỡng. Preview hiện chỉ có bốn ảnh, không đủ chọn ngưỡng cho toàn tập. Nếu dùng validation để chọn threshold thì ghi rõ đây là bước tuning, giữ test độc lập; không sửa threshold rồi gọi đó là model học tốt hơn.
3. Nếu các đối chứng vẫn chững, đo gradient trên nhiều batch/tại nhiều epoch trước khi quyết định tách trunk hoặc dùng phương pháp xử lý xung đột. Hiện chưa đủ bằng chứng để ưu tiên tách decoder vì AR–FL đối nghịch mạnh.

Chưa tự chạy thêm train trong bước phân tích này.

## Artifact

- Thống kê: `artifacts/genface_fd_epoch15_kaggle_20261027/analysis.json`.
- Đường học: `artifacts/genface_fd_epoch15_kaggle_20261027/analysis_learning_curves.png`.
- Bốn ảnh minh họa: `artifacts/genface_fd_epoch15_kaggle_20261027/validation_error_previews.png`; xanh lá=TP, đỏ=FP, xanh dương=FN, đen=TN. Đây là preview cố định, không phải mẫu chọn theo kết quả đẹp/xấu.
- Script tái tính: `tools/analyze_fd_epoch15.py`, `tools/plot_fd_epoch15_errors.py`.
