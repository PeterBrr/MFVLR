# Thử trọng số pixel theo tỷ lệ train — 21/10/2026

Đã train từ đầu 1 epoch / 500 bước trên 1,000 ảnh (500 real + 500 Diffae), dùng
`configs/diffae_pilot_resize_conv_balanced.json`. Trọng số pixel là
`[1,12.583170252813833]`, lấy từ 46,482,017 pixel nền / 3,693,983 pixel giả trên
đúng subset train sau resize 224×224. Giữ cùng seed 0, batch 2 FP32, Adam LR 1e-5,
decoder resize-conv và stop-gradient language. Không chạy test hoặc thêm epoch.

Run: `artifacts/diffae_pilot/runs/pilot_resize_conv_balanced_1ep`.
Trạng thái COMPLETED, 500 optimizer steps, thời gian gồm validation/checkpoint
1,636 giây (27 phút 16 giây). Median khoảng 3.18 giây/bước, tương đương run
weight=4 ở epoch đầu (3.16 giây/bước). Không phát hiện lỗi finite trong lượt chạy.

| Validation sau cùng 1 epoch | Weight=4 | Weight=12.58317 |
|---|---:|---:|
| AUC detection | 66.8775% | 67.5650% |
| Accuracy | 63.00% | 60.75% |
| Fake foreground IoU macro | 0.3906% | 1.9707% |
| Fake foreground Dice macro | 0.7761% | 3.8510% |
| Real false-positive area | 0.3526% | 1.8608% |

Đây là so sánh cùng ngân sách 500 bước, không so trực tiếp với checkpoint cuối
5 epochs. Chỉ có một seed/run mỗi recipe, chưa đủ đánh giá độ ổn định thống kê.
Detection threshold được chọn trên validation riêng cho từng run theo code.

Đã xuất/kiểm tra 400 mask validation 224×224, binary 0/255, cùng ground truth,
overlay và ảnh so sánh trong `masks_val/`. Ảnh xem nhanh `masks_val/preview.jpg`.
Mask dự đoán vẫn gần toàn nền; các vệt foreground tập trung ở viền ảnh. Diện tích
foreground dự đoán trung bình khoảng 1.86% trên real và 1.89% trên fake. Tăng IoU
không đủ chứng minh model học được vùng sửa. Thống kê viền 8 pixel và kết quả máy
đọc trong `comparison.json`.

Kết luận: tăng weight theo tỷ lệ có tác dụng tăng phản hồi foreground, nhưng
chưa sửa được localization trong 1 epoch này, và làm tăng false positives trên
real. Không tự tăng weight/epoch tiếp. Muốn xác định bước sửa tiếp theo cần kiểm
tra target mask, khả năng overfit segmentation và cạnh tranh FD/AR trên nhánh
ảnh; không chọn hyperparameter bằng tập test.

Checkpoint `best.pt`/`last.pt` là lượt mới, không ghi đè run cũ. Config gốc
`diffae_pilot_resize_conv.json` vẫn giữ weight=4 và 5 epochs. Tái lập lượt thử với
output mới:

```powershell
python train.py --config configs/diffae_pilot_resize_conv_balanced.json --stage pilot
```

Lệnh sẽ chọn thư mục output mới tự động; kết quả đã có không cần train lại để đọc.
