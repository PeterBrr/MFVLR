# Pilot tách CNN encoder cho nhánh mask — 25/10/2026

**Chưa thấy lợi ích trong pilot đã chạy.** Cô lập gradient ở encoder đã đúng,
nhưng nhánh tách CNN tái tạo kém hơn và mask gần toàn nền ở bước 200.

## Thiết lập

- So baseline encoder chung với CNN encoder riêng cho MD. IE/RE vẫn chung
  encoder, AD/MD **vẫn chung trunk decoder**. Không tách decoder trong phép thử này.
- Từ trọng số mới, không đọc/nạp checkpoint đã train. CNN mask là bản sao
  độc lập của CNN ảnh vừa khởi tạo, cùng giá trị nhưng khác vùng nhớ.
- Cùng 32 ảnh train (16 real, 16 Diffae), chọn đều theo manifest trong pool
  1.000 ảnh train đã dùng; cùng 64 validation (32+32), không dùng test.
- Cùng 200 updates, effective batch 4 bằng hai microbatch 2, tương đương 25
  lượt qua 32 ảnh; lịch xáo trộn được tạo trước và dùng y hệt cho hai model.
- AR+FD+FL, hệ số1; FL weighted mean với class weights `[1,4]`. Không có
  LR/CMC/KL. Adam LR1e-4, weight decay0, seed0, FP32 deterministic, TF32 tắt.
- Kiến trúc resize-conv bottleneck-only, không sigmoid/skip/augmentation mới.
- Đã xác nhận common weights, output và metric ban đầu giống nhau chính xác;
  AD/detection route khớp model ảnh gốc.

Kiểm tra tại conv đầu của encoder và U1 projection của decoder trước cập nhật:

| Loss | CNN ảnh khi đã tách | CNN mask riêng | Trunk decoder chung |
|---|---|---|---|
| AR | Có gradient | Không có đường gradient | Có gradient |
| FD | Có gradient | Không có đường gradient | Có gradient |
| FL | Không có đường gradient | Có gradient | Có gradient |

Tách encoder loại đường cạnh tranh trực tiếp giữa FL và AR/FD tại CNN, chưa
loại tương tác tại trunk decoder hoặc giữa AR và FD.

## Kết quả validation tại bước 200

| Chỉ số, trên 64 ảnh | Encoder chung | CNN mask riêng |
|---|---:|---:|
| MSE tái tạo | **0,08991** | 0,18639 |
| Detection AUC | 52,54% | 54,79% |
| Foreground IoU trung bình trên fake | **0,11857%** | 0,001563% |
| Diện tích tô nhầm trên real | 0,14499% | 0,003675% |
| Global pixel mIoU hai lớp | 46,6808% | 46,6702% |
| Weighted FL | 0,55569 | 0,54396 |

Mask gần toàn nền ở cả hai nhánh. Real FP thấp hơn và FL thấp hơn của nhánh
tách không chứng minh localization tốt hơn: nó cũng bỏ sót gần hết foreground.
Global mIoU gần46,67% ở đây bị chất lượng nền chi phối một nửa phép trung
bình; cần đọc cùng foreground IoU/confusion, không xem là mask tốt.

Trên train, AUC là100%/98,83% nhưng foreground IoU chỉ0,2482%/0,00444%.
Detection đã học nhóm train nhỏ; localization vẫn yếu. Chênh AUC validation
2,25 điểm trên64 ảnh/seed0 chưa đủ xác nhận lợi ích tổng quát.

![So sánh validation](../artifacts/genface_split_mask_encoder_pilot_20261025/val_comparison.png)

Từ trái: input, GT, AD baseline, AD tách encoder, mask baseline, mask tách.
PNG reconstruction chỉ clip `[0,1]` để hiển thị; NPZ giữ float32 gốc.

![Các mốc validation](../artifacts/genface_split_mask_encoder_pilot_20261025/validation_curves.png)

Nhánh tách có MSE tốt hơn baseline ở mốc50/100/150, nhưng xấu hơn tại mốc200;
quá trình vẫn dao động. Báo cáo dùng trạng thái cuối theo ngân sách cố định,
không chọn mốc đẹp nhất sau khi nhìn validation. Chưa thử giảm LR hoặc nhiều seed.

## Chi phí và giới hạn

- Tham số nhánh ảnh: **39.046.343 ->57.889.543**, thêm18.843.200 của CNN;
  không phải trường hợp tách cả encoder và decoder đã ước lượng trước đó.
- Vòng cập nhật và đánh giá định kỳ: khoảng125 ->200 giây, tăng60%.
  Chưa tính toàn bộ chuẩn bị dữ liệu/khởi tạo/hash/lưu checkpoint.
- Peak CUDA allocated trong vòng đo: khoảng3331 ->3885 MiB, tăng554 MiB;
  khác với toàn bộ VRAM hệ thống hoặc CUDA reserved.
- Mới32 ảnh train/64 validation và200 updates, không so trực tiếp với các
  lượt overfit4 ảnh trước đó. Chưa kết luận tách encoder luôn gây hại.
- Tách tăng cả số tham số và compute, chưa có đối chứng cùng capacity.
  Cô lập gradient đúng không bảo đảm các loss còn lại cân bằng hoặc hội tụ.

Kết luận: **chỉ tách encoder chưa đủ sửa hiện tượng trong recipe này**.
Nếu tiếp tục, cần tách ảnh hưởng LR/loss scale hoặc decoder dùng chung;
những phép thử bổ sung đó chưa được thực hiện trong pilot này.

## Tệp

- [Report và gradient checks](../artifacts/genface_split_mask_encoder_pilot_20261025/report.json).
- [Ảnh train](../artifacts/genface_split_mask_encoder_pilot_20261025/train_comparison.png).
- Trọng số cuối: `shared_encoder/final.pt`, `separate_mask_encoder/final.pt`
  trong thư mục artifact trên; schema thí nghiệm `mfvlr-split-mask-encoder-pilot-v1`,
  chưa tích hợp vào CLI `train.py/evaluate.py` chính, không lưu optimizer state.
- Config: `configs/genface_split_mask_encoder_pilot.json`.
- Script: `tools/test_split_mask_encoder_pilot.py`. Chọn output mới nếu chạy lại.

Source model/loss chính và các checkpoint cũ không bị sửa. Đã kiểm tra
common initialization, gradient isolation, ngân sách, preview và source hash.
