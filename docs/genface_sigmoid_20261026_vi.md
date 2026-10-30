# Thử sigmoid tại đầu ra AD trên GenFace — 26/10/2026

## Thay đổi triển khai

Tùy chọn `model.reconstruction_activation` trong variant `skip_resize_conv_v2`:

- `identity`: giữ RGB tuyến tính như trước, là mặc định khi config không có khóa này.
- `sigmoid`: B = sigmoid(appearance_head(features)), áp dụng trước AR và residual.

Mask head vẫn trả hai logits để tính cross-entropy; không thêm sigmoid vào mask.
AR vẫn là MSE giữa B và input RGB [0,1]. Không thay trọng số loss hoặc ảnh GT.
Detach B trước RE vẫn giữ nguyên; gradient AR đi qua sigmoid về AD và encoder.
Sigmoid không có tham số; state-dict trọng số giữ cùng tên và shape, nhưng config
phải được lưu cùng checkpoint để tái tạo đúng forward.

Đã nối tùy chọn vào cả model train và image-only inference, cho decoder concat
skip hoặc không skip của v2. A0 và config resize_conv_v1 tiếp tục hành vi cũ.

Sigmoid bảo đảm miền RGB [0,1] với đầu vào hữu hạn, nhưng không bảo đảm MSE,
mask hoặc detection tốt hơn. Đạo hàm sigmoid là B(1-B), tối đa 0,25 và nhỏ khi
bão hòa; cùng lambda_AR không có nghĩa norm gradient AR giữ nguyên.

## Thiết kế đối chứng

- Chỉ GenFace real + Diffae; không dùng FF++ hoặc test.
- 128 ảnh train và 128 validation cân bằng 64 real/64 fake mỗi split.
- Hai lượt mới từ đầu seed 0, 100 cập nhật, microbatch 2 và effective batch 4.
- D dùng identity; G dùng sigmoid; hash trọng số ban đầu phải giống nhau.
- Cùng ID ảnh và lịch batch như sáu lượt trước.
- Cùng skip concat, nearest resize-conv, detach B trước RE, language tắt.
- Lambda FD=1, AR=1, FL=0,25; class weight pixel [1,4].
- Adam LR=1e-5, weight decay=1e-3, FP32 deterministic, activation checkpointing.
- Mốc cuối 100 là kết quả chính; validation bổ sung ở bước 50.
- AUC, MSE, IoU fake, báo giả trên real; thêm min/max RGB, số giá trị ngoài
  [0,1] và gần hai biên 0/1 trên toàn bộ validation. MSE dùng RGB gốc, không clip.

51 kiểm thử pilot đã pass trước khi chạy, bao gồm sigmoid giữ nguyên mask logits
với cùng feature/trọng số, AR có gradient hữu hạn, detach vẫn chặn FD qua residual,
và train/inference khớp khi load trọng số.

Source đúng hash của sáu thí nghiệm trước đã được lưu trước khi sửa trong
`artifacts/genface_ablation_20261026/trials_001/source_snapshot/`.

## Tệp và lệnh

Config sigmoid: `configs/diffae_pilot_sigmoid.json`.
Output đối chứng: `artifacts/genface_sigmoid_20261026/trials_001/`.
Runner lưu config, lịch batch, hash nguồn/khởi tạo, metric và checkpoint từng lượt.
Checkpoint của runner là schema thí nghiệm riêng, không dùng trực tiếp CLI evaluate.py.

Lệnh chạy lại đối chứng, cần chọn thư mục output mới:

```powershell
python tools/run_genface_ablation.py --experiment sigmoid --output artifacts/genface_sigmoid_20261026/trials_repeat --updates 100 --train-images 128 --val-images 128
```

Config cũng dùng được với trainer chính; lệnh sau chạy pilot 1.000 ảnh/5 epoch
theo config, dài hơn đối chứng 128 ảnh/100 cập nhật vừa mô tả:

```powershell
python train.py --config configs/diffae_pilot_sigmoid.json --stage pilot
```

## Kết quả hoàn tất

Hai lượt đều `COMPLETED`, đủ 100 cập nhật và checkpoint `final.pt`. Source giữ
nguyên trong quá trình chạy. D/G có cùng hash toàn bộ trạng thái ban đầu.
Lượt D tái lập chính xác toàn bộ 100 giá trị loss và metric validation cuối
của D trong sáu thí nghiệm trước; việc thêm tùy chọn không đổi forward mặc định.

Kết quả validation ở bước 100 trên cùng 128 ảnh:

| Chỉ số | D: tuyến tính | G: sigmoid |
|---|---:|---:|
| AUC detection (%) ↑ | 66,748 | 75,391 |
| Accuracy tại ngưỡng 0,5 (%) | 50,000 | 50,000 |
| MSE tái tạo ↓ | 0,011966 | 0,031468 |
| IoU foreground macro trên fake (%) ↑ | 1,160 | 0,608 |
| Recall pixel giả tổng thể (%) ↑ | 1,252 | 0,648 |
| Vùng báo giả trên ảnh real (%) ↓ | 0,997 | 0,429 |
| Min / max RGB gốc | -0,668 / 1,825 | 0,053 / 0,943 |
| Giá trị kênh RGB ngoài [0,1] (%) | 3,077 | 0 |
| Số tham số | 42.442.440 | 42.442.440 |

Train MSE tương ứng 0,012032 / 0,030408; train IoU fake 1,110% / 0,521%.
Ở sigmoid, không có giá trị RGB validation nhỏ hơn 0,01 hoặc lớn hơn 0,99.
Vì vậy chưa có bằng chứng output đang bão hòa sát 0/1 ở các lần đánh giá này.

### Diễn giải

- Sigmoid giới hạn đúng miền RGB và AUC tăng khoảng 8,64 điểm phần trăm trong
  pilot một seed. Đây là cải thiện xếp hạng score, không phải accuracy tăng:
  G vẫn dự đoán cả 128 ảnh là fake tại ngưỡng 0,5, nên accuracy chỉ 50%.
- **Chưa cải thiện tái tạo theo MSE:** lỗi G cao khoảng 2,63 lần D. Trên bốn
  mẫu cố định, ảnh G trông bớt vân màu nhưng nhòe hơn và màu chưa sát input.
  Hình ảnh mượt hơn không đồng nghĩa tái tạo chính xác hơn.
- **Mask kém hơn:** IoU và recall đều giảm, ảnh mask vẫn gần toàn nền. Vùng
  báo giả trên real giảm chưa đủ để kết luận localization tốt hơn vì model
  đồng thời bỏ sót nhiều pixel giả hơn.
- Cùng lambda nhưng sigmoid đổi cả miền RGB, residual và đạo hàm AR; kết quả
  trên chưa tách được đóng góp của từng cơ chế. Không kết luận gradient đã
  vanishing chỉ từ ảnh nhòe hoặc MSE; chưa đo lại gradient AR/FL cho G.
- Không tự thay sigmoid thành mặc định. Đã giữ config riêng để đối chứng.
  Cần kiểm tra overfit AR-only/AR+FL trên tập ảnh cố định nhỏ trước khi quyết
  định tăng số bước, đổi LR hoặc cân lambda cho sigmoid.

Thời gian D/G khoảng 363,8/168,9 giây, peak CUDA allocated 5.255/5.258 MiB.
Không quy chênh lệch tốc độ cho sigmoid: chính D ở lần chạy trước mất khoảng
172,1 giây nhưng tái lập loss/metric đúng như lần này. Chưa benchmark thời gian
trong điều kiện tải máy và GPU được kiểm soát.

Giới hạn: 128 train/128 validation, khoảng 3,125 epoch, một seed, chưa test;
chưa xác nhận mức tăng AUC có ổn định khi mở rộng dữ liệu hoặc đổi seed.

### Ảnh và dữ liệu đã lưu

- [Ảnh input/GT/AD/mask D và G](../artifacts/genface_sigmoid_20261026/trials_001/val_D_manual_fl_vs_G_sigmoid.png).
- [Ảnh AD riêng](../artifacts/genface_sigmoid_20261026/trials_001/all_AD.png).
- [Mask riêng](../artifacts/genface_sigmoid_20261026/trials_001/all_mask.png).
- [Bảng metric](../artifacts/genface_sigmoid_20261026/trials_001/comparison.json).
- [Report đầy đủ](../artifacts/genface_sigmoid_20261026/trials_001/report.json).

Bốn mẫu minh họa ở index 0,1,126,127 được giữ như sáu thí nghiệm trước.
Metric dùng đủ 128 ảnh; PNG clip RGB về [0,1] để hiển thị, MSE và `.npz` dùng
output gốc. `source_snapshot/` lưu source đúng hash cho lần đối chứng này.
