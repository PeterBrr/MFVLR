# Biến thể resize + convolution, stop-gradient language và weighted mask

Đây là biến thể thực nghiệm `resize_conv_v1` theo yêu cầu người dùng, tách khỏi
A0. Chưa chạy pilot dài hoặc chứng minh cải thiện AUC/IoU. Checkpoint cũ được
giữ nguyên. Model mới phải train từ đầu, không resume hoặc nạp decoder A0.

## Thay đổi decoder

Mỗi tầng dùng `Conv2d 1×1` giảm kênh ở độ phân giải thấp, resize ×2, sau đó hai
`Conv2d 3×3 + ReLU`. Phép chiếu 1×1 được đặt trước resize để giảm bộ nhớ và chi
phí; nó hoán đổi được với nearest/bilinear resize vì là phép affine theo kênh.
Không còn `ConvTranspose2d` trong decoder mới. Giữ kích thước
1024×14×14 → 64×224×224, một trunk dùng chung và hai head appearance/mask riêng.
Nearest-neighbor là mặc định; config cũng hỗ trợ `bilinear` với
`align_corners=False`. Không thêm skip connection hoặc normalization.

Cơ sở phương án: [Deconvolution and Checkerboard Artifacts, Odena et al.](https://distill.pub/2016/deconv-checkerboard/).
Đây là lựa chọn kỹ thuật để thử nghiệm trên MFVLR, không phải recipe tác giả
MFVLR công bố và không bảo đảm loại mọi artifact sau train.

## Chặn ảnh hưởng language lên mask

`language_visual_gradients: "stop_all"` detach visual embedding ở **tất cả**
đầu vào FLT/VIM, adapter và visual side của CMC. Vì thế LR, CMC và KL không có
gradient về MVE hoặc decoder; FLT, adapter và contrastive temperature vẫn học.
Giá trị forward không bị thay đổi bởi detach.

Phải chặn cả encoder chung vì nó cũng tạo feature đầu vào cho mask. Chỉ detach
ảnh reconstruction ở một đường chưa loại được ảnh hưởng qua appearance encoder.
Đánh đổi là language không còn huấn luyện visual representation; đây không còn
là language-guided visual training theo A0. Nếu muốn ablation, policy `coupled`
khôi phục đường gradient language nhưng vẫn dùng decoder resize của biến thể.

FD, FL và AR vẫn cập nhật nhánh ảnh. AR/FD vẫn có thể ảnh hưởng tới mask thông
qua encoder/trunk chung; thay đổi này chỉ cô lập nhóm LR/CMC/KL.

## Trọng số pixel mask

Trong `configs/diffae_pilot_resize_conv.json`:

```json
"mask_background_weight": 1.0,
"mask_foreground_weight": 4.0,
"mask_loss_weight": 1.0
```

Pixel giả (class 1) có trọng số tương đối gấp 4 pixel nền (class 0). Đây là điểm
khởi đầu bảo thủ cho thử nghiệm, không phải hyperparameter đã được xác nhận.
Weighted CE dùng `sum(w[target] * NLL) / sum(w[target])`, tính bằng log-softmax,
gather và weighted sum để hỗ trợ CUDA deterministic, tránh kernel weighted
`nll_loss2d` không deterministic của PyTorch. Nó không nhân toàn bộ
mask loss với 4; `mask_loss_weight` mới là hệ số nhân toàn bộ FL. Mọi pixel vẫn
được tính; real mask toàn nền vẫn bị phạt khi dự đoán foreground. Các loss khác
giữ nguyên định nghĩa và hệ số. Cần theo dõi IoU fake **và** false-positive area
real trên validation; không chọn weight theo test.

## Lệnh chạy

Từ project root, trong `my_env`:

```powershell
python train.py --config configs/diffae_pilot_resize_conv.json --stage tiny
python train.py --config configs/diffae_pilot_resize_conv.json --stage pilot
```

Tiny vẫn dùng 32 ảnh / 100 bước. Pilot dùng 1,000 ảnh (500 real + 500 Diffae),
5 epochs, batch 2 FP32. Các run mới nằm ở `tiny_resize_conv_v1` hoặc
`pilot_resize_conv_v1` với suffix tự tăng khi thư mục đã có nội dung.
Các config A0 cũ vẫn xây dựng model A0, không tự đổi kiến trúc hoặc loss.

Evaluation và export tự dựng kiến trúc theo config trong checkpoint:

```powershell
python evaluate.py --checkpoint artifacts/diffae_pilot/runs/pilot_resize_conv_v1/best.pt --split test --output artifacts/diffae_pilot/runs/pilot_resize_conv_v1/test_metrics.json
python export_masks.py --checkpoint artifacts/diffae_pilot/runs/pilot_resize_conv_v1/best.pt --split test --output artifacts/diffae_pilot/runs/pilot_resize_conv_v1/masks_test
```

Nếu run có suffix, thay đúng đường dẫn terminal in ra. Không nạp checkpoint A0
vào biến thể mới: strict state loading sẽ từ chối topology khác nhau.

## Kiểm chứng đã thực hiện

- 14 test pass (CPU và CUDA), gồm regression pipeline A0, shape/gradient decoder mới,
  constant-feature upsampling, trọng số gradient pixel 1:4, hệ số FL tổng và
  gradient detach. Trọng số lớp bằng 1 cho kết quả FL bằng baseline.
- Weighted CE forward/gradient khớp CPU reference và lặp lại giống hệt trên GPU
  khi `torch.use_deterministic_algorithms(True)`. Smoke test ban đầu chưa bật
  chế độ này; đã bổ sung và chạy lại sau lỗi ở lượt train đầu tiên.
- Full-model GPU với deterministic=True, 2 ảnh train: forward/sáu loss/total backward finite. LR, CMC,
  KL không có đường gradient đến các tensor encoder, decoder và mask head được
  kiểm tra; FL vẫn có gradient khác 0 tại cả ba.
- Language encoder tiếp tục nhận LR/CMC/KL; adapter nhận KL; temperature nhận CMC.
- Visual-only strict load và inference parity: sai số detection/mask đều 0.
- Model full có 152,067,272 tham số. Không thực hiện optimizer update trong
  full-model check, chưa đo peak memory của Adam hoặc chất lượng sau train.

Báo cáo GPU deterministic: `artifacts/resize_conv_variant_smoke/deterministic_report.json`. Tái lập:

```powershell
python tools/check_resize_variant.py --output artifacts/resize_conv_variant_smoke/deterministic_report.json
```
