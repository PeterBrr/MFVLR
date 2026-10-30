# Skip connection, detach residual và trọng số loss thủ công — 26/10/2026

Đã thêm biến thể `skip_resize_conv_v2`, chạy qua factory/pipeline pilot hiện
có. Đây là thí nghiệm kỹ thuật; không coi cấu trúc bổ sung là mã gốc tác giả.

## Những phần đã triển khai

### Bốn skip connection

Encoder cung cấp feature64×224²,128×112²,256×56²,512×28². Decoder tăng mẫu
bottleneck1024×14² qua28/56/112/224, ghép channel với skip cùng độ phân giải,
rồi hai conv3×3+ReLU. Trước resize dùng projection1×1 để giảm channel.
AD/MD vẫn chung trunk, riêng head3/2 kênh; AD tuyến tính, không thêm sigmoid.

`model.skip_connections`:

- `concat`: dùng cả bốn skip từ appearance encoder.
- `none`: đối chứng bottleneck-only, cùng wrapper/loss và gradient policy.

AR/FL vẫn truyền qua skip và bottleneck vào encoder. Decoder không nhận skip
từ residual encoder. Nếu thêm skip làm reconstruction gần identity thì residual
cũng có thể mất tín hiệu hữu ích: cần xem detection cùng tái tạo/mask.

### Detach đúng đường gradient

`model.residual_gradient`:

- `coupled`: `R=abs(B-A)`, FD có đường cập nhật ngược qua B vào AD/trunk.
- `detach_reconstruction`: `R=abs(B.detach()-A)`, FD không cập nhật decoder
  qua đường này. AR vẫn so B gốc với A, FL vẫn dùng mask logits gốc.

```text
Encoder ảnh -> decoder có skip -> B -----------------------> AR
                           |     B.detach() -> |B-A| -> RE -> FD
                           +--> mask logits ----------------> FL
```

Đây là stop-gradient, không đóng băng trọng số decoder. FD vẫn cập nhật
encoder IE/RE dùng chung và detector. AR/FL vẫn cùng tác động lên trunk;
detach này không loại mọi xung đột. Giá trị forward giữ nguyên khi bật/tắt
detach; graph backward thay đổi.

`language_visual_gradients=stop_all` vẫn chặn loss language về nhánh ảnh như
biến thể trước. Đây là cơ chế khác với detach reconstruction trước RE.

### Sáu hệ số loss riêng

```text
L = lambda_fd*FD + lambda_ar*AR + lambda_fl*weighted_FL
  + lambda_lr*LR + lambda_cmc*CMC + lambda_kl*KL
```

Config phải ghi đủ sáu giá trị hữu hạn, không âm và ít nhất một giá trị dương.
Loss có hệ số0 không được tính, không dùng phép0*loss để che giá trị không hữu
hạn. `history.jsonl` báo loss chưa nhân lambda; loss tắt báo0. `total` đã nhân.

Config khởi đầu để thử, chưa tối ưu:

```json
"loss_weights": {"fd": 1.0, "lr": 0.0, "cmc": 0.0, "fl": 0.25, "ar": 1.0, "kl": 0.0}
```

Class weight pixel vẫn `[background=1, foreground=4]`, khác lambda_FL0,25.
Trong v2, giữ `mask_loss_weight=1`, chỉnh hệ số toàn FL bằng `loss_weights.fl`.
Chưa tự thay reduction AR/FL hoặc tự đo/chọn class weight mới.

Khi LR/CMC/KL đều0: không tạo FLT, không tokenize, không chạy nhánh language.
Trainer dùng tensor ID0 làm placeholder không tham gia loss. Khi chỉ CMC/KL
bật và LR=0: chỉ chạy LE, bỏ LD và vocabulary logits. Khi LR>0 mới chạy LD.
Cập nhật26/10/2026: EOS pooling, mask target PAD trong LR và L2 CMC đã được
nối vào biến thể này; xem [báo cáo triển khai và gradient](eos_pad_l2_va_gradient_ar_fl_vi.md).
Normalization LD vẫn chưa đổi; các config ảnh giữ LR/CMC/KL=0.

## Bộ config để so từng thay đổi

| Bước | Config | Skip | Residual | Lambda FL |
|---|---|---|---|---:|
| A | `diffae_pilot_no_skip_coupled.json` | Không | Coupled | 1 |
| B | `diffae_pilot_skip_coupled.json` | Concat | Coupled | 1 |
| C | `diffae_pilot_skip_detach_unit.json` | Concat | Detach B | 1 |
| D | `diffae_pilot_skip_detach.json` | Concat | Detach B | 0,25 |

So A→B kiểm tra skip, B→C kiểm tra detach, C→D kiểm tra lambda_FL.
Cả bốn giữ seed0, nearest resize, class weights[1,4], AR=FD=1, language=0,
optimizer và ngân sách tương ứng như nhau. Kiến trúc skip khác shape tham số
decoder; cùng seed không có nghĩa mọi trọng số decoder giống hệt A/B.

Tiny:32 train,1 epoch,16 updates. Pilot:1.000 train,5 epoch,2.500 updates;
batch2, LR1e-5, Adam weight decay1e-3. Validation/test giữ split và số ảnh cũ.
Không lấy test để chọn lambda; không mặc định D tốt hơn C hoặc các config khác.

```powershell
python train.py --config configs/diffae_pilot_skip_detach.json --stage tiny
python train.py --config configs/diffae_pilot_skip_detach.json --stage pilot
```

Mỗi lượt bắt đầu mới, không resume checkpoint decoder cũ sang kiến trúc skip.
Trainer tự thêm hậu tố thư mục nếu tên mặc định đã có dữ liệu.

## Kiểm tra

- Tests: bốn skip đều ảnh hưởng output và nhận gradient FL; sai shape bị từ chối.
- FD mất đường vào AD/trunk khi detach nhưng giữ gradient encoder/detector;
  AR và FL vẫn học decoder và encoder, mask head không đi qua AD head.
- Bật/tắt detach cho cùng model giữ output forward bằng nhau chính xác.
- Cấu hình no-skip/coupled có state và output khớp model resize-conv cũ.
- Kiểm tra scaling gradient của lambda, loss0 không tính, không tạo FLT khi
  language tắt, CMC/KL không chạy LD khi LR tắt, strict load sang model ảnh.
- Bộ `tests/pilot`: **39 test pass** trước lượt kiểm tra GPU.

Lượt kiểm tra pipeline thật dùng config D, tiny, lưu riêng tại:
`artifacts/diffae_pilot/runs/skip_detach_smoke_20261026/`.
Trạng thái COMPLETED:32 ảnh train,16 updates,1 epoch, validation400 ảnh;
khoảng58 giây cả lượt, peak CUDA allocated khoảng5221 MiB trên RTX4050 6GB.
Đã xác minh tổng loss đúng FD+AR+0,25FL ở mọi bước, các loss language bằng0.
Đã nạp lại best.pt bằng image-only factory với strict state load và xuất
bốn ảnh AD validation tại `ad_val/preview.png` trong cùng thư mục.

Validation sau lượt ngắn: AUC44,735%, foreground IoU fake1,6497%, diện tích
báo giả trên real3,1552%. Các số thấp, chưa đủ để kết luận cải thiện; chưa
chạy đầy đủ bốn đối chứng A–D hay pilot1.000 ảnh. Không dùng kết quả16 bước
để chọn trọng số tối ưu. Xem `status.json`/`history.jsonl` để đối chiếu.

Do source đã đổi, provenance của checkpoint lịch sử có thể không khớp source
hiện tại. Giữ nguyên kiểm tra hash; không nới kiểm tra để resume/đánh giá nhầm
graph mới với checkpoint cũ. Các kết quả cũ vẫn được giữ trong artifact.

## Đề xuất xử lý các vấn đề còn lại

| Vấn đề | Bước tiếp theo | Lợi ích kỳ vọng và giới hạn |
|---|---|---|
| AR/FL vẫn dùng chung trunk | Theo dõi norm/cosine; nếu có căn cứ, thử trunk AD/MD riêng với cùng budget | Tách đường cạnh tranh tại decoder, tăng tham số; encoder vẫn có thể nhận nhiều loss |
| LD không LayerNorm | Tích hợp lựa chọn nghiên cứu P3 pre-LN, đối chứng A0/Q3 hiện có trước khi tăng trọng số LR | Kiểm tra activation/conditioning; không khẳng định LN đã sửa lỗi học mask |
| PAD/EOS | Đã triển khai trong skipv2; kiểm chứng lợi ích khi mở nhánh language | Bỏ target PAD, giữ token-sum/batch-mean; EOS cuối; chưa đổi attention mask |
| CMC raw dot | Đã thêm L2 trong CMC của skipv2; kiểm chứng khi bật CMC | Feature FD/KL giữ gốc; tau và negative semantics vẫn cần đo |
| VIM một key | Giữ class-token baseline; đo mức phụ thuộc query/ảnh, thử nhiều key ở nhánh nghiên cứu riêng nếu cần | Nhiều key tạo lựa chọn theo query nhưng đổi kiến trúc/compute; sửa scale không chữa singleton |
| Quy ước residual VIM | Thử lựa chọn2*x+Delta đã chốt như một yếu tố riêng, không ghép cùng thay LN | Chưa triển khai; tăng phần residual có thể đổi thang activation, chưa xác nhận ý tác giả |
| Prompt shortcut | Hoán đổi/che visual conditioning ở cả LE/LD, đo LR content và chất lượng ảnh-only | Xác định dùng ảnh; không suy LR giảm là hiểu ảnh tốt hơn |

Ưu tiên trước mắt: xác nhận học AR/FL trên nhóm nhỏ với skip và detach; sau
đó so các config kiểm soát. Khi nhánh ảnh ổn định mới mở CMC/KL, rồi LR với
hệ số nhỏ có căn cứ từ phép đo gradient. Trọng số thủ công là thí nghiệm,
không phải hệ số đã được paper hoặc thực nghiệm hiện tại xác nhận tối ưu.

Căn cứ nội bộ: `src/mfvlr_research/decoder_a11.py`,
`docs/ad_scratch_test_vi.md`, `docs/split_mask_encoder_pilot_vi.md`,
`docs/nhung_diem_can_lam_ro_paper_va_code.txt`.
