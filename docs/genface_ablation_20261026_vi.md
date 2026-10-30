# Đối chứng các chỉnh sửa trên GenFace — 26/10/2026

## Kế hoạch đã chốt trước khi train

Chỉ dùng **GenFace real + Diffae**, không dùng FF++ hoặc test.

-128 ảnh train cân bằng64 real/64 Diffae, chọn đều trong pool1.000 ảnh train.
-128 validation cân bằng64/64, chọn đều từ split validation400 ảnh sẵn có.
-Mỗi lượt khởi tạo mới seed0;100 optimizer updates; cùng lịch xáo trộn.
-Effective batch4 bằng hai microbatch2, tương đương3,125 lượt qua128 ảnh.
-Adam LR1e-5, weight decay1e-3; temperature không weight decay như config.
-FP32 deterministic, TF32 tắt, không augmentation hay thay GT.
-Class weight pixel cố định[1,4]; không chọn lại ngưỡng hay weight bằng test.
-Validation detection dùng ngưỡng0,5, mask argmax; không hậu xử lý/gate mask.
-Activation checkpointing áp dụng cho mọi lượt; kiểm tra admission cho cùng
  hai cập nhật giữ các giá trị loss bằng nhau chính xác khi bật/tắt cơ chế này.

| Lượt | Skip | Residual B vào RE | Lambda FL | Language |
|---|---|---|---:|---|
| A_no_skip | Không | Coupled | 1 | Tắt |
| B_skip | Concat4 cấp | Coupled | 1 | Tắt |
| C_skip_detach | Concat4 cấp | Detach | 1 | Tắt |
| D_manual_fl | Concat4 cấp | Detach | 0,25 | Tắt |
| E_language_legacy | Concat4 cấp | Detach | 0,25 | PAD pooling, LR có PAD, CMC raw dot |
| F_language_corrected | Concat4 cấp | Detach | 0,25 | EOS cuối, LR bỏ PAD, L2 CMC |

Tất cả có lambda_AR=lambda_FD=1. E/F có thêm lambda_LR=1e-4,
lambda_CMC=lambda_KL=0,01; đây là hệ số khởi đầu thủ công, chưa được tối ưu.
E/F bật gradient language về nhánh ảnh để kiểm tra đóng góp cho visual-only.
Detach B vẫn chặn đường về decoder qua residual, nhưng encoder dùng chung
vẫn nhận tác động language. Hai lượt E/F phải có hash mọi trọng số ban đầu
giống nhau. Positive CMC vẫn theo index, KL hai phía cùng học như trước.

So A→B kiểm tra skip, B→C kiểm tra detach, C→D kiểm tra lambda_FL.
So D→E/F kiểm tra bổ sung language với hệ số đã ghi. So E→F kiểm tra **gói ba
thay đổi** ngôn ngữ; không tách được đóng góp riêng EOS/PAD/L2 từ phép so đó.

Shape decoder A/B khác nhau, nên cùng seed không có nghĩa mọi trọng số
decoder giống nhau; encoder và detector khởi tạo tương ứng vẫn được ghi hash.
CMC tính similarity trên microbatch2; accumulation không tạo thêm negatives.

## Phạm vi kết luận

Đánh giá AUC, MSE tái tạo, foreground IoU/Dice trên fake, false-positive area
trên real, global pixel mIoU hai lớp và tài nguyên. Các loss LR có định nghĩa
khác giữa E/F nên không dùng scalar LR làm tiêu chí duy nhất để chọn model.

Một seed,128 ảnh train/128 validation và100 updates chỉ là pilot nội bộ.
Chưa đánh giá cross-generator, danh tính chưa gặp hoặc hiệu quả theo paper.
Theo dõi mốc50 và100; trạng thái cuối100 là kết quả chính, không chọn mốc đẹp
nhất sau khi nhìn validation. Chưa thay LayerNorm LD hoặc residual VIM hai lần.

## Tệp theo dõi

Runner: `tools/run_genface_ablation.py`.
Thư mục: `artifacts/genface_ablation_20261026/trials_001/`.

- `report.json`: trạng thái, ID ảnh, lịch batch, config từng lượt, loss và metric.
- `<tên_lượt>/final.pt`: checkpoint cuối schema thí nghiệm riêng, có config và
  language_mode; không phải schema CLI `evaluate.py` chính.
- `val_<lượt_trước>_vs_<lượt_sau>.npz`: tensor preview gốc để kiểm tra.
- Queue đã hoàn tất cả sáu lượt; kết quả ở phần dưới.

Kiểm tra bộ nhớ có language đã pass2 cập nhật trên GPU hiện tại, nhưng peak
CUDA allocated khoảng7,1 GiB ngay cả khi checkpoint activation, lớn hơn6 GB
VRAM vật lý. Đây là chỉ số cấp phát của CUDA, không đo trực tiếp residency
hoặc thời gian paging; cần ghi thời gian thực tế, không suy tốc độ từ FLOPs.

```powershell
python tools/run_genface_ablation.py --output artifacts/genface_ablation_20261026/trials_repeat --updates 100 --train-images 128 --val-images 128
```

Chọn output mới khi tái chạy. Không cập nhật checkpoint cũ hay source model
trong khi queue đang chạy; provenance được kiểm lại khi hoàn tất.

## Kết quả đã hoàn tất trên máy hiện tại

Trạng thái `COMPLETED`: 6/6 lượt, mỗi lượt đủ 100 cập nhật và 128 ảnh validation.
Không nạp trọng số đã train. Source giữ nguyên trong suốt queue, mọi loss được
ghi nhận đều hữu hạn. B/C/D có cùng hash trạng thái ban đầu; E/F cũng có cùng
hash toàn bộ trạng thái và metric ban đầu. A/D có cùng khởi tạo encoder và
detector, nhưng decoder khác cấu trúc. Cả sáu checkpoint `final.pt` đã được lưu.

**Bảng dưới là validation ở bước 100, không phải test hoặc kết quả paper.**
A là decoder resize-conv không skip; không đồng nhất với A0 dùng transposed convolution.

| Lượt | AUC (%) | MSE tái tạo ↓ | IoU vùng giả trên fake (%) ↑ | Vùng báo giả trên real (%) ↓ | Thời gian (giây) |
|---|---:|---:|---:|---:|---:|
| A: không skip | 61,87 | 0,083683 | 1,585 | 1,368 | 77,6 |
| B: thêm skip | 62,84 | 0,043693 | 1,029 | 0,910 | 179,3 |
| C: skip + detach B | 65,99 | 0,018745 | 0,850 | 0,572 | 175,9 |
| D: C + lambda_FL=0,25 | 66,75 | 0,011966 | 1,160 | 0,997 | 172,1 |
| E: D + language cũ | 40,77 | 0,013010 | 1,201 | 1,044 | 667,9 |
| F: D + EOS/PAD/L2 | 64,14 | 0,012300 | 1,328 | 1,108 | 732,7 |

Thời gian gồm train và các lần đánh giá sau khi bắt đầu train; không bao gồm
toàn bộ khởi tạo/ghi checkpoint. Số tham số A/B–D/E–F lần lượt khoảng
39,31/42,44/155,20 triệu. Peak CUDA allocated tương ứng khoảng
3.110/5.255–5.260/7.117–7.119 MiB; chỉ số cấp phát này không đo trực tiếp
residency trong VRAM hoặc thời gian paging.

### Điều đã cải thiện

- **A → B:** skip giúp MSE giảm khoảng 48%; ảnh AD giữ được cấu trúc khuôn mặt
  rõ hơn. AUC tăng nhẹ, nhưng IoU mask giảm.
- **B → C:** detach B trước RE giúp giảm thêm MSE và tăng AUC trong pilot này.
  Đây chỉ chặn gradient qua residual về AD; AR và FL vẫn cập nhật trunk decoder.
  Không có bằng chứng rằng mọi xung đột loss đã được giải quyết.
- **C → D:** giảm lambda_FL từ 1 xuống 0,25 cho MSE và AUC tốt nhất trong sáu
  lượt ở mốc cuối. IoU tăng từ 0,850% lên 1,160%, đồng thời báo giả trên real
  tăng. Vì thế chưa thể coi đây là lựa chọn tốt nhất cho localization.
- **E → F:** ba sửa đổi language cùng nhau nâng AUC 40,77% → 64,14%, MSE giảm
  nhẹ; IoU tăng nhỏ kèm false-positive area tăng. Tuy nhiên F chưa vượt D
  không language về AUC/MSE. Không tách được đóng góp riêng EOS, bỏ PAD và L2
  từ thí nghiệm gộp này; cũng chưa có nhiều seed để xác nhận độ ổn định.

### Điều chưa giải quyết được

**Mask vẫn gần toàn nền.** Kiểm tra ảnh xuất thấy các lượt chủ yếu dự đoán nền,
còn một ít vùng trắng ở biên ảnh. IoU fake chỉ 0,85–1,58% ở cả sáu lượt.
Recall pixel giả tổng thể chỉ khoảng 0,90–1,73%, tức bỏ sót hơn 98% pixel giả.
Train IoU cũng chỉ khoảng 0,78–1,53%; vấn đề hiện tại không chỉ xuất hiện trên
validation. Các con số này chưa xác định được một nguyên nhân duy nhất.

GT foreground chiếm 6,54% pixel validation. Nếu dự đoán toàn nền, global
two-class mIoU đã đạt 46,732%; các model hiện khoảng 46,84–46,92%. Vì vậy mIoU
tổng thể khoảng 47% không chứng minh model định vị tốt. Chỉ số này khác với
IoU foreground macro trên fake trong bảng.

AD cải thiện nhưng còn nhòe, sai màu và vân lưới trong các ảnh minh họa. MSE
thấp hơn không bảo đảm residual giàu tín hiệu giả mạo hơn hoặc mask chính xác hơn.
AUC cũng không phải accuracy: D đạt AUC 66,75% nhưng accuracy chỉ 50% ở
ngưỡng 0,5; F đạt AUC 64,14%, accuracy 52,34%. Chưa hiệu chỉnh ngưỡng trong
thí nghiệm này.

### Hướng xử lý tiếp theo được đề xuất

1. Giữ GenFace và dùng cấu hình D làm đối chứng gọn cho nhánh ảnh; giữ F làm
   đối chứng language đã sửa. Chưa thay cấu hình mặc định chỉ vì AUC của pilot.
2. Ưu tiên kiểm tra khả năng overfit mask trên 8–16 ảnh cố định: so FL-only,
   AR+FL và AR+FD+FL với cùng khởi tạo/lịch batch; theo dõi recall/precision,
   xác suất foreground và gradient từng phần. Chỉ đánh giá trên chính tập nhỏ
   này để chẩn đoán khả năng học, không coi là kết quả tổng quát hóa.
3. Nếu FL-only học mask được nhưng thêm AR làm kém, đo lại gradient AR/FL và
   đối chứng trunk AD/MD riêng. Nếu FL-only cũng không học, kiểm tra đường
   mask logits/GT, mức học và phân bố pixel trước khi tăng thêm độ phức tạp.
4. Sau khi mask học được trên tập nhỏ, mở rộng pilot và nhiều seed; tách riêng
   các ablation EOS/PAD/L2 trước khi kết luận về từng thay đổi language.

Đây là đề xuất bước tiếp; chưa tự động chạy thêm các thí nghiệm này. Toàn bộ
queue vừa hoàn tất chỉ dùng GenFace real + Diffae, không dùng FF++ hay test.

## Ảnh và bảng đã xuất

Trong `artifacts/genface_ablation_20261026/trials_001/`:

- `comparison.json`, `summary.md`: bảng tổng hợp cuối; có thêm train metric,
  precision/recall pixel, số tham số và bộ nhớ.
- `metrics_comparison.png`: biểu đồ bốn chỉ số validation.
- `training_losses.png`: đường loss AR/FD/FL chưa nhân lambda của từng lượt.
- `all_AD.png`: input và ảnh tái tạo của sáu lượt.
- `all_mask.png`: input, GT và mask dự đoán của sáu lượt.
- `val_D_vs_F.png`: đối chiếu trực tiếp nhánh ảnh gọn và language đã sửa.
- `val_*_vs_*.png/.npz`: các cặp liền nhau; tiêu đề PNG đã sửa theo đúng tên
  lượt, không dùng nhãn Shared/Separate CNN của helper thí nghiệm cũ.

Bốn mẫu minh họa được cố định trước tại index 0, 1, 126, 127 của validation,
gồm hai real và hai Diffae. Metric dùng đủ 128 ảnh. RGB hiển thị được clip
về [0,1]; metric MSE và tensor `.npz` dùng output gốc, không clip.

Tái xuất bảng/ảnh sau khi queue hoàn tất:

```powershell
python tools/summarize_genface_ablation.py --run artifacts/genface_ablation_20261026/trials_001
```

## Đối chứng sigmoid tiếp theo

Đã chạy thêm D tuyến tính và G sigmoid với cùng khởi tạo/data/lịch 100 cập nhật.
G đạt AUC 75,39% so với D 66,75%, nhưng MSE tăng 0,01197 → 0,03147 và IoU
fake giảm 1,16% → 0,61%. Sigmoid vẫn là tùy chọn thử nghiệm riêng.
[Chi tiết, giới hạn và ảnh D/G](genface_sigmoid_20261026_vi.md).
Source trước khi thêm sigmoid đã được giữ đúng hash trong `source_snapshot/`
của thư mục sáu thí nghiệm này; không ghi lại checkpoint hoặc metric lịch sử.
