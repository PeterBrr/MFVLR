# Đối chứng loss GenFace trên Kaggle — 26/10/2026

**Cập nhật:** người dùng đã yêu cầu chuyển về máy hiện tại. Đã gửi hủy Kaggle
session362484117; trạng thái sau đó là COMPLETE, không còn RUNNING. Không
diễn giải trạng thái này là hoàn tất bốn lượt train. Theo dõi lần chạy mới tại
[đối chứng local](genface_loss_overfit_local_vi.md). Các mục bên dưới lưu lịch sử chuẩn bị.

**Cập nhật:** người dùng đã yêu cầu chuyển về máy hiện tại. Đã gửi hủy Kaggle
session362484117; trạng thái sau đó là COMPLETE, không còn RUNNING. Không
diễn giải trạng thái này là hoàn tất bốn lượt train. Theo dõi lần chạy mới tại
[đối chứng local](genface_loss_overfit_local_vi.md). Các mục bên dưới lưu lịch sử chuẩn bị.

## Mục tiêu và cấu hình đã chuẩn bị

Kiểm tra khả năng overfit trên cùng 16 ảnh train GenFace: 8 real, 8 Diffae.
Chỉ đánh giá trên chính 16 ảnh này để chẩn đoán khả năng học, không dùng
validation/test hoặc kết luận khả năng tổng quát hóa.

| Lượt | Loss đang tối ưu |
|---|---|
| FL_only | FL |
| AR_only | AR |
| AR_FL | AR + FL |
| AR_FD_FL | AR + FD + FL |

- 400 cập nhật mỗi lượt, khoảng 100 epoch trên 16 ảnh.
- Cùng seed 0, hash khởi tạo và lịch batch; microbatch 2, effective batch 4.
- Lambda của loss bật bằng 1; pixel background/foreground weight [1,4].
- Adam LR 1e-4, weight decay 0 để kiểm tra overfit; khác optimizer của pilot
  128 ảnh trước nhưng giữ cùng điều kiện giữa bốn lượt mới này.
- Skip concat 4 cấp, nearest resize-conv, AD tuyến tính, detach B trước RE.
- Language tắt; AR/FL vẫn chia sẻ trunk decoder.
- Đo MSE, IoU/Dice fake, false-positive area real, precision/recall pixel và
  loss ở bước 50/100/.../400. AUC trên train chỉ là chẩn đoán, không chọn model
  bằng AUC của tập overfit.
- Đo norm và cosine AR/FL trên 4 cặp cố định tại khởi tạo và cuối mỗi lượt.
  Phép đo vẫn tính đạo hàm của AR/FL dù loss đó tắt khi train, nhưng không cập nhật.
- Lưu final checkpoint custom schema `mfvlr-loss-overfit-v1` và ảnh preview.

## Kiểm tra đã hoàn tất tại máy

`artifacts/kaggle_loss_overfit_20261026/cpu_smoke_001/report.json`:
4/4 lượt hoàn tất, mỗi lượt chỉ một cập nhật CPU và một cặp đo gradient.
Đường decode trực tiếp cho AR/FL khớp chính xác output của full visual route.
Đây là kiểm tra script, chưa phải kết quả của thí nghiệm 400 bước trên Kaggle.

Gói: `artifacts/kaggle_loss_overfit_20261026/package_001/`.
Manifest ghi sample IDs và hash source/payload. Tensor RGB uint8 được khôi phục
thành float32 chia 255 đúng bit với reader hiện tại; GT binary giữ nguyên.
Payload chỉ có các ảnh/GT đã chọn và source cần thiết, không chứa credentials,
file cấu hình Codex, tài liệu cá nhân hay toàn bộ dataset.

## Trạng thái tải lên và khởi chạy

- Kaggle MCP đã xác thực và đọc được quota GPU.
- Thử nhúng toàn bộ gói vào code bị Kaggle từ chối vì source notebook phải nhỏ
  hơn 1 MB. Chưa khởi chạy GPU.
- Đã chuẩn bị phương án tách gói khoảng 2 MB thành dataset RIÊNG TƯ mới:
  `nha4von/mfvlr-genface-loss16-20261026`.
- Đã tạo notebook riêng tư, GPU T4, Internet tắt, timeout 2 giờ, đính kèm
  đúng dataset đó. Không sửa các dataset GenFace/Diffae đang có của tài khoản.
- Bộ duyệt tự động từ chối lệnh tải gói, yêu cầu xác nhận rõ payload ảnh khuôn
  mặt/GT + source và đích lưu. Người dùng đã xác nhận rõ "Đồng ý tải lên và chạy";
  lần tải sau đó thành công. Dataset được xác minh `is_private=true`, version1 Ready.
- MCP save_notebook không lưu danh sách dataset đính kèm trong hai lần đầu;
  notebook version1/2 dừng trước train vì thiếu input. Dùng Kaggle CLI/SDK chính
  thức để lưu version3 đã giải quyết đính kèm. Metadata được đọc lại và assert:
  private=true, GPU=true, Internet=false, dataset_sources đúng dataset đã chấp thuận.
- **Version3 đang RUNNING theo lần kiểm tra MCP gần nhất.** Log chưa được API
  trả về, nên chưa xác nhận số cập nhật GPU hoặc có metric cuối. Không coi
  kiểm tra CPU một bước là kết quả train Kaggle.

Notebook: https://www.kaggle.com/code/nha4von/mfvlr-genface-loss-overfit-20261026-001

Dataset: https://www.kaggle.com/datasets/nha4von/mfvlr-genface-loss16-20261026

Kernel ID: `140976160`; version đang theo dõi: `3`.
Payload SHA256: `f5e3db0c5bb6df066af13e49c4e859eb362adfc81a39a74b710ad4f36edae0b9`.
Khi chạy xong, report/checkpoint/preview nằm trong `loss_overfit_results/`
ở Output của notebook; code và input đã giải nén nằm trong `mfvlr_loss_project/`.

## Tệp thực thi

- `tools/run_loss_overfit.py`: bốn lượt train và đo gradient.
- `tools/prepare_kaggle_loss_overfit.py`: chọn dữ liệu, kiểm tra tensor và đóng gói.
- `tools/upload_kaggle_loss_bundle.py`: tạo dataset mới bằng Kaggle CLI/SDK,
  `public=False`, sử dụng token từ môi trường; không ghi token vào payload.
- `tools/launch_kaggle_loss_overfit.py`: lưu và khởi chạy notebook với đúng
  đính kèm dataset qua CLI/SDK. Không chạy lại khi phiên3 còn hoạt động.

Lệnh upload đã thực hiện sau khi được xác nhận cho payload và đích ở trên;
không chạy lại để tránh cố tạo trùng dataset:

```powershell
python tools/upload_kaggle_loss_bundle.py --package artifacts/kaggle_loss_overfit_20261026/package_001 --owner nha4von
```
