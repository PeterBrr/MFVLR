# Đối chứng loss GenFace trên máy hiện tại — 26/10/2026

Người dùng yêu cầu bỏ Kaggle và chạy tại máy. Đã gửi lệnh hủy Kaggle session
`362484117` của notebook `nha4von/mfvlr-genface-loss-overfit-20261026-001`;
API chấp nhận lệnh hủy. Không xóa notebook/dataset riêng tư đã tạo.
Kiểm tra sau lệnh hủy: Kaggle trả `COMPLETE`, không còn `RUNNING`. Trạng thái
này không được coi là bằng chứng bốn lượt Kaggle đã train hoàn tất.
Kiểm tra sau lệnh hủy: Kaggle trả `COMPLETE`, không còn `RUNNING`. Trạng thái
này không được coi là bằng chứng bốn lượt Kaggle đã train hoàn tất.

Đã khởi chạy GPU local RTX 4050 Laptop, Python môi trường `my_env`, output:
`artifacts/genface_loss_overfit_local_20261026/run_001/`.

Giữ nguyên gói 16 ảnh GenFace train: 8 real + 8 Diffae và GT đã kiểm tra.
Bốn lượt mới từ đầu: FL_only, AR_only, AR_FL, AR_FD_FL; 400 cập nhật mỗi lượt.
Cùng seed/hash khởi tạo và lịch batch; microbatch 2/effective batch 4; Adam
LR 1e-4, weight decay 0; lambda loss bật bằng 1; pixel weights [1,4].
Skip concat, AD tuyến tính, detach B trước RE; không language hoặc sigmoid.

Đây là phép overfit trên train, không dùng validation/test. Theo dõi MSE,
IoU/Dice fake, vùng báo giả real, precision/recall và gradient AR/FL. Không
diễn giải AUC trên 16 ảnh train là khả năng tổng quát hóa của mô hình.

## Kết quả hoàn tất

Trạng thái `COMPLETED`: 4/4 lượt đủ 400 cập nhật trên CUDA, không lỗi. Hash trạng
thái khởi tạo giống nhau, source giữ nguyên, toàn bộ loss history hữu hạn.
Mỗi lượt đã lưu `final.pt`, `preview.png`, `preview.npz`, gradient đầu/cuối và
metric mỗi 50 cập nhật. Các số dưới là cuối bước400, trên chính 16 ảnh train.

| Lượt | MSE tái tạo ↓ | IoU fake (%) ↑ | Dice fake (%) ↑ | Recall pixel giả (%) | Báo giả trên real (%) |
|---|---:|---:|---:|---:|---:|
| FL_only | 0,725601* | 1,620 | 3,139 | 1,722 | 0,257 |
| AR_only | 0,000386 | 13,848* | 24,229* | 100,000* | 99,997* |
| AR_FL | 0,003816 | 48,810 | 65,488 | 87,029 | 0,000 |
| AR_FD_FL | 0,005638 | 37,333 | 54,227 | 82,991 | 0,428 |

\* FL_only không tối ưu tái tạo. AR_only không tối ưu mask: dự đoán gần toàn
foreground, kể cả ảnh thật, nên IoU/recall của nó không chứng minh định vị tốt.
Các metric vẫn được đo để chẩn đoán tác động lên nhánh không được giám sát.

AR_FD_FL đạt AUC=accuracy=F1=100% và FD CE=0,000215 trên16 ảnh đã học. Đây là
khả năng ghi nhớ train, không phải kết quả test/validation.

### Diễn giải

1. **Không thể kết luận AR chỉ gây hại cho FL.** AR_FL tốt hơn FL_only rõ rệt
   về mask trong lần chạy này. FL_only đã có IoU8,93% ở bước100 nhưng cuối còn
   1,62%; tối ưu riêng FL chưa ổn định với LR1e-4/batch/loss hiện tại. Cần xem
   đường học hoặc thử LR khác trước khi quy kết lỗi kiến trúc.
2. AD có thể tái tạo sát input trên tập nhỏ: AR_only MSE0,000386. Thêm FL làm
   MSE tăng lên0,003816, đồng thời học được mask. Đây là đánh đổi hai mục tiêu
   tại cấu hình đang dùng, chưa chứng minh cần tách decoder.
3. Thêm FD vào AR_FL làm IoU giảm48,81% →37,33% và MSE tăng0,003816 →0,005638,
   trong khi detection ghi nhớ được16 ảnh. Cần ưu tiên đo FD với AR/FL tại
   encoder chung hoặc thử giảm lambda_FD. Detach B chỉ chặn FD qua residual về
   decoder; FD vẫn cập nhật encoder dùng chung và qua đó có thể ảnh hưởng mask.
4. Cosine AR/FL trên trunk ở cuối AR_FL có3/4 cặp dương,1/4 cặp âm, trung bình
   khoảng0,034. Ở AR_FD_FL có1/4 cặp dương,3/4 âm, trung bình khoảng-0,015.
   Các giá trị gần0 và phụ thuộc trạng thái/batch; không phải luôn đối nghịch.
   Đó là đạo hàm thô của AR/FL, chưa phải phép đo xung đột FD với hai loss này.
5. Bốn trăm cập nhật trên16 ảnh, một seed chưa phân biệt được ảnh hưởng ổn
   định của từng loss trên GenFace nói chung. Không so trực tiếp với pilot128
   ảnh vì LR/weight decay/budget và tập ảnh khác nhau.

Ưu tiên tiếp theo được đề xuất: giữ AR_FL làm đối chứng, thử thêm FD với
lambda nhỏ hơn trong cùng điều kiện và đo gradient FD–AR/FD–FL ở encoder.
Chưa tự chạy thêm thí nghiệm hoặc thay loss mặc định.

### Ảnh đã xuất

- [FL-only](../artifacts/genface_loss_overfit_local_20261026/run_001/FL_only/preview.png).
- [AR-only](../artifacts/genface_loss_overfit_local_20261026/run_001/AR_only/preview.png).
- [AR+FL](../artifacts/genface_loss_overfit_local_20261026/run_001/AR_FL/preview.png).
- [AR+FD+FL](../artifacts/genface_loss_overfit_local_20261026/run_001/AR_FD_FL/preview.png).
- [Report đầy đủ](../artifacts/genface_loss_overfit_local_20261026/run_001/report.json).

Preview có4 ảnh cố định (index0,1,14,15), metric dùng đủ16 ảnh. RGB PNG được
clip về[0,1] để hiển thị; MSE và reconstruction trong NPZ dùng giá trị gốc.

Lệnh đã khởi chạy, không chạy lại cùng lúc:

```powershell
python tools/run_loss_overfit.py --data artifacts/kaggle_loss_overfit_20261026/package_001/train16.npz --config artifacts/kaggle_loss_overfit_20261026/package_001/config.json --output artifacts/genface_loss_overfit_local_20261026/run_001 --updates 400 --device cuda --gradient-pairs 4
```
