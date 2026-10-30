# Đối chứng trọng số FD trên Kaggle — 26/10/2026

Mục tiêu: kiểm tra giảm lambda_FD có giảm tác động bất lợi lên mask khi encoder
dùng chung hay không. Không đổi kiến trúc hoặc thuật toán tối ưu giữa bốn lượt.

## Cấu hình

| Lượt | Lambda AR | Lambda FL | Lambda FD |
|---|---:|---:|---:|
| FD_0 | 1 | 1 | 0 |
| FD_0p1 | 1 | 1 | 0,1 |
| FD_0p25 | 1 | 1 | 0,25 |
| FD_1 | 1 | 1 | 1 |

- Cùng16 ảnh GenFace train (8real/8Diffae), seed0 và lịch batch; 400 cập nhật.
- Microbatch2, effective batch4; Adam LR1e-4, weight decay0; class weights[1,4].
- Skip concat, nearest resize-conv, AD tuyến tính, detach B trước RE; language tắt.
- GPU Tesla T4, PyTorch2.11.0+cu128/Python3.13.15; FP32 deterministic, TF32 tắt.
- Kaggle cấp hai T4 nhưng model chạy một GPU; không dùng DDP hoặc đổi batch.
- Môi trường khác máy local (torch2.7.1+cu118), nên FD_0 và FD_1 đều được chạy
  lại trên Kaggle để làm đối chứng; không so chênh lệch với kết quả local như
  bằng chứng riêng của lambda.
- Chỉ overfit trên train; không dùng val/test hoặc tuyên bố tổng quát hóa.

## Đo gradient

Tại khởi tạo và bước400, đo trên cùng4 cặp real/fake. Tính norm, cosine và norm
đã nhân lambda cho AR–FL, FD–FL và FD–AR ở CNN encoder chung/trunk/head.
FD=0 vẫn được đo đạo hàm FD thô để chẩn đoán, nhưng không đưa FD vào cập nhật.
FD vẫn phải có gradient None tại decoder vì detach. Nhánh transformer/global
của encoder không nằm trong nhóm CNN này; không diễn giải phép đo là bao phủ
toàn bộ encoder. Phép đo không thực hiện optimizer step.

## Log trực tiếp

- Python `-u`, `PYTHONUNBUFFERED=1`, mỗi cập nhật in một dòng JSON và flush.
- Metric trên16 ảnh in mỗi50 cập nhật; log thêm khởi tạo, cặp gradient và
  bắt đầu/kết thúc từng lượt.
- Remote lưu `fd_weight_results/live_log.jsonl`, `progress.json`, `report.json`.
- Kaggle CLI2.2.5 dùng SSE để đọc log ngay khi chạy; local ghi liên tục vào
  `artifacts/genface_fd_weight_kaggle_20261026/remote_live.log`.
- Stream đã nhận log GPU-ready và các optimizer updates thực tế, không chỉ
  dựa vào trạng thái RUNNING của MCP.

Xem file đang được ghi trong PowerShell:

```powershell
Get-Content "D:\New folder (2)\MFVLR\artifacts\genface_fd_weight_kaggle_20261026\remote_live.log" -Tail 20 -Wait
```

Để tự nối lại stream (chạy bằng Python base hỗ trợ CLI mới):

```powershell
& 'C:\Users\NHA\miniconda3\python.exe' -u tools/watch_kaggle_fd_logs.py
```

## Nguồn và trạng thái

Notebook riêng tư:
https://www.kaggle.com/code/nha4von/mfvlr-genface-fd-weights-20261026-001

Kernel ID140989250, version1, GPU bật/Internet tắt. Đã đọc metadata xác minh
đính kèm `nha4von/mfvlr-genface-loss16-20261026`. Dùng lại payload đã được người
dùng chấp thuận, không tải lại ảnh. Bootstrap xác minh hash bundle và source.

Wrapper SHA256: `5ee9136f087dd0573d3f2fe0409e3edba858b0f79bf5f7eddde268da856fcad0`.
Wrapper được nhúng vào notebook và lưu vào source provenance của từng checkpoint.
Dataset chứa source nền đã đóng băng; không phụ thuộc source local thay đổi sau đó.

Đã kiểm tra một bước CPU của FD_0p1: chạy thành công, xuất metric/log/gradient,
FD không có gradient trực tiếp tại decoder. Chưa coi đây là kết quả Kaggle.

Trạng thái cuối: **COMPLETE trên Kaggle, COMPLETED trong report; 4/4 lượt đủ400
cập nhật**, cùng hash khởi tạo, source giữ nguyên. Log trực tiếp đã kết thúc
sau khi ghi đủ từng bước của bốn lượt. Kết quả đã tải về máy.
Runner: `tools/run_fd_weight_ablation.py`; launcher: `tools/launch_kaggle_fd_weight.py`.
Không chạy lại launcher nếu chỉ muốn xem kết quả.

## Kết quả cuối bước400 trên16 ảnh train

| Lambda FD | IoU fake (%) ↑ | MSE ↓ | Precision pixel giả (%) | Recall pixel giả (%) | FP area real (%) | Accuracy train (%) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 45,777 | 0,003529 | 47,099 | 90,804 | 0,000 | 50* |
| 0,1 | 45,338 | 0,003953 | 59,829 | 64,605 | 0,000 | 100 |
| 0,25 | 40,527 | 0,004708 | 51,861 | 64,549 | 0,009 | 100 |
| 1 | 42,480 | 0,005747 | 44,592 | 86,159 | 0,113 | 100 |

\* FD=0 không tối ưu detection; accuracy50%/AUC51,56% của nhánh không được
giám sát chỉ là số chẩn đoán. Ba lượt có FD đều AUC/accuracy100% trên16 ảnh
đã train, không phải100% trên ảnh mới.

So FD0,1 với FD1: IoU tăng khoảng2,86 điểm phần trăm, MSE thấp hơn, precision
tăng nhưng recall giảm. Mask ít tô lan hơn về tổng thể, đồng thời bỏ sót nhiều
pixel giả hơn. FD0,1 có IoU gần FD0 (chênh khoảng0,44 điểm) và học được detection,
nên là **ứng viên đối chứng tiếp theo**, chưa phải trọng số tối ưu đã xác nhận.

Quan hệ không đơn điệu: FD0,25 cho IoU thấp hơn cả FD0,1 và FD1. Không thể nói
giảm FD càng nhiều thì mask chắc chắn càng tốt. Đây là một seed và tập overfit nhỏ.

## Gradient ở encoder CNN

Bốn cặp cố định là8 ảnh trong16 ảnh train, không phải toàn bộ tập.
Cosine dưới đây là trung bình bốn cặp ở cuối bước400; các group CNN và trunk
độc lập về tham số, còn group cha/con từng stage không được cộng lại.

| Lambda FD | Cosine FD–FL | Số cặp âm | Cosine FD–AR | Số cặp âm |
|---:|---:|---:|---:|---:|
| 0 | +0,140 | 0/4 | +0,031 | 1/4 |
| 0,1 | +0,369 | 1/4 | +0,172 | 1/4 |
| 0,25 | +0,297 | 0/4 | +0,053 | 1/4 |
| 1 | -0,145 | 3/4 | -0,121 | 3/4 |

FD0 được đo loss/gradient FD thô dù FD không tham gia tối ưu. Ở cùng trạng thái
model, nhân lambda dương không đổi cosine; cosine khác giữa các lượt vì quá
trình train đã tạo ra các trạng thái model khác nhau.

Khởi tạo có cosine FD–FL gần0 và cả dấu âm/dương, không phải luôn đối nghịch.
Kết quả cuối FD1 cung cấp bằng chứng xung đột cục bộ, nhưng chưa chứng minh
đó là nguyên nhân duy nhất làm giảm IoU: chưa đo gradient ở các mốc giữa quá
trình, và gradient Adam còn phụ thuộc moment tích lũy.

Ở cuối FD1, trung bình norm gradient đã nhân lambda trên CNN: FD≈0,01997,
FL≈2,20014, AR≈0,12213. FD không chiếm ưu thế về norm so với FL ở mốc này.
Không suy rằng FD đã áp đảo suốt400 bước chỉ từ việc cosine âm hoặc mask kém hơn.
Detach được assert trong mọi probe: FD không có gradient trực tiếp tại decoder.

## Nhận xét ảnh và hướng tiếp theo

AD vẫn có vân màu; đổi trọng số FD chưa giải quyết hiện tượng này. Mask đã học
được vùng giả nhưng còn thô và tô lan, chưa khớp các cấu trúc mảnh trong GT.
Chưa thêm sigmoid, Dice loss, boundary loss, PCGrad hay thay kiến trúc trong
thí nghiệm này.

Ưu tiên tiếp theo được đề xuất là xác nhận FD0,1 so với FD1 trên pilot lớn hơn
và nhiều seed, dùng validation tách biệt; nếu cần kết luận nguyên nhân thì đo
FD–FL/FD–AR thêm ở giữa quá trình. Chưa thay loss mặc định hoặc tự chạy thêm.

## Kết quả lưu tại máy

Thư mục: `artifacts/genface_fd_weight_kaggle_20261026/`.

- [Ảnh mask cùng input/GT](../artifacts/genface_fd_weight_kaggle_20261026/all_masks.png).
- [Ảnh AD](../artifacts/genface_fd_weight_kaggle_20261026/all_AD.png).
- [Biểu đồ metric](../artifacts/genface_fd_weight_kaggle_20261026/metrics.png).
- [Đường học](../artifacts/genface_fd_weight_kaggle_20261026/learning_curves.png).
- [Phân bố cosine4 cặp](../artifacts/genface_fd_weight_kaggle_20261026/gradient_cosines.png).
- [Bảng tổng hợp JSON](../artifacts/genface_fd_weight_kaggle_20261026/comparison.json).
- [Report đầy đủ](../artifacts/genface_fd_weight_kaggle_20261026/downloaded/fd_weight_results/report.json).
- [Toàn bộ log đã stream](../artifacts/genface_fd_weight_kaggle_20261026/remote_live.log).

Ảnh minh họa cố định index0,1,14,15; metric dùng đủ16 ảnh. PNG clip RGB về[0,1]
để hiển thị; MSE và tensor NPZ không clip. Các preview đã được assert có cùng
input và GT giữa bốn lượt. Checkpoint vẫn nằm trong Output notebook riêng tư;
chỉ report/log/preview đã được tải về để tránh tải lại hàng trăm MB trọng số.

Luồng log dùng chức năng SSE của [Kaggle CLI chính thức](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/kernels.md).
