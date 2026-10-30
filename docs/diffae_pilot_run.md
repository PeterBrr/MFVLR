# Chạy pilot Diffae

Biến thể resize + convolution, chặn gradient language và trọng số foreground:
xem [hướng dẫn riêng](resize_conv_variant_vi.md). Config mới là
`configs/diffae_pilot_resize_conv.json`; các config A0 bên dưới giữ nguyên recipe.

Pipeline mới phục vụ pilot kỹ thuật theo yêu cầu người dùng, dùng model A0 và sáu objective hiện có. Các acceptance/config lịch sử không bị sửa; G01 vẫn FAIL và chưa có bằng chứng hội tụ. Không có lượt train A0 thực tế nào được chạy khi chuẩn bị các file này.

## 1. Chuẩn bị

```powershell
conda activate my_env
cd "D:\New folder (2)\MFVLR"
python tools/finalize_diffae_pilot.py
```

Lệnh freeze đọc `checked_candidates.jsonl` và `decode_report.json` đã tạo trước đó, loại bản trùng trong cùng split, từ chối trùng xuyên split, ghi manifest trong `artifacts/diffae_pilot/manifests`. Hiện có 4,999 train, 400 val, 400 test. Chạy lại với nội dung không đổi được; không ghi đè manifest đã freeze bằng nội dung khác. Chưa bù ảnh train bị loại.

`train.py`/`evaluate.py` tự tìm `src` và `.pilot_deps` theo vị trí file, không cần đặt PYTHONPATH thủ công. `my_env` đã có torch 2.7.1+cu118; dependency tokenizer đúng pin đã nằm trong `.pilot_deps`. Trên máy khác, dùng môi trường CUDA phù hợp và `requirements-pilot.txt`. NumPy/Pillow/runtime được ghi vào provenance khi chạy; file requirements không phải toàn bộ CUDA environment lock.

## 2. Kiểm tra dữ liệu rồi chạy tiny

```powershell
python train.py --stage tiny --check-data
python train.py --stage tiny
```

`--check-data` kiểm tra hash và đọc 32 ảnh train được chọn cố định, 400 val, 400 test, kiểm tra token; không dựng model. Train subset 32 ảnh có 16 real + 16 Diffae; batch 2, FP32, Adam LR 1e-6, tối đa 100 bước. Đầu ra mặc định `artifacts/diffae_pilot/runs/tiny`.

Lệnh có thể dừng do OOM hoặc điều kiện finite/determinism không đạt trên GPU 6 GB. Script ghi phase/error vào `status.json`, không tự giảm batch, đổi precision/loss, clipping hoặc chuyển CPU. Peak memory đầy đủ chưa được đo. Nếu dừng trước checkpoint đầu tiên thì không có checkpoint để resume.

Đo ngắn trước khi cấp ngân sách 100 bước:

```powershell
python train.py --stage tiny --max-steps 2 --output artifacts/diffae_pilot/runs/probe
```

Đây là 2 optimizer steps thực tế, không phải forward-only profile. Chỉ chạy khi muốn thực hiện update. Resume probe đến tổng 100 bước:

```powershell
python train.py --stage tiny --max-steps 100 --resume artifacts/diffae_pilot/runs/probe/last.pt --output artifacts/diffae_pilot/runs/probe
```

## 3. Pilot 5 epochs

Sau khi tiny ổn định và đủ bộ nhớ:

```powershell
python train.py --stage pilot
```

Đầu ra `artifacts/diffae_pilot/runs/pilot`. Pilot khởi tạo lại A0 từ đầu, LR 1e-5, tối đa 5 epochs trên 4,999 ảnh; không tự fine-tune từ tiny. Một epoch có 2,499 bước batch 2, bỏ 1 sample cuối theo thứ tự shuffle thay đổi mỗi epoch; 5 epochs có 12,495 optimizer steps. Các tham số trong `configs/diffae_pilot.json`.

Sáu loss giữ nguyên hệ số 1 và reductions. CMC dùng batch thật 2. Adam weight decay 1e-3, loại `objectives.cmc.log_temperature` khỏi decay; không scheduler trong pilot 5 epochs, không AMP/augmentation/clipping. Bộ khởi tạo model hiện cố định seed 0; CLI từ chối seed khác thay vì giả vờ thay seed model.

## 4. Checkpoint và resume

```powershell
python train.py --stage pilot --resume artifacts/diffae_pilot/runs/pilot/last.pt
```

- `last.pt`: lưu atomic mỗi 200 bước, cuối epoch và cuối budget; lưu model/Adam/RNG/cursor/epoch/step/config/manifest/source/runtime provenance.
- `best.pt`: checkpoint tốt nhất sau validation hoàn tất mỗi epoch, chọn AUC rồi FD CE thấp hơn; hòa thì giữ epoch sớm hơn. Threshold chọn từ validation bằng F1 trên grid 0..1 bước 0.01, hòa ưu tiên gần 0.5.
- Terminal hiển thị thanh tiến trình tổng số bước, epoch hiện tại, tốc độ/ETA, total loss, FD, FL và learning rate. Không in một dòng JSON cho mỗi bước.
- `history.jsonl`: sáu loss, total, gradient norm, LR, elapsed, peak CUDA, kết quả validation. Có thời gian chờ dữ liệu, forward/loss, backward/kiểm tra gradient, optimizer/kiểm tra trạng thái và sự kiện lưu checkpoint để tìm bottleneck. Các thời gian là wall-clock host với các điểm đồng bộ kiểm tra finite, không phải CUDA profiler. Khi resume từ checkpoint cũ, log có thể chứa các dòng bước của lần chạy gián đoạn sau checkpoint; dùng checkpoint step/status để phân biệt.
- Resume yêu cầu cùng source/config/stage/runtime/device/manifests và recipe (ngoại trừ `--max-steps`). Thứ tự mẫu khôi phục theo epoch seed và cursor, num_workers=0. Chỉ tải checkpoint do pipeline này tạo. Kiểm tra next-step equality đã thực hiện trên model nhỏ CPU; chưa xác nhận exact resume full A0/CUDA.
- Khi chạy mới với output mặc định, script tự chọn thư mục trống tiếp theo (`pilot_001`, `pilot_002`... hoặc `tiny_001`...) và in đường dẫn trước khi dựng model. Các file cũ được giữ lại. Nếu truyền `--output` cụ thể đã có nội dung, script yêu cầu chọn một đường dẫn mới. Đừng đổi config hoặc sửa mã nguồn giữa run nếu cần exact resume.

## 5. Đánh giá

```powershell
python evaluate.py --checkpoint artifacts/diffae_pilot/runs/pilot/best.pt --split test --output artifacts/diffae_pilot/runs/pilot/test_metrics.json
```

Evaluation dựng `ImageOnlyMFVLR`, lấy đúng MVE/decoder/detector từ checkpoint full model, strict-load, không dựng FLT hoặc dùng prompt/nhãn làm đầu vào. Threshold test lấy từ validation đã lưu trong checkpoint. Tiny chưa hoàn thành epoch validation không được dùng để test.

Metrics: ROC-AUC (tính ties), ACC, F1, confusion matrix; foreground IoU/Dice macro chỉ trên fake, false-positive area trên real. Empty/empty fake được tính 1; không gọi foreground IoU là paper mIoU. Evaluation source và manifest phải khớp checkpoint, hash file ảnh/mask được kiểm tra khi đọc.

## Protocol dữ liệu và giới hạn

Ảnh decode RGB, resize bilinear trực tiếp về 224×224, float32 [0,1]; mask binary 0/1 hoặc 0/255 chuyển foreground bằng >0, resize nearest-neighbor. Không thêm crop/alignment hoặc tạo mask thiếu. Kích thước ảnh-mask nguồn phải khớp. Real mask phải toàn nền. Prompt Real/Diffae qua registry và tokenizer CLIP đã pin, 4×77 tokens. Đây là quy ước pilot được ghi rõ, không chứng minh mask gốc hoặc preprocessing giống paper.

Chưa có metadata source/identity hoặc chứng cứ nguồn gốc mask. Kiểm tra exact content và split không bảo đảm subject-disjoint. Các kết quả chỉ là nội bộ; không tuyên bố tái lập độ chính xác paper.

## Kiểm tra phần mềm

```powershell
$env:PYTHONPATH = "$PWD\.pilot_deps;$PWD\src"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest tests/pilot --confcutdir=tests/pilot -q
```

Test độc lập kiểm tra freeze/dedup/leakage, transform/token/hash, metric ties và checkpoint atomic; model nhỏ kiểm tra update/save/resume bằng nhau ở bước tiếp theo. Không chạy bộ test lịch sử chứa full-model optimizer. Không coi các test này là bằng chứng full model vừa GPU hoặc học được dữ liệu.
