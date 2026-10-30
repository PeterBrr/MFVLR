# Language trên GenFace5.000: train từ đầu, mục tiêu15epoch

Người dùng yêu cầu dùng5.000ảnh,15epoch và xác nhận rõ **không fine-tune checkpoint cũ**. Kế hoạch warm-start trước đó đã hủy, chưa upload gói158MB đó. Gói mới chỉ code/token/assets,1.483.479byte; không chứa trọng số đã train, ảnh hoặc test.

## Giao thức cuối

- 5.000train (2.500real/2.500Diffae),400validation cố định; tái dùng dataset riêng tư trước, không dùng test.
- Seed0, toàn bộ image/language khởi tạo từ đầu. Kiểm tra từng module image với image-only factory; hash image ban đầu được ghi theo phiên bản PyTorch. Cùng source cũ/mới trên local cho cùng hash; không áp hash random-init khác phiên bản PyTorch một cách mù quáng. Trên Kaggle, assert hash với baseline cũ cùng torch2.11.0+cu128.
- Target15epoch,18.750cập nhật, batch hiệu dụng4/microbatch2, AdamWD0. ImageLR1e-4 như baseline, language/adapter/temperatureLR1e-5 để khởi đầu thận trọng. LR ngôn ngữ là lựa chọn thử, không phải recipe tác giả.
- FD0,1/AR1/FL1/LR1e-4/CMC0,01/KL0,01; pixel weights[1,4]. CMC thực sự dùng microbatch2.
- Language **coupled** để có ảnh hưởng lên encoder ảnh; stop_all không thể kiểm chứng lợi ích language lên image bằng đường gradient này. Residual VIM double + pre_norm, EOS cuối, bỏ PAD LR, L2 CMC. AD tuyến tính/skip/resize-conv/detach reconstruction như baseline.
- Không đổi singleton K/V, full LE memory hoặc thuật toán pairing CMC. Chưa hiệu chỉnh trọng số tối ưu.

## Chia phiên và checkpoint

Phiên đầu chạy đến cuối epoch1 (1.250cập nhật). Target toàn thí nghiệm vẫn15epoch; epoch1 chưa phải kết quả cuối. Cần đo tốc độ/VRAM trên T4 để xác định số epoch trong các phiên sau. Quota kiểm tra trước launch còn18,39giờGPU; chưa thể bảo đảm đủ15epoch trong quota hiện tại.

Runner hỗ trợ `--resume` **checkpoint của chính lượt scratch mới**, kiểm tra config/source/data/assets và lịch batch, nạp model/optimizer/RNG rồi tiếp tục bước kế tiếp. Đây khác việc fine-tune từ model cũ. Checkpoint mỗi100cập nhật và mỗi epoch, ghi qua file tạm. Giới hạn mềm8giờ, timeout notebook10giờ. Nếu dừng theo stage, report làSTAGE_COMPLETED; nếu dừng do budget làPARTIAL_TIME_BUDGET, không gọi là hoàn thành15epoch.

Mỗi epoch đánh giá image và loss/token-accuracy language trên400validation, threshold detection0,5/mask argmax; không tuning theo test. Độ chính xác token teacher-forcing không chứng minh language sử dụng ảnh đúng ý nghĩa hoặc model tổng quát hóa.

Mode `--image-only` có sẵn để chạy control cùng runner (tắt LR/CMC/KL); chưa khởi chạy thêm control mới trong phiên này. Kết quả baseline5.000/30epoch cũ đã có metric ởepoch15 để tham khảo; nếu công bố tác động language, cần đối chiếu điều kiện và budget, không so language15 với image30 rồi quy khác biệt riêng cho language.

## Đóng gói và xác nhận

- Dataset ảnh đã có: `nha4von/mfvlr-genface5000-epoch30-20261027`.
- Dataset code/token riêng tư: `nha4von/mfvlr-language5000-scratch-assets-20261028`.
- Notebook: https://www.kaggle.com/code/nha4von/mfvlr-language5000-scratch15-20261028-train
- Bundle SHA256: `3b84ee9352a2be078e96e073e62992a9c60a7d3b7bc0577d4dbdf98ae1c479c2`.

Bộ duyệt tự động từ chối payload158MB warm-start cũ vì chưa xác nhận rõ nội dung/đích. Người dùng sau đó sửa yêu cầu sangscratch, rồi xác nhận **"Đồng ý tải lên và train từ đầu"** cho gói1,5MB mới và đích riêng tư trên. Upload thành công, API trảready. Không chuyển checkpoint cũ lên dataset mới.

Hai MCP Kaggle lỗi transport nên dùng CLI/API chính thức. API đã tạo notebook version1 thành công; việc đọc status gặp timeout mạng, nhưng SSE đã nhậnbootstrap hash/GPU vàdata_ready đúng5.000train/400val. Chưa khẳng định đã hoàn thành epoch nào khi chỉ cóbootstrap.

## Kiểm tra trước launch

Runner scratch đã chạy qua một cập nhật trên local16train/16val, toàn bộ gradient/tham số hữu hạn, lưu checkpoint đầy đủ. Smoke resume đã nạp lại model/optimizer/RNG của chính lượt mới, giữ số bước đã có (không train thêm trong smoke resume). Không coi các metric16ảnh là kết quả chất lượng của lượt5.000.

Artifact: `artifacts/language5000_scratch_kaggle_20261028/`; log `remote_live.log`, launch_record.json, package_001/ và local_smoke_002/.

```powershell
Get-Content artifacts/language5000_scratch_kaggle_20261028/remote_live.log -Tail 20 -Wait
```

Chưa commit/push. Các kết quả model image-only trước đó giữ nguyên.

## Cập nhật sau khi nhận log train thực tế

SSE xác nhận đã đánh giá epoch0 và cập nhật1–20/18.750 trênT4, loss hữu hạn. `scratch_initialization_verified` trênKaggle có hash image `712b6bbc1a1c627e52879b0a6f7fc6a2d53ed876c5da27b1f172f252503215a0`, khớp baseline cùngphiênbản. Không có trọng số lịch sử trong gói mới hoặc được nạp vào khởi tạo. Tốc độ ban đầu1,4–1,5giây/cập nhật, ước tính epoch1 khoảng30–35phút cộng biến độngI/O; chưa dùng để đảm bảo thời gian15epoch.

Epoch0 trên400validation: AUC42,375%, IoU10,325%, LR418,094, CMC0,6961, KL22,548, nonPAD token accuracy0%. Đây là model ngẫu nhiên ban đầu, không phải chất lượng sautrain. Không chọn ngưỡng theo metric này.

## Phiên1 hoàn thành epoch1, tiếp tục phiên2

Đã tải report và xác minh STAGE_COMPLETED, completed_updates1250, target_epochs15, source_unchanged=true. Đây chỉ là hoàn thành chặng đầu, chưa hoàn thành15epoch.

| Validation epoch1 | Image-only cũ cùng epoch | Language scratch |
|---|---:|---:|
| AUC | 97,62% | 98,66% |
| Accuracy threshold0,5 | 90,25% | 89,25% |
| IoU fake macro | 11,65% | 18,16% |
| Dice fake macro | 20,43% | 30,08% |
| MSE AD | 0,001791 | 0,002129 |
| Diện tích báo giả trên ảnh thật | 2,82% | 2,92% |

Chưa kết luận language tốt hơn sau một epoch: IoU/AUC tăng nhưng accuracy/MSE chưa tốt hơn. Trọng số image ban đầu/split/schedule seed0 và imageLR khớp; kiến trúc language/optimizer group mới được khai báo riêng. Baseline là lượt image-only cũ, chưa chạy thêm control cùng runner mới.

Language validation: LR418,094→5,519; CMC0,6961→0,1793; KL22,548→0,000869; nonPAD token accuracy99,743%. Prompt rất lặp theo hai nhãn và teacher-forcing/full LE memory nên token accuracy cao chưa chứng minh nhánh language hiểu ảnh hoặc giúp tổng quát hóa. Chưa dùng test.

Đã khởi chạy notebook phiên2: https://www.kaggle.com/code/nha4von/mfvlr-language5000-scratch15-stage2-20261028-train . Version1, kernelID141330576, private=true, GPU T4, Internet=false. Metadata xác nhận đúng hai dataset và kernel_sources từ output phiên1. Mục tiêu tiếp tục step1251 đến18750, epoch2–15; checkpoint model/optimizer/RNG của chính lượt scratch, không phải fine-tune model image-only lịch sử. Giới hạn mềm8giờ của runner vẫn có thể tạo thêm chặng nếu chưa đủ budget.

Report phiên1: `artifacts/language5000_scratch_kaggle_20261028/downloaded_stage1/language5000_results/report.json`.
Log phiên2: `artifacts/language5000_scratch_kaggle_20261028/remote_stage2_live.log`.
Script chuẩn bị/launch tiếp tục: `tools/continue_language5000.py`.

## Kết quả hoàn thành — kiểm tra ngày 29/10/2026

Kaggle phiên2 COMPLETE; report xác nhận COMPLETED, đủ 18.750 cập nhật/15 epoch, 5.000 ảnh train và 400 validation; source_unchanged=true. Có 16 lần đánh giá gồm epoch0. Khởi tạo scratch được xác minh khớp hash visual baseline; phiên2 tiếp tục checkpoint của chính lượt scratch. Log local dừng ở epoch9 do ngắt kết nối không phản ánh trạng thái cuối trên Kaggle.

So sánh validation tại cùng epoch15, seed0, cùng dữ liệu và ngưỡng detection0,5/mask argmax:

| Metric | Không language | Có language |
|---|---:|---:|
| AUC | 99,9075% | 99,9600% |
| Accuracy | 94,75% | 99,00% |
| IoU vùng giả macro | 33,57% | 35,56% |
| Dice vùng giả macro | 49,80% | 52,08% |
| MSE tái tạo AD | 0,00079936 | 0,00059086 |
| Diện tích báo giả trên ảnh thật | 0,81% | 1,48% |

IoU tăng 1,99 điểm phần trăm; Dice tăng 2,28 điểm; MSE giảm khoảng 26,1%. Tuy nhiên mask trên ảnh thật báo giả nhiều hơn, tăng 0,67 điểm phần trăm. IoU cao nhất trong lượt language là 36,00% tại epoch14; metric này không đồng nghĩa checkpoint epoch14 đã được lưu riêng. Chưa thể kết luận chắc chắn language tốt hơn với chỉ một seed và baseline lịch sử; chưa đánh giá test cho lượt language.

Language epoch15: LR=0,00646545; CMC=0,00094176; KL=0,00038589; nonPAD token accuracy=100%. Các prompt lặp theo nhãn và teacher forcing khiến token accuracy không phải bằng chứng độc lập về hiểu ảnh.

Report đã tải về máy: `artifacts/language5000_scratch_kaggle_20261028/downloaded_stage2/language5000_results/report.json`. Checkpoint vẫn ở output notebook Kaggle; bước kiểm tra này chỉ tải report, chưa tải checkpoint về máy.

## Test và xuất output — 29/10/2026

Đã tải final.pt (621.183.119 byte) về downloaded_stage2/language5000_results. SHA256: `fbd342d6ae0989c1b6d45f2240405865828f0ee1fdd9d957de4f9fb57bb3e772`. Dùng mã nguồn đóng băng từ gói đã train, kiểm tra hash source/data/config và nạp strict toàn bộ state_dict. Validation parity local so với Kaggle: AUC và IoU lệch0, MSE lệch -6,55e-12.

Test trên400 ảnh gốc, cân bằng200real/200Diffae; kiểm tra không trùng ID hoặc nội dung RGB sau resize với train/val. Chưa xác minh subject-disjoint. Test này đã dùng trong lịch sử dự án, không gọi là tập test hoàn toàn chưa từng xem.

| Metric | Validation epoch15 | Test epoch15 |
|---|---:|---:|
| AUC | 99,9600% | 99,9425% |
| Accuracy | 99,00% | 98,50% |
| F1 detection | 99,00% | 98,50% |
| IoU vùng giả macro | 35,56% | 36,14% |
| Dice vùng giả macro | 52,08% | 52,68% |
| Diện tích báo giả trên ảnh thật | 1,48% | 1,79% |
| MSE AD | 0,00059086 | 0,00059043 |

Test detection: TP197/TN197/FP3/FN3. Pixel precision toàn bộ=39,12%; recall=72,76%. Mask vẫn có xu hướng phủ rộng và báo giả ở các chi tiết/biên trên ảnh thật. Ảnh AD đã giữ được nội dung/màu tổng thể nhưng một số ví dụ còn vệt màu/ô lưới. Đây là quan sát preview, không phải bằng chứng xác định nguyên nhân.

Inference chỉ chạy visual_only trên trọng số được train có language, đúng đường đánh giá validation của runner. Không đưa prompt theo nhãn hoặc nhãn test vào model. Ngưỡng detection0,5 và mask argmax giữ nguyên, không tối ưu trên test. PNG AD clip[0,1] để hiển thị; MSE dùng output thô.

Output: `artifacts/language5000_scratch_kaggle_20261028/test_001/`:
- `comparison.png`:8 ảnh đầu theo thứ tự cố định, cột input/AD/GT/mask/overlay.
- `preview.png` và `preview.npz`:16 ảnh đầu, có AD và xác suất thô trong NPZ.
- `input/`, `AD/`, `GT/`, `mask/`, `overlay/`: mỗi thư mục400 PNG; mask255 là vùng giả.
- `probability/`:400 mảng float NPY224×224, xác suất pixel giả.
- `index.json`: ánh xạ tên file/sample_id, nhãn GT, score detection, diện tích mask và IoU từng ảnh.
- `test_metrics.json`: metrics và kiểm chứng nguồn/checkpoint/parity.

Baseline image-only đã test trước đây là epoch30 (IoU37,28%), còn lượt language này epoch15 (IoU36,14%); không coi đây là so sánh công bằng về tác động language. Đối chứng cùng epoch15 hiện có trên validation ở bảng phía trên.

Script: `tools/test_language5000_checkpoint.py`. Đã kiểm tra đủ400 output mỗi loại và xem contact sheet; chưa commit/push.
