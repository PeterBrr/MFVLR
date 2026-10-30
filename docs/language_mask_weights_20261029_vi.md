# Đối chứng trọng số pixel giả trên GenFace — 29/10/2026

Mục tiêu: kiểm tra giảm foreground weight có giảm mask tô dư và báo giả trên ảnh thật mà vẫn giữ IoU/recall hay không. Lượt language15epoch trước có test precision pixel39,12%, recall72,76%, nhưng thí nghiệm mới chỉ dùng train/validation; không chọn cấu hình theo test.

## Giao thức định trước

| Lượt | Nền / giả | Khởi tạo | Mục tiêu |
|---|---|---|---|
| FW4 | 1 / 4 | Lượt scratch đã hoàn thành | Đối chứng lịch sử15epoch |
| FW2 | 1 / 2 | Scratch seed0 | 15epoch, phiên đầu đến5 |
| FW1 | 1 / 1 | Scratch seed0 | 15epoch, phiên đầu đến5 |

Giữ nguyên5.000train (2.500real/2.500Diffae),400validation, lịch batch, seed0 và trọng số visual ngẫu nhiên ban đầu. Dùng lại hai dataset riêng tư đã được cho phép, không tải thêm ảnh/checkpoint cũ lên dataset. Không fine-tune từ FW4. Những phiên sau chỉ resume checkpoint của chính FW2 hoặc FW1.

Kiến trúc/loss khác không đổi: skip concat, nearest resize-conv, AD tuyến tính, detach reconstruction trước RE; language coupled, EOS pooling, bỏ PAD khỏi LR, L2CMC, pre-norm LE/LD, residualVIM2lần. Hệ số FD0,1/AR1/FL1/LR1e-4/CMC0,01/KL0,01; Adam imageLR1e-4/languageLR1e-5,weight_decay0; effective batch4/microbatch2,FP32 deterministic. Mục tiêu18.750updates mỗi lượt.

Lấy mã nguồn/assets đã đóng băng của lượt FW4, kiểm hash trước khi thay config. Sửa riêng foreground weight và thay dòng evaluator hardcode[1,4] bằng đọc trọng số từ config, để loss validation phản ánh đúng lượt. Không thay công thức CE: weighted sum chia tổng weight pixel trong microbatch. Cùng lambda_FL không có nghĩa norm gradient FL giữ nguyên khi đổi class weight; đây là đối chứng recipe trọng số.

Nguồn model/helper giữ nguyên; runner mới có hash riêng được ghi report. Hash khởi tạo visual và lịch batch phải khớp FW4; buffer class weight trong full model sẽ khác có chủ đích. MSE/IoU/AUC epoch0 phải khớp với đối chứng trong cùng môi trường; scalar FL epoch0 khác là hợp lệ.

## Đánh giá

- Dùng cùng400validation sau mỗi epoch. So cùng mốc5/15epoch; không so FW2/1epoch5 với FW4epoch15 rồi quy kết tác động weight.
- Chỉ số chính: IoU foreground macro trên ảnh giả, Dice, pixel precision/recall toàn split, diện tích báo giả real; kèm AUC/accuracy và MSEAD.
- Runner lưu confusion pixel; precision=TP/(TP+FP), recall=TP/(TP+FN), tính khi thu thập report. Không thay ngưỡng detection0,5 hoặc maskargmax.
- Không gate mask theo detection, không thêm Dice/boundary loss trong phép thử này.
- Chỉ chọn ứng viên nếu IoU cải thiện và đánh đổi recall/FP được báo rõ. Chưa có tiêu chí tự động tuyên bố tốt hơn từ một metric.
- Một seed/baseline lịch sử chưa đủ kết luận ổn định; nếu có ứng viên thì xác nhận thêm seed trước test. Subject-disjoint chưa xác minh; phạm vi chỉ GenFace/Diffae.

## Quota và phiên chạy

Khi chuẩn bị, Kaggle còn36.635,998giây≈10,18giờ GPU; làm mới31/10/2026 lúc00:00UTC (07:00Việt Nam). Hai lượt15epoch mới dự kiến vượt quota còn lại. Vì vậy chạy FW2/FW1 đến5epoch trước để có đối chứng cùng ngân sách; tổng mục tiêu15epoch không đổi, cần phiên tiếp theo/quota còn đủ để hoàn tất.

Mỗi chặng kết thúc lưu model/optimizer/RNG/schedule và report trạng tháiSTAGE_COMPLETED; trạng thái này không đồng nghĩa đã train đủ15epoch. Lưu checkpoint mỗi100updates và cuối epoch. Chặng tiếp tục dùng `--previous`, `--start-step`, `--stage`, `--end-epoch` của launcher; checkpoint phải đúng foreground weight và provenance.

Artifact: `artifacts/language_mask_weights_kaggle_20261029/`.
Launcher: `tools/launch_language_mask_weights.py`.
Notebook riêng tư:
- FW2: https://www.kaggle.com/code/nha4von/mfvlr-lang5000-fw2-s1-20261029-train
- FW1: https://www.kaggle.com/code/nha4von/mfvlr-lang5000-fw1-s1-20261029-train

Đây là kế hoạch và giao thức; trạng thái thực tế phải đọc launch_record, log bootstrap/train và report, không suy từ tên notebook. Không commit/push hoặc thay checkpoint FW4.

## Xác minh khởi chạy

Đã chạy local smoke FW2 trên16train/16val: một optimizer update thành công, gradient/tham số hữu hạn, lưu checkpoint và reportCOMPLETED. Hash visual khởi tạo khớp tham chiếu local cùngtorch. Smoke không phải kết quả chất lượng5.000ảnh.

Đã push thành công version1 cho cả FW2/FW1; Kaggle xác nhậnRUNNING. Metadata đọc ngược xác nhận private=true,GPU=true,Internet=false,đúng hai dataset và kernel_sources rỗng (không gắn checkpoint cũ). Log bootstrap cả hai xác nhận TeslaT4/torch2.11.0+cu128, foreground weight2/1, train5.000/val400, target15epoch/end5epoch và test_used=false. Runner patch SHA256 chung `67ce52a6608c75825ac5cad1851416311e1d92b2aaec85ad539f62fa81864fd5`.

Đã thấy data_ready và evaluate_start epoch0 của cả hai. Chưa có kết quả validation sau train khi ghi cập nhật này. Log local: `weight_2_live.log`, `weight_1_live.log` trong artifact gốc; mỗi update được flush trên Kaggle, cần nối lại watcher nếu client đóng.

Log tiếp theo xác nhận cả FW2/FW1 đã thực sự optimizer update (quan sát ít nhất3/6bước), loss hữu hạn. Hash visual khởi tạo cả hai khớp FW4 `712b6bbc1a1c627e52879b0a6f7fc6a2d53ed876c5da27b1f172f252503215a0`. Epoch0 cùng AUC0,42375, IoU0,10324972267078081, MSE0,6667249861359597 và language losses; chỉ scalar FL khác theo trọng số. Tốc độ đầu khoảng1,4giây/update; mốc5epoch dự kiến khoảng2,7–3giờ mỗi lượt, hai lượt chạy song song trên các phiên GPU riêng. Đây là ước lượng ban đầu, chưa phải thời điểm hoàn thành bảo đảm.

## Tiến độ: đã hoàn thành chặng5epoch của cả hai lượt

Kaggle cả hai COMPLETE; report tải về xác nhận STAGE_COMPLETED,6.250updates/lượt,source_unchanged=true,target_epochs15. Đây là kết thúc chặng5epoch, chưa phải hoàn thành15epoch; hiện hai phiên đã dừng. Hash dữ liệu và toàn bộ lịch18.750batch khớp FW4, hash visual khởi tạo cũng khớp. Log local cũ dừng ở khoảngstep1200 không phản ánh trạng thái cuối; báo cáo này dựa report output Kaggle.

So validation cùng mốc5epoch,400ảnh:

| Weight pixel giả | IoU fake macro | Dice fake macro | Precision pixel | Recall pixel | FP area real | AUC | Accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 (đối chứng) | 29,99% | 45,65% | 33,79% | 65,10% | 1,96% | 99,8875% | 98,00% |
| 2 | 30,68% | 46,35% | 46,07% | 44,28% | 1,07% | 99,7925% | 96,25% |
| 1 | 12,84% | 21,24% | 57,00% | 12,69% | 0,36% | 99,6550% | 95,75% |

FW2 tăng IoU khoảng0,68điểm phần trăm, precision tăng và báo giả real giảm; đổi lại recall giảm20,82điểm. FW1 quá dè dặt tại mốc này, bỏ sót phần lớn pixel giả. Chưa kết luận FW2 thắng cuối15epoch hay FW1 không thể học nếu train lâu hơn. Scalar FL không so trực tiếp vì class weight khác. Chưa dùng test trong các lượt mới.

MSE AD FW4/FW2/FW1:0,00266469/0,00107368/0,00165975.
Report tương ứng: `weight_2_stage_1/downloaded/language5000_results/report.json` và `weight_1_stage_1/downloaded/language5000_results/report.json` trong artifact gốc.

Quota đọc tại lần báo cáo này còn17.669,001giây≈4,91giờGPU, không có giờ reserved; chưa đủ hoàn thành cả hai lượt từ5→15epoch. Quota làm mới07:00ngày31/10/2026giờViệt Nam. Checkpoint chặng5 được runner lưu trong output từng notebook; chưa khởi chạy chặng2 trong lần kiểm tra tiến độ này.

## Tiếp tục theo yêu cầu sử dụng quota còn lại

Đã khởi chạy phiên2 của FW2/FW1, mỗi phiên từ checkpoint6250 của chính lượt tương ứng, mục tiêu tổng15epoch nhưng giới hạn8600giây (2giờ23phút20giây) cho runner mỗi phiên. Tổng ngân sách17.200giây; chừa469giây cho bootstrap và lưu checkpoint trong17.669giây quota còn lại. Đây là ngân sách dự kiến; quota máy chủ và biến động thời gian có thể khác. Runner chủ động checkpoint khi chạm ngân sách, ngoài checkpoint định kỳ100updates/cuối epoch. Không cố chạy đến khi máy chủ cưỡng bức ngắt ghi file.

Lệnh tiếp tục dùng launcher với `--stage 2 --start-step 6250 --end-epoch 15 --budget-seconds 8600 --previous <notebook_chặng1>`; giữ model, optimizer, RNG, lịch batch, data và config cũ. Chỉ sửa ngưỡng soft-time-budget của runner8giờ thành8600giây. Vì runner có hash mới, bootstrap kiểm tra hash runner cũ và toàn bộ source còn lại, chuyển riêng provenance runner trong bản sao checkpoint, ghi `resume_budget_audit.json` gồm hash checkpoint đầu vào/source cũ/mới. Không thay trọng số/optimizer/loss để vượt kiểm chứng; runner tiếp tục strict validation với provenance mới. Checkpoint đầu vào trên Kaggle giữ nguyên.

Notebook phiên2, version1, Kaggle xác nhậnRUNNING:
- FW2: https://www.kaggle.com/code/nha4von/mfvlr-lang5000-fw2-s2-20261029-train
- FW1: https://www.kaggle.com/code/nha4von/mfvlr-lang5000-fw1-s2-20261029-train

Log local: `weight_2_stage2_live.log`, `weight_1_stage2_live.log`; launch_record trong `weight_2_stage_2/`, `weight_1_stage_2/`. Dự kiến tiến đến khoảngepoch9 hoặc đầu10; chưa coi là kết quả thực tế cho đến khi đọc report. Khi chạm budget, trạng thái mong đợi PARTIAL_TIME_BUDGET với checkpoint mới, chưa hoàn thành15epoch.

Đã đọc ngược metadata xác nhận mỗi phiên2 chỉ gắn notebook chặng1 của đúng weight; private/GPU/dataset đúng. Cả hai đã in `resume_verified` ởstep6250 và có optimizer updates6251trở đi,epoch6,loss hữu hạn. Hash checkpoint đầu vào FW2=`a9c22609e05f14af1b65201f021e140dd206e63b891f65bf98b292b579d56674`, FW1=`8a3214e3a7788f24d60181eda5ee102cc79689d482e13fd383b2b1f609b797d1`; runner có budget mới hash `20d627350cd711bd9892891343f1af44e3e34d4e2ee3700cfef6468252e520c3`. Đã bắt đầu train tiếp thực tế, chưa có metric validation mới sauepoch5.

## Kết thúc phiên2 theo ngân sách quota — báo cáo mới nhất

Kaggle cả hai COMPLETE, nhưng report xác nhận PARTIAL_TIME_BUDGET, không phải hoàn thành15epoch. FW2 dừngstep11.818 (568/1.250bước củaepoch10, khoảng9,45epoch); FW1 dừngstep12.106 (856/1.250bước củaepoch10, khoảng9,68epoch). Report cả hai source_unchanged=true, dữ liệu và lịchbatch khớp FW4. Đã tải report vào `weight_2_stage_2/downloaded/language5000_results/report.json` và đường tương ứng FW1. Metric mới nhất của cả hai là cuối epoch9,step11.250; chưa có đánh giá cuối epoch10 và không gán metric epoch9 cho checkpoint cuối phiên.

So validation400ảnh tại cùng epoch9:

| Weight | IoU fake macro | Dice fake macro | Precision pixel | Recall pixel | FP area real | AUC | Accuracy | MSE AD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 34,43% | 50,78% | 37,52% | 71,95% | 2,24% | 99,9025% | 98,75% | 0,00126795 |
| 2 | 33,03% | 49,12% | 49,00% | 48,60% | 1,05% | 99,9775% | 98,50% | 0,00067544 |
| 1 | 20,78% | 33,49% | 58,65% | 22,96% | 0,40% | 99,8250% | 96,00% | 0,00065784 |

FW2 bớt tô dư và tăng precision, nhưng chưa vượt IoU FW4 ởepoch9 (thấp hơn1,41điểm phần trăm), đồng thời recall giảm23,35điểm. FW1 đã tăng IoU soepoch5 nhưng vẫn bỏ sót nhiều vùngGT. Không kết luận FW2 là cấu hình tốt nhất chỉ dựa epoch5; kết quả phụ thuộc mốc và đánh đổi precision/recall. Các lượt mới chưa dùng test, chưa đủ15epoch, một seed.

Quota API sau hai phiên: đã dùng107.665,686/108.000giây, còn334,314giây≈5phút34giây; không khởi chạy phiên GPU mới với phần lẻ này. Làm mới07:00ngày31/10/2026giờViệt Nam. Mục tiêu tiếp theo vẫn là tiếp tục từng checkpoint cuối phiên2 tới15epoch khi đủ quota, giữ nguyên recipe; bước bắt đầu lần sau phải là11.818/12.106, không phải11.250 hay6.250.

## Tiếp tục trên máy local theo yêu cầu

Đã chuẩn bị queue local tại `artifacts/language_mask_weights_local_20261029/run_001/`, supervisor `tools/continue_mask_weights_local.py`. Tiếp tục FW2 từstep11.818 rồi FW1 từ12.106, mục tiêu18.750updates/15epoch mỗi lượt; không train lại từ đầu và không dùngtest. Tải last.pt đầy đủ (model/optimizer/RNG/schedule/history) từ output phiên2 của đúng weight.

Source runner/model và assets được dựng lại byte-identical theo hash trong report Kaggle. Giữ nguyên giới hạn8600giây/chặng của runner để không sửa provenance; supervisor chạy FW2/FW1 tuần tự, luân phiên sau mỗi chặng cho tới khi cả hai COMPLETED. Mỗi chặng dùng output mới, checkpoint100updates/cuối epoch và cuối budget; supervisor tự resume chặng sau từlast.pt. Không chạy hai model đồng thời trên RTX4050.

Wrapper local kiểm tra checkpoint schema và khôi phục CPU RNG/GPU0 RNG. Kaggle có2GPU nhưng train chỉGPU0; local có1GPU nên bỏ trạng thái RNG củaGPU1không dùng khi gọi set_rng_state_all. Không thay tensor model, optimizer, lịch batch hoặc source để bỏ qua kiểm chứng. Ghi lại input checkpoint SHA256 trongqueue_status và hash wrapper/supervisor trongqueue_started.

Môi trường thay đổi: T4/torch2.11.0+cu128→RTX4050Laptop/torch2.7.1+cu118. Tiếp tục mixed-environment không bảo đảm bitwise tương đương train toàn bộ trênKaggle; khi so với FW4 cần ghi rõ giới hạn này. Không áp dụng augmentation/normalization mới.

Theo dõi `queue_status.json`, `queue.log`, `queue_error.log`; log train từng lượt trong `weight_2/segment_001.log`, `weight_1/segment_001.log` và các chặng tiếp. Queue được chạyẩn,PID ban đầu31752. Trạng thái WAITING_FOR_DOWNLOAD nghĩa là chưa bắt đầu train, không phải lỗi. Queue tự kiểm tra file tải hoàn tất trước khi nạp.

Lần nạp local đầu tiên dừng trước optimizer update do source provenance trên Linux dùng dấu `/`, còn runner khi tạo key trênWindows dùng `\`. Đã sửa wrapper chỉ chuẩn hóa key đường dẫn src trong metadata khi nạp, sau khi kiểm từng hash file; không sửa byte nguồn hoặc hash để bỏ qua sai khác nội dung. Giữ output lỗi tại `weight_2/segment_001.log`. Queue chạy lạiPID17864, logsupervisor `queue_retry.log`/`queue_retry_error.log`, logtrainFW2 `weight_2/segment_002.log`. Đã thấy resume_verified ởstep11.818, target15epoch trênGPUlocal. Checkpoint FW2 tải về1.865.764.125byte, SHA256 `504633d8c6aa9b7afb1bdf1e35c9fe11649695635fd2ac8d45e084a5d360aed6`.

Đã tải xong cả hai checkpoint. FW2 đã optimizer update thực tế11.819–11.823 ởepoch10, gradient/tham số hữu hạn; FW1 xếp hàng, chưa có cập nhật local khi ghi nhận này. Các bước sau warmup đầu khoảng9,6–10,1giây/update, chậm hơn Kaggle; chưa ước lượng hoàn thành chắc chắn từ vài bước đầu. Không có lỗi mới trongqueue_retry_error tại lần kiểm tra này. Giữ máy hoạt động để queue local chạy; đóng máy/sleep sẽ làm gián đoạn tiến trình local.

## Đã dừng theo yêu cầu người dùng

Đã dừng supervisor17864 và worker29780, kiểm tra cả hai PID không còn tồn tại. FW2 quan sát tớiupdate12.567, nhưng checkpoint đầy đủ gần nhất đã lưu làstep12.500/cuối epoch10 (`weight_2/segment_002/last.pt`). Nếu tiếp tục phải resume từ12.500, không coi67updates sau đó đã lưu. FW1 chưa train local, giữ checkpoint Kaggle12.106. Queue đánh dấu STOPPED_BY_USER; không tự khởi chạy lại.

## Người dùng yêu cầu tiếp tục train local

Đã thêm tùy chọn supervisor `--resume-stopped`: kiểm tra supervisor/worker cũ không còn chạy, giữ checkpoint và số segment đang có, không dùng lại checkpoint khởi đầu11.818 choFW2. Khởi chạy queuePID30524; FW2 dùng `weight_2/segment_002/last.pt` step12.500, output mới `weight_2/segment_003/`; FW1 vẫn chờ từ12.106. Mục tiêu15epoch và recipe giữ nguyên; không bổ sung augmentation hoặc đổi class weight.

Theo dõi supervisor `queue_resume_001.log`, `queue_resume_001_error.log`, trạng thái `queue_status.json`; trainFW2 `weight_2/segment_003.log`. Các file đều trong `artifacts/language_mask_weights_local_20261029/run_001/`. Lệnh tiếp tục này thay thế yêu cầu dừng trước đó.

## Báo cáo ngày30/10/2026: validation epoch10 và tiếp tục sau gián đoạn

FW2 chặng003 đã kết thúc PARTIAL_TIME_BUDGET ởstep13.739, lưu checkpoint/source_unchanged=true; chỉ còn11updates nữa tới mốc cuối epoch11. FW1 đã chạy tới12.743 trong log nhưng checkpoint bền vững gần nhất12.700. Khi kiểm tra, supervisor30524 và worker15864 không còn tồn tại, GPU không train; queue_status cònRUNNING là trạng thái cũ. Không suy nguyên nhân tắt/sleep nếu không có log hệ thống.

Metric mới nhất cùng mốc cuối epoch10,step12.500,400validation:

| Weight | IoU fake macro | Dice fake macro | Precision pixel | Recall pixel | FP area real | AUC | Accuracy | MSE AD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 31,66% | 47,57% | 30,67% | 80,83% | 4,22% | 99,9750% | 99,00% | 0,00100601 |
| 2 | 34,73% | 51,15% | 42,30% | 59,46% | 1,80% | 99,6075% | 97,75% | 0,00081436 |
| 1 | 26,01% | 40,64% | 59,23% | 30,22% | 0,35% | 99,8850% | 98,25% | 0,00061020 |

FW2 hơn FW4 về IoU3,06điểm và giảm báo giảreal, nhưng recall/detection thấp hơn. FW1 tiếp tục cải thiện IoU (20,78% ởepoch9→26,01% ởepoch10), vẫn bỏ sót nhiềuGT. Xếp hạng FW2/FW4 đã đổi giữaepoch9 và10; chưa chọn weight cuối cùng. So sánh đang mixed-environment: FW4 trainKaggle, FW2/FW1 tiếp tục local. Chưa đủ15epoch, chưa test các weight mới.

Yêu cầu tiếp tục train15epoch vẫn có hiệu lực, không có yêu cầu dừng mới. Đã khởi chạy phục hồi với `--resume-interrupted`, kiểm tra PID cũ không hoạt động và đọc step từ last.pt thật trước khi nạp; không dùng bước cuối log chưa lưu. QueuePID8236, logs `queue_recover_001.log`/`queue_recover_001_error.log`; FW2 output mới `weight_2/segment_004/`, FW1 sẽ output mới `weight_1/segment_002/`. Giữ recipe và checkpoint gốc, không đưa thêm augmentation.
