# Kế hoạch train MFVLR với GenFace cục bộ

Ngày khảo sát: 17/10/2026. Dữ liệu: `D:\BaiduNetdiskDownload`.
Phạm vi hiện tại: khảo sát và lập kế hoạch; chưa chạy model, backward hay optimizer.

**Cập nhật khi bắt đầu pilot:** Người dùng đã chọn train riêng theo generator và yêu cầu bắt đầu pilot Diffae. Đã xác nhận môi trường `my_env`. Báo cáo kiểm kê ban đầu bên dưới tính nhầm thư mục real `masks` vào image và chưa loại staging, nên số real/ambiguity tổng cũ không được dùng chọn mẫu. Công cụ mới `tools/prepare_diffae_pilot.py` chỉ dùng thư mục real `images`, ghép `masks` riêng và loại staging. Trạng thái triển khai được ghi ở [báo cáo khởi động pilot](diffae_pilot_start.md).

## 1. Kết luận và hiện trạng

Ưu tiên pilot GenFace đang có thay cho tải FF++ mới. Đây là cập nhật đề xuất dựa trên việc thư mục cục bộ thực sự có các nhánh mask; chưa xác nhận chất lượng hoặc nguồn gốc mask. Giữ A0 và sáu objective hiện có; mục tiêu đầu tiên là một checkpoint nhỏ, train/resume được và đánh giá trên split độc lập.

- GPU đã kiểm tra: RTX 4050 Laptop, 6,141 MiB VRAM. Chưa đo peak memory của MFVLR; không khẳng định full training vừa GPU.
- Ổ D còn khoảng 96.13 GiB tại thời điểm quét. RAM chưa xác minh được do quyền truy cập hệ thống.
- Python hiện được shell chọn là `C:\msys64\ucrt64\bin\python.exe`, không có torch/Pillow/psutil. Điều này không kết luận rằng mọi môi trường Python trên máy đều thiếu các gói này.
- README là tài liệu lịch sử. `REPOSITORY_STATUS.md` và D02B10/D02B11 phản ánh trạng thái mới hơn.
- `PreparedTrainer.fit/step/backward` chưa triển khai thực thi. Loader mới nhận tensor trong RAM; checkpoint hiện có không lưu trạng thái optimizer để resume train.
- Full A0 có 154,156,232 tham số theo báo cáo tích hợp. G01 còn FAIL kiểm tra cộng gradient; LR objective và gradient từng rất lớn. Chưa có bằng chứng hội tụ.

## 2. Dữ liệu thực tế

Quét chỉ đọc toàn bộ nhánh real/fake, đối chiếu ba manifest. Báo cáo máy đọc tại `artifacts/genface_planning_20261017/inventory.json` (artifact cục bộ, có thể bị Git ignore). Các số sau là bản chụp danh sách tên file, **không phải số mẫu đủ điều kiện train**. Chưa kiểm tra decode, kích thước file, hash, binary mask hay identity.

| Split | Dòng manifest | Tìm thấy tên ảnh | Thiếu tên ảnh | Fake có tên mask cùng nhánh | Fake thiếu tên mask | Dòng có nhiều đường dẫn ảnh |
|---|---:|---:|---:|---:|---:|---:|
| Train | 278,329 | 184,557 | 93,772 | 126,874 | 33,500 | 24,440 |
| Val | 34,790 | 23,157 | 11,633 | 13,254 | 6,910 | 2,993 |
| Test | 33,156 | 22,275 | 10,881 | 15,768 | 3,683 | 2,823 |

Real tìm thấy theo manifest: 24,183 / 2,993 / 2,824. Nhiều basename real có nhiều đường dẫn; không lấy file đầu tiên làm mặc định. Không phát hiện basename trùng giữa ba split, nhưng điều đó không chứng minh tách biệt danh tính hay ảnh nguồn.

| Nhánh fake | File ảnh | File mask |
|---|---:|---:|
| AM / Diffusion | 55,423 | 55,166 |
| AM / GAN | 35,685 | 26,000 |
| EFS / Diffusion | 28,489 | 26,000 |
| EFS / GAN | 28,000 | 28,000 |
| FS / Diffusion | 42,846 | 40,055 |
| FS / GAN | 15,001 | 0 |

Không dùng báo cáo `genface_report` cũ làm số liệu hiện tại: số file đã thay đổi và script cũ chỉ quét AM/EFS/FS, không tính real.

Các điểm cần xử lý trước:

1. Xác định đường dẫn canonical của real giữa CelebA_HQ/FFHQ và các bản sao. Đối chiếu nhãn bốn cấp với tài liệu phát hành; không suy ra toàn bộ bảng mã từ số nguyên hay prefix.
2. Ghép mask theo nhánh + generator + sample/source ID, xác nhận duy nhất; kiểm tra màu/giá trị, kích thước, hướng và overlay ảnh-mask. Không thay mask thiếu bằng ảnh trắng hoặc mask toàn 0.
3. Kiểm tra nguồn mask và quy trình tạo: mask thư mục cục bộ không tự động tương đương mask trong bài MFVLR. Real có thể dùng mask nền toàn 0 khi quy ước dữ liệu đã được xác nhận; EFS phải khớp semantics foreground toàn ảnh của baseline.
4. Loại/quarantine file lỗi, nhãn không rõ và cặp mơ hồ; báo cáo số bị loại theo split/generator. Không ép đủ quota bằng cách bỏ kiểm tra.
5. Hash ảnh gốc, mask và tensor; kiểm tra bản sao/nội dung gần trùng và nhóm source/identity. Với face swap phải tính cả source và target. Giữ split phát hành nếu hợp lệ; loại xung đột trước chọn pilot. Thiếu metadata identity phải ghi rõ, không tuyên bố subject-disjoint.
6. Registry hiện chưa có `StyleGAN2` và `IA-FaceS`, dù tên file này có trên đĩa. Không ánh xạ chúng thành StyleGAN3/LatentTransformer. Các alias như LatDiff/CollDiff/LatTrans cần bảng mapping có chứng cứ.

GenFace công bố giới hạn sử dụng nghiên cứu phi thương mại và phân phối dữ liệu tại [repository chính thức](https://github.com/Jenine-321/GenFace#agreement), kiểm tra ngày 17/10/2026. Không đưa ảnh vào Git.

## 3. Lộ trình triển khai và điều kiện chuyển bước

| Bước | Công việc / đầu ra | Điều kiện hoàn thành |
|---|---|---|
| 1 — Environment | Chọn Python tương thích, tạo môi trường riêng; kiểm tra dependency pins, tokenizer, snapshot và artifact ngoài Git; ghi versions/device/RAM/disk | Import được package; xác minh CUDA bằng torch thực tế, không suy ra từ dòng CUDA của nvidia-smi |
| 2 — Dataset audit | Manifest canonical JSONL, bảng nhãn/prompt, exclusion report, hash, kiểm tra mask và leakage | Mọi sample được chọn có đường dẫn duy nhất, provenance và split hợp lệ |
| 3 — Loader | Dataset đọc file theo nhu cầu, transform ảnh/mask đồng bộ, collator bảo toàn ID, cache token | Tensor RGB float32 [3,224,224] trong [0,1], mask int64 [224,224] chỉ 0/1; token 4×77; đúng thứ tự sample |
| 4 — Numerical/resource gate | Xem xét lỗi G01 còn tồn tại; diagnostic same-forward có giới hạn; forward/backward không update trên real + fake ở batch 2 | Không lỗi cấu trúc, NaN/Inf; ghi cả sáu loss, gradient và VRAM. FAIL hữu hạn phải có quyết định xử lý trước update, không tự sửa ngưỡng |
| 5 — Trainer/resume | Dùng `with_language` → `compute_objectives(SuppliedTargets)`; thêm optimizer, logging, checkpoint atomic và resume | Kiểm tra update hữu hạn và next-step resume trên fixture nhỏ; không viết lại sáu loss trong trainer |
| 6 — Tiny run | 32 ảnh cố định, 16 real + 16 fake đã xác minh; tối đa 100 optimizer steps | Kiểm tra update, xu hướng từng loss và khả năng học tập nhỏ; không dùng kết quả để tuyên bố chất lượng tổng quát |
| 7 — Pilot | 2,000 train / 400 val / 400 test; balanced real/fake; train/val tối đa 5 epochs | Checkpoint best/last và báo cáo validation; chỉ đánh giá test sau khi chốt recipe |
| 8 — Mở rộng | Train đầy đủ tập hợp lệ; thí nghiệm cross-generator riêng; seed 0/1/2 khi ngân sách cho phép | Chốt protocol, batch thật và tài nguyên; báo cáo độ phủ dữ liệu thiếu và khác biệt với paper |

Pilot ưu tiên **Real + Diffae (AM/Diffusion)** để kiểm tra localization có vùng sửa cục bộ; nếu audit không đạt, chọn Real + DiffFace có provenance đầy đủ. Không trộn mọi generator ngay ở lượt đầu. Quota là mục tiêu sau audit, chưa phải số mẫu đã chọn. Giữ DiffFace hoặc generator khác làm miền đánh giá riêng khi audit cho phép; không dùng miền đó chọn checkpoint.

Preprocessing phải được chốt trước code: xác minh ảnh đã crop/aligned hay chưa; chỉ resize nếu protocol cho phép, ảnh dùng interpolation được ghi rõ, mask nearest-neighbor. Không tự thêm face alignment, ImageNet normalization, augmentation, hoặc threshold mask chưa rõ semantics. Pilot đầu dùng transform xác định được, chưa augmentation.

## 4. Recipe đề xuất

Đây là recipe kỹ thuật cần triển khai và kiểm chứng, không phải config chạy được hiện tại hay khẳng định toàn bộ tham số đều được paper quy định.

| Tham số | Tiny / pilot trên máy hiện tại | Run dài sau pilot |
|---|---|---|
| Model | Full A0, khởi tạo từ đầu theo baseline | Giữ cùng A0 |
| Loss | FD + LR + CMC + FL + AR + KL, hệ số đều 1 | Giữ nguyên |
| Precision | FP32 để kiểm tra baseline | AMP chỉ sau đối chiếu số học và ghi thành thay đổi riêng |
| Batch thật | 2, chỉ khi profile đủ bộ nhớ | Mục tiêu 8 theo reference proposal, cần profile lại |
| Accumulation | 1 | Không coi accumulation là thay thế batch thật của CMC |
| Optimizer | Adam, LR 1e-6 cho tiny; pilot đề xuất 1e-5 nếu tiny ổn định | Tham chiếu proposal LR 1e-4 sau đánh giá ổn định |
| Adam khác | betas (0.9,0.999), eps 1e-8; weight decay 1e-3, loại CMC log-temperature theo proposal | Giữ cùng quy ước |
| Scheduler | Không decay trong tiny/pilot tối đa 5 epochs | Tối đa 45 epochs; nhân 0.1 sau mỗi 15 epochs hoàn tất |
| Sampling | Pilot balanced, seed 0, không lặp cùng source trong một batch nếu có thể | Reference proposal dùng uniform shuffle, không class balancing; ghi rõ thay đổi khi chuyển |
| Gradient clipping | Chưa thêm | Nếu cần phải ghi recipe mới và đo tác động |
| Validation | Mỗi epoch; chọn best bằng AUC, tie-break FD CE rồi epoch sớm hơn | Giữ quy tắc |

LR nhỏ hơn là thử nghiệm thận trọng trước gradient lớn, không bảo đảm ổn định hay chữa lỗi G01. Nếu loss/gradient/update không hữu hạn, dừng và lưu chẩn đoán; không tự giảm loss weight, bỏ PAD token hoặc chèn normalization để đạt pass.

Batch 1 làm CMC suy biến về 0 trong baseline, nên chỉ dùng kiểm tra bộ nhớ/giao diện. Batch 2 × accumulation 4 không có cùng tập negatives với batch 8. Nếu batch 2 FP32 không vừa 6 GB, dừng ở profile; cân nhắc activation checkpointing sau kiểm tra equivalence hoặc GPU có nhiều VRAM hơn. Không tự bỏ nhánh language hay đóng băng backbone rồi gọi đó là full MFVLR.

Ước lượng riêng tham số + gradient + hai moment Adam FP32 của 154,156,232 tham số khoảng 2.30 GiB; chưa gồm activations, logits 308×49,408, buffer, CUDA workspace và optimizer temporaries. Vì vậy chưa thể cam kết batch 2 vừa máy, cũng chưa thể dự báo giờ train. Sau warmup, đo median giây/step và peak VRAM; thời gian dự kiến = steps/epoch × giây/step × epochs cộng validation/checkpoint. Không cấp phát toàn dataset dưới dạng tensor trong RAM.

## 5. Checkpoint, đánh giá và đầu ra

- Checkpoint version mới: model, optimizer, scheduler, global step/epoch, RNG Python/NumPy/torch/CUDA, sampler/cursor, config, source revision, manifest/tokenizer hashes, trạng thái hoàn tất atomic. Kiểm tra resume ở môi trường xác định; ghi giới hạn determinism GPU.
- Ghi riêng sáu loss, total, gradient norm, update norm, LR optimizer, peak VRAM, thời gian step và phân bố lớp/generator. Lưu `last`, `best`, history và config; không lưu checkpoint sau mọi step.
- Detection: ROC-AUC, ACC, F1, confusion matrix; chọn threshold bằng validation, khóa trước test. AUC cần cả hai lớp trong partition.
- Localization: IoU/Dice foreground trên fake; báo cáo riêng real false-positive area. Định nghĩa rõ empty-mask và macro/micro averaging; nếu báo mIoU hai lớp thì ghi riêng, không đổi tên foreground IoU thành paper mIoU.
- Inference/evaluation phải dùng `visual_only`, không đưa prompt hoặc nhãn thật vào model. Export checkpoint visual-only với mapping detector và kiểm tra parity với full checkpoint ở chế độ eval.
- Pilot thiếu dữ liệu, batch nhỏ, subset một generator và mask chưa chứng minh trùng protocol paper không phải replication điểm số paper. Cross-generator phải là các run train-source riêng khi muốn đối chiếu protocol tương ứng.

Các file dự kiến bổ sung ở giai đoạn triển khai: `tools/audit_genface.py`, dataset/collator trong `src/mfvlr_research/`, `configs/genface_pilot.yaml`, CLI train/evaluate, checkpoint schema và test integration/resume có giới hạn. Tên này là kế hoạch, chưa tồn tại do tài liệu này.

Ưu tiên tiếp theo: **chốt manifest canonical và semantics mask, sau đó làm loader + profile batch 2**, trước khi cam kết chạy dài. Yêu cầu hiện tại chỉ là lập kế hoạch nên chưa bật training hay sửa các trạng thái phê duyệt lịch sử.

## Tài liệu đối chiếu

- [Trạng thái repository](../REPOSITORY_STATUS.md)
- [Các khoảng trống tích hợp](phase2d2_d02b10_integration_gaps.md)
- [Roadmap hiện có](phase2d2_d02b10_roadmap.md)
- [Phân tích paper trong repo](paper_analysis.md)
- [Reference proposal](../configs/paper_reference.yaml)
- [Registry prompt](../src/mfvlr/prompts.py)
- [Trainer chuẩn bị hiện có](../src/mfvlr_research/d02b9/trainer.py)
