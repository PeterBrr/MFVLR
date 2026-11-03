# Đối chứng residual vào nhánh mask và VIM single/double (đăng ký trước 30/10/2026)

Viết trước khi chạy, để tiêu chí không bị chỉnh theo kết quả. Mục "Kết quả" được bổ sung sau khi chạy xong.

## Câu hỏi và lý do

**R — Đưa residual vào nhánh mask có tăng IoU không?** Trên checkpoint foreground weight 2 (bước 14.600), residual trong vùng GT lớn hơn ngoài vùng GT rõ rệt trên ảnh giả (+0,0093) so với ảnh thật ghép cặp dùng cùng mask (+0,00044), nhưng decoder hiện không nhận residual.

**V — VIM residual `single` có tốt hơn `double` không?** Với `double`, bỏ phần cập nhật của block cuối encoder gần như không đổi đầu ra (0,05%), so với 78% ở block 0.

## Thiết lập chung

- 1.000 train / 400 validation (`artifacts/genface_fd_validation_kaggle_20261026/package_001`), GenFace real + Diffae, không augmentation. **Không dùng test.**
- Trainer `src/mfvlr_research/experiment.py`; đã đối chiếu từng bit với `tools/run_language5000.py` trên 1 cập nhật (617 tensor giống hệt).
- Seed 0, 1, 2, ghép cặp: cùng khởi tạo và cùng lịch batch trong mỗi seed.
- Adam, LR ảnh 1e-4, LR language 1e-5, không weight decay, không scheduler. Effective batch 4, microbatch 2. Foreground weight 4. Detection threshold 0,5, mask argmax.
- Giới hạn bộ nhớ GPU `gpu_memory_fraction` 0,88 ở mọi lượt: không giới hạn thì bộ nhớ reserved lên tới 8,9 GiB trên GPU 6 GiB và mỗi cập nhật ảnh chậm hơn 5,5 lần (3,64 so với 0,66 giây). Mỗi chế độ tự lặp lại chính xác từng bit (đã kiểm tra lặp 30 cập nhật), nhưng hai chế độ khác nhau ở mức làm tròn tích lũy: loss cập nhật đầu trùng 8 chữ số, sau 30 cập nhật 0,8459 so với 0,8473. Vì vậy đối chiếu từng bit với runner cũ chỉ áp dụng khi không giới hạn.

| Đối chứng | Baseline | Biến thể | Loss (fd/ar/fl/lr/cmc/kl) | Epoch |
|---|---|---|---|---:|
| R | `residual_mask_none` | `residual_mask_concat` | 0,1 / 1 / 1 / 0 / 0 / 0 | 15 |
| V | `vim_double` | `vim_single` | 0,1 / 1 / 1 / 1e-4 / 0,01 / 0,01 | 10 |

Số epoch đã tăng so với bản nháp đầu (10 và 5) sau khi phát hiện giới hạn bộ nhớ, trước khi có bất kỳ kết quả nào. Ở 15 epoch, baseline R có thể đối chiếu sơ bộ với lượt FD 0,1 trên Kaggle (IoU trung bình 26,20%), nhưng khác môi trường và đường tính nên không so sánh chặt.

`residual_mask_concat`: logits mask cộng thêm một nhánh hiệu chỉnh nhận đặc trưng trunk và residual đã detach, chuẩn hóa theo sai số trung bình của từng ảnh. Lớp cuối khởi tạo bằng 0 nên lúc đầu output trùng baseline. Đây là lựa chọn kỹ thuật thử nghiệm, không phải kiến trúc trong paper.

Đối chứng V đều dùng pre-norm và EOS pooling; language coupled.

## Chỉ số và tiêu chí

**Chỉ số chính của R:** IoU fake macro trên validation, lấy trung bình 3 epoch cuối của mỗi seed, rồi lấy chênh lệch ghép cặp.

**R được coi là đáng xác nhận trên 5.000 ảnh** khi đồng thời:
1. Tốt hơn baseline ở cả 3/3 seed.
2. Chênh lệch trung bình ít nhất +2 điểm IoU.
3. FP area real không tăng quá 0,5 điểm và AUC không giảm quá 0,5 điểm.

**Chỉ số chính của V:** ảnh hưởng của các block encoder và mức phụ thuộc vào prompt, đo trên 40 ảnh validation cuối lượt train.

**V được coi là chấp nhận được** khi đồng thời:
1. Ảnh hưởng của block 11 cao hơn `double` ít nhất 10 lần ở cả 3 seed.
2. IoU (trung bình 3 epoch cuối) không giảm quá 1 điểm so với `double`.
3. Train hữu hạn, không lỗi số học.

Cosine giữa đặc trưng text khác prompt được báo cáo để xem `single` có làm text phụ thuộc prompt hơn không, nhưng không dùng để quyết định, vì đây là lượt ngắn và VIM vẫn chỉ có một token visual.

## Giới hạn đã biết trước

- 3 seed và 400 ảnh validation không đủ để kết luận có ý nghĩa thống kê.
- Validation đã được dùng nhiều lần trong dự án.
- Quy mô nhỏ hơn lượt 5.000 ảnh; với V, IoU chỉ đóng vai trò kiểm tra không làm hại.
- Chưa xác minh subject-disjoint.
- Môi trường: RTX 4050 6 GB, torch 2.7.1; không so sánh trực tiếp với số trên Kaggle.

## Kết quả (bổ sung 31/10/2026, sau khi chạy xong)

Cả 12 lượt hoàn thành, loss hữu hạn ở mọi cập nhật, không dùng test. Số liệu lấy từ `artifacts/ablation_residual_vim_20261030`. Mọi chỉ số dưới đây là trung bình 3 epoch cuối của từng seed, rồi trung bình 3 seed; chênh lệch tính theo điểm phần trăm.

### R — residual vào nhánh mask: KHÔNG đạt tiêu chí

| Chỉ số | `none` | `concat` | Chênh lệch | Theo seed 0 / 1 / 2 |
|---|---:|---:|---:|---|
| IoU fake | 27,66 | 28,94 | +1,28 | +1,56 / +1,58 / +0,71 |
| Dice fake | 42,76 | 44,39 | +1,63 | +2,00 / +2,05 / +0,84 |
| AUC | 98,86 | 99,57 | +0,71 | +0,13 / +1,64 / +0,35 |
| FP area real | 3,05 | 2,76 | −0,30 | +0,48 / −2,25 / +0,88 |
| Precision pixel | 33,29 | 33,65 | +0,37 | −1,15 / +3,06 / −0,81 |
| Recall pixel | 53,94 | 58,01 | +4,07 | +6,94 / +0,94 / +4,33 |

| Tiêu chí | Kết quả |
|---|---|
| Tốt hơn ở 3/3 seed | Đạt |
| IoU tăng trung bình ít nhất 2 điểm | **Không đạt (+1,28; độ lệch chuẩn giữa các seed 0,50)** |
| FP không tăng quá 0,5 điểm, AUC không giảm quá 0,5 điểm | Đạt theo trung bình 3 seed; riêng seed 2 FP tăng 0,88 điểm |

Theo tiêu chí đã đăng ký, chưa chuyển sang xác nhận trên 5.000 ảnh. Hiệu ứng cùng chiều ở cả 3 seed nhưng nhỏ hơn ngưỡng, và nhỏ hơn dao động IoU giữa các epoch (2,5–7,1 điểm trong 5 epoch cuối).

### V — VIM single so với double: ĐẠT tiêu chí

| Chỉ số | `double` | `single` | Chênh lệch | Theo seed 0 / 1 / 2 |
|---|---:|---:|---:|---|
| IoU fake | 24,63 | 25,75 | +1,13 | +2,33 / +0,57 / +0,47 |
| AUC | 98,96 | 98,78 | −0,18 | −0,26 / +0,41 / −0,70 |
| FP area real | 4,90 | 5,46 | +0,56 | +2,30 / −0,33 / −0,29 |
| Recall pixel | 49,28 | 52,36 | +3,08 | +9,96 / +2,64 / −3,36 |

| Đo cuối lượt | `double` (seed 0 / 1 / 2) | `single` (seed 0 / 1 / 2) |
|---|---|---|
| Ảnh hưởng block 11 | 0,052% / 0,062% / 0,049% | 39,7% / 33,2% / 48,1% |
| Ảnh hưởng block 0 | 95,4% / 99,9% / 92,5% | 45,9% / 40,7% / 53,7% |
| RMS đầu ra encoder | 8.469 / 7.745 / 10.476 | 11,0 / 12,4 / 10,0 |
| Cosine text: khác prompt, cùng ảnh | 0,915 / 0,935 / 0,986 | 0,996 / 0,992 / 0,995 |
| Cosine text: cùng prompt, khác ảnh | 0,225 / 0,120 / 0,171 | 0,174 / 0,260 / 0,408 |

| Tiêu chí | Kết quả |
|---|---|
| Ảnh hưởng block 11 cao hơn ít nhất 10 lần ở cả 3 seed | Đạt (771 / 536 / 983 lần) |
| IoU không giảm quá 1 điểm | Đạt (tăng 1,13 điểm) |
| Train hữu hạn, không lỗi số học | Đạt |

`single` sửa được vấn đề các block sau mất tác dụng, nhưng **không** làm text phụ thuộc prompt hơn: cosine giữa hai prompt còn tăng từ khoảng 0,945 lên 0,995, tức text gần như chỉ phản ánh đặc trưng ảnh. Ở epoch 10 IoU vẫn đang tăng và còn dao động mạnh, nên số IoU của V chỉ là kiểm tra không làm hại.

### Sai lệch so với đăng ký

- Ảnh hưởng của block được đo trên 1 ảnh validation (cosine mới dùng 40 ảnh), khác với mô tả chung "40 ảnh" ở trên. Chênh lệch hàng trăm lần nên kết luận không phụ thuộc vào điểm này.
- Tiêu chí FP của R không nói rõ tính theo trung bình hay từng seed; ở đây tính theo trung bình và báo kèm từng seed.
- Hai lượt gặp lỗi hết VRAM và được tự chạy lại từ checkpoint: `residual_mask_none` seed 0 (từ bước 1.000) và `residual_mask_concat` seed 2 (từ bước 1.750). `history.jsonl` vẫn liền mạch; ảnh hưởng ở mức làm tròn của việc resume trên GPU chưa được đo.
