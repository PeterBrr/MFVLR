# Đối chứng Dice + foreground weight 2 trên 5.000 ảnh (đăng ký trước 02/11/2026)

Viết trước khi chạy. Mục "Kết quả" được bổ sung sau khi chạy xong.

## Câu hỏi và lý do

**Giảm foreground weight từ 4 xuống 2 có giữ được mức tăng IoU của Dice mà không làm tăng FP trên ảnh real không?**

Trong [đối chứng 01/11](doi_chung_iou_5000_20261101_vi.md), Dice (FW 4) tốt hơn baseline ở 3/3 seed, trung bình +1,97 điểm IoU, và cao hơn ở 24/24 cặp epoch-seed từ epoch 8–15. Nhánh này vẫn không đạt tiêu chí vì chưa tới +2 điểm và FP real tăng trung bình 0,93 điểm, phần lớn do seed 2, epoch 13 (7,44%). Ở lượt 1 seed trước có language, FW 2 giảm FP real ở epoch 10 từ 4,22% xuống 1,80% mà IoU trung bình epoch 8–10 không giảm. Dice tự phạt việc tô thừa trên ảnh giả, nên có thể bù cho recall mất đi khi giảm FW.

## Thiết lập

- Nơi chạy: Kaggle T4, tài khoản `nha4von`, notebook riêng tư `mfvlr-iou5000-dicefw2-20261102`, hai GPU, mỗi lượt một process. Lúc kiểm tra quota: `nha4von` còn 10,23 giờ, `v5nhdf` còn 20,38 giờ nhưng không đọc được dataset riêng tư của `nha4von`.
- Chỉ chạy nhánh mới, seed 0, 1, 2. **Baseline và Dice FW 4 dùng lại các lượt ngày 01/11**: cùng code (hash 19 file nguồn khớp với report hôm qua), cùng dữ liệu, cùng khởi tạo và lịch batch theo seed. Kiểm tra môi trường: đánh giá epoch 0 (trước khi train, chỉ phụ thuộc khởi tạo) của nhánh mới phải trùng với baseline cùng seed. Nếu không trùng, không so ghép cặp mà phải chạy lại baseline.
- Giữ nguyên dữ liệu, 15 epoch (18.750 cập nhật), không language, FD 0,1, AR 1, FL 1, Dice 1, batch hiệu dụng 4. Không dùng test.
- Dự kiến khoảng 7,5 giờ quota.

| Nhánh | Config | Thay đổi so với baseline |
|---|---|---|
| Baseline (01/11) | `g5k_baseline` | Không |
| Dice FW 4 (01/11) | `g5k_dice` | + 1 × Dice |
| **Dice FW 2** | `g5k_dice_fw2` | + 1 × Dice, foreground weight 4 → 2 |

## Tiêu chí

Giữ nguyên tiêu chí ngày 01/11. Chỉ số chính là IoU fake macro trên validation, trung bình epoch 13–15, so ghép cặp với baseline. **Dice FW 2 đạt khi đồng thời:**
1. Tốt hơn baseline ở cả 3/3 seed.
2. Chênh lệch trung bình ít nhất +2 điểm IoU.
3. FP area trên ảnh real, trung bình epoch 13–15 và trung bình 3 seed, không tăng quá 0,5 điểm. AUC không giảm quá 0,5 điểm.
4. Train không có lỗi số học.

So sánh với Dice FW 4 chỉ dùng để mô tả, không phải tiêu chí. Nếu đạt, bước tiếp theo là kéo baseline và Dice FW 2 lên 30 epoch.

## Giới hạn đã biết trước

- Baseline chạy ở một phiên Kaggle khác. Kiểm tra epoch 0 chỉ xác nhận khởi tạo và đánh giá giống nhau, không bảo đảm mọi phép tính train trùng bit.
- Đổi FW cùng lúc với thêm Dice, nên nếu đạt sẽ không tách được đóng góp riêng của FW 2. Không có nhánh FW 2 không Dice.
- FW 2 trước đây chỉ được thử 1 seed, có language và chưa đủ 15 epoch.
- 3 seed và 400 ảnh validation không đủ để kết luận ý nghĩa thống kê. Mọi ảnh giả đều từ Diffae.

## Kết quả (bổ sung 02/11/2026)

Cả 3 lượt `COMPLETED`, đủ 18.750 cập nhật, không có giá trị không hữu hạn, không dùng test. Kết quả lấy từ `artifacts/ablation_iou5000_kaggle_20261101/dicefw2`. Mỗi lượt mất khoảng 3,1 giờ T4, cả notebook 5,8 giờ. **Kiểm tra epoch 0: trùng baseline cùng seed ở cả 3 seed** (AUC, IoU fake, FP real), nên so ghép cặp với baseline 01/11 là hợp lệ. Script chấm cho lại đúng số của Dice FW 4 đã ghi ngày 01/11 (+1,97 IoU, +0,93 FP).

Trung bình epoch 13–15, điểm phần trăm:

| Seed | IoU baseline | IoU Dice FW 2 | Chênh | FP real baseline | FP real Dice FW 2 | Chênh FP | AUC baseline | AUC Dice FW 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 34,30 | 37,33 | +3,03 | 1,24 | 1,20 | −0,04 | 99,94 | 99,83 |
| 1 | 35,54 | 37,42 | +1,88 | 1,03 | 1,33 | +0,30 | 99,98 | 99,96 |
| 2 | 35,76 | 37,16 | +1,40 | 0,76 | 0,92 | +0,15 | 99,98 | 99,87 |
| **TB** | 35,20 | **37,30** | **+2,10** | 1,01 | 1,15 | **+0,14** | 99,97 | 99,89 (−0,08) |

**Đạt cả 4 tiêu chí:** tốt hơn ở 3/3 seed; +2,10 ≥ +2; FP real +0,14 ≤ 0,5 và AUC −0,08 ≥ −0,5; không lỗi số học. Cao hơn baseline ở 23/24 cặp epoch-seed từ epoch 8–15. FP real cao nhất trong epoch 13–15 là 1,71% (seed 1), so với 7,44% của Dice FW 4.

Mô tả, không phải tiêu chí: IoU gần như bằng Dice FW 4 (37,30 so với 37,18), nhưng FP real thấp hơn (1,15 so với 1,94) và AUC giảm ít hơn (−0,08 so với −0,25). Vậy giảm FW từ 4 xuống 2 chủ yếu kéo FP xuống mà không mất IoU.

Lưu ý khi đọc: mức +2,10 chỉ vượt ngưỡng 0,10 điểm, và 2/3 seed tăng dưới +2. Kết luận là Dice FW 2 đạt theo tiêu chí đã đăng ký, nhưng mức tăng thật có thể nhỏ hơn hoặc lớn hơn ngưỡng. Như đã nêu, không tách được phần đóng góp riêng của FW 2 và của Dice.
