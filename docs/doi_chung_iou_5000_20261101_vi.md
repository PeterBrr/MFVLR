# Đối chứng tăng IoU trên 5.000 ảnh (đăng ký trước 01/11/2026)

Viết trước khi chạy. Mục "Kết quả" được bổ sung sau khi chạy xong.

## Câu hỏi và lý do

**Trên 5.000 ảnh, lật ngang hoặc thêm Dice loss có tăng IoU của nhánh ảnh không?**

Đo trên 3 checkpoint baseline 1.000 ảnh (chỉ train và validation):
- Gần như không overfit (IoU train 27,8–31,9; validation 27,4–29,5), nên đối chứng trên 1.000 ảnh dễ đánh giá thấp các cách chống overfit.
- Chỉnh ngưỡng mask không giúp được: chọn ngưỡng trên một nửa validation rồi chấm trên nửa còn lại không tăng IoU, và ngưỡng tốt nhất cho từng ảnh (không dùng được thật) cũng chỉ đạt 29,4–31,7.
- Phần lớn lỗi nằm sát GT: precision macro 32–40% tăng lên 69–79% nếu cho lệch 2 pixel. IoU gộp ô 8×8 đạt 55–63.
- GT vụn ngay từ file gốc 256×256: trung vị 460 mảnh mỗi ảnh, tự lệch 1 pixel chỉ còn IoU 53,3. Resize nearest không tạo ra độ vụn này (bản 224 có 383 mảnh, khớp 100% với nearest-resize từ file gốc).

Chỉ riêng lượt 5.000 ảnh × 30 epoch seed 0 từng cho +8,20 điểm so với 1.000 ảnh × 15 epoch (dữ liệu và số cập nhật cùng tăng), với khoảng cách train/validation 12,31 điểm. Vì vậy đối chứng này chạy ở 5.000 ảnh, nơi chống overfit có cơ hội phát huy. Dice được thử vì trực tiếp tối ưu độ chồng lấp của từng ảnh, thay vì từng pixel độc lập.

## Thiết lập

- Nơi chạy: Kaggle T4, tài khoản `nha4von`, hai notebook riêng tư, mỗi GPU một lượt, mỗi lượt một process. Lúc kiểm tra quota: `nha4von` còn 25,92 giờ, `v5nhdf` còn 20,38 giờ nhưng không đọc được dataset riêng tư của `nha4von`.
- `mfvlr-iou5000-s01-20261101`: seed 0 và 1. `mfvlr-iou5000-s2-20261101`: seed 2. Mọi nhánh của một seed nằm trong cùng notebook.
- Dữ liệu: dataset riêng tư `nha4von/mfvlr-genface5000-epoch30-20261027`, bundle SHA256 `1a372662…dc1b`, 5.000 train (2.500 thật, 2.500 Diffae), 400 validation (cùng hash với các đối chứng trước). Không dùng test.
- Giữ nguyên: không language, FW 4, FD 0,1, AR 1, FL 1, batch hiệu dụng 4, microbatch 2, 15 epoch (18.750 cập nhật), cùng khởi tạo và cùng lịch batch theo seed.
- Dự kiến dùng khoảng 16 giờ quota và khoảng 10 giờ thực tế.

| Nhánh | Config | Thay đổi so với baseline |
|---|---|---|
| Baseline | `g5k_baseline` | Không |
| Flip | `g5k_flip` | Lật ngang p 0,5 |
| Dice | `g5k_dice` | Loss mask = CE có trọng số + 1 × Dice |

Dice là soft Dice trên xác suất vùng giả, tính cho từng ảnh rồi lấy trung bình, smoothing 1, **chỉ trên ảnh giả** vì ảnh thật không có vùng giả và đã do CE phạt. Microbatch không có ảnh giả thì phần Dice bằng 0. Trọng số 1 được chọn trước, không dò.

## Tiêu chí

Chỉ số chính: IoU fake macro trên validation, trung bình 3 epoch cuối của mỗi seed, rồi lấy chênh lệch ghép cặp so với baseline.

**Mỗi nhánh được coi là cải thiện khi đồng thời:**
1. Tốt hơn baseline ở cả 3/3 seed.
2. Chênh lệch trung bình ít nhất +2 điểm IoU.
3. FP area trên ảnh real không tăng quá 0,5 điểm và AUC không giảm quá 0,5 điểm.
4. Train không có lỗi số học.

Hai nhánh được đánh giá độc lập; nếu cả hai đạt, nhánh có IoU trung bình cao hơn sẽ là baseline mới và tổ hợp Dice + flip được thử ở lượt sau. Precision, recall và Dice vẫn được báo cáo.

## Giới hạn đã biết trước

- Không có nhánh Dice + flip, nên không đo được tương tác giữa hai phép.
- So với 1.000 ảnh × 15 epoch, dữ liệu và số cập nhật vẫn cùng tăng 5 lần. Lượt 5.000 × 30 epoch cũ chỉ để tham khảo vì khác code, số epoch và chỉ có seed 0.
- Tiêu chí ở 3 epoch cuối của 15 epoch; IoU lượt cũ còn tăng sau epoch 15 (33,57 ở epoch 15, 36,41 ở epoch 30), nên nhánh cần train lâu hơn có thể bị bất lợi.
- 3 seed và 400 ảnh validation không đủ để kết luận về ý nghĩa thống kê. Mọi ảnh giả đều từ Diffae, nên không kiểm tra tổng quát sang generator khác.
- GT vụn khiến IoU pixel rất nhạy với sai lệch nhỏ. Mức IoU tối đa đạt được thực tế chưa biết; các con số về độ vụn không phải một cận trên chính thức.

## Kết quả (bổ sung 02/11/2026)

Cả 9 lượt `COMPLETED`, đủ 18.750 cập nhật, không có giá trị không hữu hạn, không dùng test. Kết quả lấy từ `artifacts/ablation_iou5000_kaggle_20261101/s01` và `/s2`. Mỗi lượt baseline và flip mất khoảng 3,1 giờ T4, Dice khoảng 3,6 giờ. Số dưới đây là trung bình epoch 13–15, tính bằng điểm phần trăm.

| Seed | IoU baseline | Flip | Dice | FP real baseline | FP real flip | FP real Dice | AUC Dice so với baseline |
|---|---|---|---|---|---|---|---|
| 0 | 34,30 | 35,74 (+1,44) | 37,48 (+3,17) | 1,24 | 1,53 (+0,29) | 1,47 (+0,22) | −0,31 |
| 1 | 35,54 | 34,56 (−0,98) | 37,04 (+1,50) | 1,03 | 1,53 (+0,50) | 1,21 (+0,18) | −0,35 |
| 2 | 35,76 | 35,19 (−0,58) | 37,01 (+1,25) | 0,76 | 0,85 (+0,09) | 3,14 (+2,38) | −0,10 |
| **TB** | | **−0,04** | **+1,97** | | **+0,29** | **+0,93** | **−0,25** |

**Flip: không đạt** ở tiêu chí 1 và 2, vì chỉ tốt hơn ở 1/3 seed và trung bình −0,04 điểm.

**Dice: không đạt.** Nhánh này tốt hơn ở 3/3 seed, AUC giảm không quá 0,5 điểm và không có lỗi số học. Tuy vậy, mức tăng trung bình +1,97 thấp hơn ngưỡng +2, và FP real tăng trung bình 0,93 điểm, vượt ngưỡng 0,5. Mức tăng FP chủ yếu do seed 2, epoch 13: FP vọt lên 7,44% rồi trở lại 1,06% ở epoch 14 và 0,93% ở epoch 15. Ngưỡng không được nới sau khi xem kết quả.

Quan sát ngoài tiêu chí: ở epoch 8–15, Dice có IoU cao hơn baseline ở cả 24/24 cặp epoch-seed, và Dice macro tăng 1,3–3,5 điểm. Đây là tín hiệu nhất quán nhất trong các đối chứng tăng IoU đến nay, nhưng không thay được kết luận theo tiêu chí. Nếu muốn khẳng định Dice, cần một đối chứng mới, đăng ký trước, có cách kiểm soát FP real.
