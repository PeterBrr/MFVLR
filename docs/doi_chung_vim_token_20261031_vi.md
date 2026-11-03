# Đối chứng VIM nhiều token (đăng ký trước 31/10/2026)

Viết trước khi chạy. Mục "Kết quả" được bổ sung sau khi chạy xong.

## Câu hỏi và lý do

VIM một key làm mọi từ nhận cùng một vector ảnh, và đặc trưng text gần như không phụ thuộc prompt (cosine giữa hai prompt khoảng 0,995 ở `vim_single`). **Khi K/V là cả 197 token ảnh, attention có thật sự khác nhau giữa các từ và làm text phụ thuộc prompt hơn không, mà không làm hại IoU?**

Đây là biến thể cải tiến của dự án, không phải tái hiện paper. Bảng XV của paper có so sánh K/V dùng mọi token, nhưng không mô tả cách ghép token ảnh và residual.

## Thiết lập

- Baseline: `vim_single` đã chạy trong `artifacts/ablation_residual_vim_20261030/vim`, không chạy lại. Code sau khi thêm biến thể vẫn cho kết quả giống hệt từng bit sau 30 cập nhật (617 tensor, seed 1).
- Biến thể: `configs/experiments/vim_tokens_all.json`. Mọi thiết lập khác giữ nguyên: 1.000 train / 400 validation, 10 epoch, seed 0, 1, 2, cùng khởi tạo và cùng lịch batch, không dùng test.
- K/V là tổng từng token của nhánh appearance và residual (197 token). Scale chia `sqrt(d/r)`, theo `CrossAttention` trong code công khai của cùng tác giả; paper in `sqrt(r/d)`. CMC, KL, LR giữ nguyên.

## Tiêu chí

**Coi là có tác dụng khi đồng thời, ở cả 3 seed:**
1. Delta của VIM khác nhau giữa các từ: `vim_delta_position_spread` ít nhất 0,01 (baseline bằng 0).
2. Text phụ thuộc prompt hơn: cosine giữa hai prompt, cùng ảnh (`text_cosine_other_prompt_same_visual`), thấp hơn baseline ít nhất 0,05.
3. IoU (trung bình 3 epoch cuối) không giảm quá 1 điểm so với baseline, tính trung bình 3 seed.
4. AUC không giảm quá 0,5 điểm, tính trung bình 3 seed, và train không có lỗi số học.

Không đặt ngưỡng tăng IoU, vì câu hỏi chính là cơ chế attention. Mức thay đổi IoU, FP area và `vim_attention_entropy_ratio` vẫn được báo cáo.

## Giới hạn đã biết trước

- Sau 30 cập nhật, attention còn gần đều (entropy tương đối 0,986), nên kết quả phụ thuộc vào việc nó có sắc nét hơn sau khi train không.
- Text vẫn nhận đặc trưng ảnh, nên CMC vẫn có thể tự thỏa mãn; thử nghiệm này không sửa điểm đó.
- Pilot chỉ có 2 prompt, khác nhau ở từ real/fake và mô tả Diffae.
- 3 seed và 400 ảnh validation không đủ để kết luận về ý nghĩa thống kê.

## Kết quả

Chạy local từ 31/10/2026, kết quả trong `artifacts/ablation_vim_tokens_20261031`. Cả 3 seed chạy đủ 2.500 cập nhật, không phải chạy lại, không có lỗi số học, cùng dữ liệu và cùng lịch batch với baseline.

**Không đạt tiêu chí, vì tiêu chí 2 và 3 không đạt.**

| Tiêu chí | Ngưỡng | Seed 0 | Seed 1 | Seed 2 | Kết quả |
|---|---|---|---|---|---|
| 1. `vim_delta_position_spread` | ≥ 0,01 | 0,094 | 0,097 | 0,063 | Đạt |
| 2. Mức giảm cosine giữa hai prompt | ≥ 0,05 | 0,055 | 0,035 | 0,030 | Không đạt |
| 3. Thay đổi IoU (trung bình −2,76) | ≥ −1 điểm | −7,56 | −1,36 | +0,63 | Không đạt |
| 4. Thay đổi AUC (trung bình +0,16) | ≥ −0,5 điểm | −0,18 | −0,12 | +0,76 | Đạt |

Trung bình 3 epoch cuối của 3 seed:

| Chỉ số (%) | `vim_single` | Nhiều token | Chênh lệch |
|---|---|---|---|
| IoU | 25,75 | 22,99 | −2,76 |
| Dice | 40,28 | 36,39 | −3,89 |
| AUC | 98,78 | 98,94 | +0,16 |
| Accuracy | 93,36 | 91,33 | −2,03 |
| FP area trên ảnh real | 5,46 | 4,78 | −0,69 |
| Precision | 30,05 | 30,83 | +0,78 |
| Recall | 52,36 | 43,27 | −9,10 |

Nhận xét:
- Attention vẫn gần đều sau khi train: `vim_attention_entropy_ratio` là 0,956, 0,967 và 0,970.
- Cosine giữa hai prompt giảm từ khoảng 0,995 xuống khoảng 0,955, nhưng ảnh không gần prompt đúng hơn prompt sai (`visual_to_text_matched` so với `visual_to_text_other_prompt`: 0,380/0,389, 0,357/0,354, 0,418/0,426).
- Mức giảm IoU chủ yếu đến từ seed 0, nơi epoch 8 và 9 thấp (11,3 và 13,6). Riêng epoch 10, IoU trung bình là 28,0 so với 27,7 của baseline. Đây là số xem thêm, không thuộc tiêu chí đã đăng ký.

Quyết định: giữ VIM một key với residual `single`, không dùng biến thể nhiều token cho các lần chạy tiếp theo.
