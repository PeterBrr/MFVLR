# Đối chứng tăng dữ liệu lên 10.000 ảnh train, validation 2.000 (đăng ký trước 03/11/2026)

Viết trước khi chạy. Mục "Kết quả" được bổ sung sau khi chạy xong.

## Câu hỏi và lý do

**Tăng train từ 5.000 lên 10.000 ảnh, trong đó một nửa ảnh real lấy từ FFHQ, có tăng IoU không?**

- Ngày 02/11, Dice + FW 2 (37,30) và ResNet-50 (36,95) cùng dừng quanh IoU 37. ResNet-50 học nhanh hơn hẳn nhưng không vượt được mức trần, nên cần kiểm tra trực tiếp xem lượng dữ liệu có phải là giới hạn hay không.
- Mức +8,20 IoU khi tăng từ 1.000 lên 5.000 ảnh đã đổi cùng lúc số ảnh và số cập nhật. Lượt này giữ số cập nhật gần bằng nhau.
- Validation 400 ảnh quá nhiễu: Dice + FW 2 chỉ vượt ngưỡng 0,10 điểm, trong khi IoU dao động 2–3 điểm giữa các epoch. Validation 2.000 ảnh giảm nhiễu này.

## Phát hiện về dữ liệu khi chuẩn bị

- Mỗi ảnh `Diffae_<thuộc tính>_k` là ảnh `FFHQ/k` đã bị sửa: sai khác trung bình 8–10/255 so với FFHQ cùng số, và 55–117/255 so với CelebA cùng số. Split chính thức chia Diffae theo chỉ số FFHQ: train 0–55.999, val 56.000–62.999, test 63.000–69.999.
- Mọi ảnh real trong manifest chính thức là CelebA-HQ (30.000 tên 0–29.999). Trước đây có 16 nghìn ảnh bị loại vì trùng tên: đó là CelebA 10000–29999, trùng tên với FFHQ 10000–29999 nhưng là ảnh khác.
- Vì vậy đến nay **real luôn là CelebA và fake luôn là FFHQ đã sửa**. Model có thể phân loại thật/giả bằng cách nhận ra bộ ảnh nào, thay vì nhận ra vết sửa. Điều này có thể góp phần làm AUC luôn khoảng 99,9%. Người dùng chọn trộn thêm ảnh real FFHQ để giảm lối tắt này.

## Dữ liệu (`artifacts/genface10k_20261103`, `tools/prepare_genface10k.py`)

| Split | CelebA real | FFHQ real | Diffae | Tổng | Ghi chú |
|---|---:|---:|---:|---:|---|
| Train | 2.499 | 2.501 | 5.000 | 10.000 | Chứa nguyên 5.000 ảnh cũ, đúng thứ tự cũ |
| Val | 500 | 500 | 1.000 | 2.000 | Chứa nguyên 400 ảnh val cũ, đúng thứ tự cũ |
| Test | 1.000 | 0 | 1.000 | 2.000 | Chứa 400 ảnh test cũ. Không đóng gói, không dùng |

- Chỉ lấy ảnh trong split chính thức. Ảnh FFHQ real chỉ lấy trong dải chỉ số của chính split đó, nên không khuôn mặt FFHQ nào xuất hiện ở hai split (script kiểm tra bằng assert). Test chỉ dùng CelebA, vì ảnh FFHQ trên đĩa dừng ở 63001 và mọi ảnh FFHQ khác là ảnh gốc của một ảnh giả train hoặc val.
- Chọn ngẫu nhiên xác định (seed 0). Mỗi ảnh được đọc thử; mask phải nhị phân, mask real phải toàn 0, mask fake phải khác rỗng. Ảnh bị loại nếu trùng nội dung RGB gốc hoặc trùng sau khi resize 224 với bất kỳ ảnh nào đã chọn ở mọi split.
- Tensor tạo bằng đúng `PilotFaces` như các gói cũ. Script kiểm tra 5.000 ảnh train và 400 ảnh val cũ trong gói mới trùng bit với gói cũ.
- Trong cùng một split, một ảnh FFHQ real và ảnh Diffae sửa từ nó có thể cùng có mặt: 223 cặp ở train, 80 cặp ở val.
- Gói 5k cũ vốn đã có một ảnh FFHQ real (`FFHQ/19929.jpg`). Ngày 17/10 ảnh này được chọn vì ảnh CelebA cùng tên không có trên đĩa. Chỉ số 19929 thuộc dải train nên không rò rỉ. Vì vậy train có 2.499 CelebA và 2.501 FFHQ.
- Không ảnh nào bị loại khi kiểm tra. Bundle nặng 1,97 GB, SHA256 `804020c7…f18c`. `train.npz` có hash `d789fb83…a3a8`, `val.npz` có hash `fdd34d9a…4b06`.

## Thiết lập

| Nhánh | Config | Train | Epoch | Cập nhật |
|---|---|---|---:|---:|
| 5k (đối chứng) | `g10k_dicefw2_train5k` | `train5k.npz`, chính là file train cũ (hash `e234984a…6534`) | 15 | 18.750 |
| **10k** | `g10k_dicefw2` | `train.npz` mới | 7 | 17.500 |

- Cả hai dùng Dice + FW 2, cấu hình đã đạt ngày 02/11. Mọi tham số khác giữ nguyên: không language, FD 0,1, AR 1, FL 1, Dice 1, batch hiệu dụng 4, encoder train từ đầu. Cả hai được chấm trên cùng val 2.000 ảnh, seed 0, 1, 2. Không dùng test.
- Nhánh 10k có **ít hơn 6,7% cập nhật** vì 7,5 epoch không phải số nguyên. Chọn 7 thay vì 8 để nếu nhánh 10k thắng thì không thể quy cho số cập nhật nhiều hơn.
- Nhánh 5k có cùng dữ liệu train, cùng khởi tạo và cùng lịch batch với lượt Dice + FW 2 ngày 02/11. Chỉ validation khác.
- Code mới: khóa config tùy chọn `train_file` (mặc định `train.npz`, config cũ không đổi), và report lưu điểm từng ảnh (`per_sample`) để chấm lại theo nguồn ảnh hoặc trên 400 ảnh val cũ.
- Nơi chạy: Kaggle T4, `v5nhdf`, notebook riêng tư `mfvlr-g10k-20261103` cho cả 6 lượt, budget 39.600 giây. Lúc khởi chạy: `nha4von` còn 4,42 giờ, `v5nhdf` còn 15,03 giờ. Dự kiến khoảng 10–11 giờ. Người dùng đã đồng ý tải bundle lên dataset riêng tư `v5nhdf/mfvlr-genface10k-20261103` (đã xác nhận riêng tư, 1 file 1.974.957.831 byte) và chạy notebook.

## Tiêu chí

Chỉ số chính là IoU fake macro trên val 2.000 ảnh, trung bình 3 epoch cuối của mỗi nhánh (5k: epoch 13–15; 10k: epoch 5–7), so ghép cặp theo seed. **Nhánh 10k đạt khi đồng thời:**
1. Tốt hơn nhánh 5k ở cả 3/3 seed.
2. Chênh lệch trung bình ít nhất +2 điểm IoU.
3. FP area trên 1.000 ảnh real (trung bình 3 epoch cuối, trung bình 3 seed) không tăng quá 0,5 điểm. AUC không giảm quá 0,5 điểm.
4. Train không có lỗi số học.

Nếu đạt, dữ liệu 10k thành nền mới, rồi mới tính đến dùng nhiều ảnh hơn.

**Chỉ để mô tả, không phải tiêu chí:**
- FP area và tỷ lệ bị phân loại là fake, tách theo ảnh real CelebA và ảnh real FFHQ. Nhánh 5k chưa từng thấy ảnh real FFHQ. Nếu nó gán nhiều ảnh real FFHQ là fake, giả thuyết lối tắt CelebA/FFHQ được ủng hộ.
- IoU trên đúng 400 ảnh val cũ. Với nhánh 5k, số này phải trùng lượt Dice + FW 2 ngày 02/11 nếu train lặp lại được giữa hai phiên Kaggle.

## Giới hạn đã biết trước

- Nhánh 10k đổi cùng lúc lượng dữ liệu và thành phần ảnh real (thêm FFHQ). Nếu đạt, không tách được phần nào do lượng dữ liệu, phần nào do bỏ lối tắt. Muốn tách cần thêm nhánh 10k chỉ có CelebA.
- Val có 500 ảnh real FFHQ mà nhánh 5k chưa từng thấy, nên tiêu chí 3 (FP, AUC) nghiêng về phía nhánh 10k. Tiêu chí chính là IoU trên ảnh giả, không bị ảnh hưởng trực tiếp.
- Hai nhánh khác số epoch. "3 epoch cuối" ứng với các thời điểm khác nhau trong lịch học.
- Mọi ảnh giả vẫn là Diffae. Không kiểm chứng được việc tách danh tính người giữa CelebA và FFHQ.

## Kết quả (03/11/2026): KHÔNG ĐẠT

Kết quả nằm ở `artifacts/ablation_g10k_kaggle_20261103/downloaded/results`.

**Lượt chạy.** Seed 0 và 1 của cả hai nhánh chạy xong (`COMPLETED`), không có lỗi số học và không dùng test. Seed 2 của cả hai nhánh bị dừng vì hết budget 11 giờ (`PAUSED_TIME_BUDGET`): nhánh 5k dừng ở epoch 2, nhánh 10k dừng ở epoch 1. Thời gian thực tế của mỗi lượt là 4,6 giờ với nhánh 10k và 5,5 giờ với nhánh 5k, dài hơn nhiều so với ước tính 3,1 giờ. Nguyên nhân chính là mỗi epoch phải chấm trên val 2.000 ảnh, và nhánh 5k có 16 lần chấm. Seed 2 không được chạy lại, vì kết quả seed 2 không thể đổi kết luận: nhánh 10k đã thua ở 2 seed, nên tiêu chí 1 (tốt hơn ở 3/3 seed) chắc chắn không đạt.

| Seed | IoU 5k (epoch 13–15) | IoU 10k (epoch 5–7) | Chênh | FP real 5k | FP real 10k | AUC 5k | AUC 10k |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 37,99 | 36,94 | −1,05 | 5,27 | 4,11 | 98,88 | 99,64 |
| 1 | 37,27 | 36,53 | −0,74 | 3,98 | 2,25 | 98,67 | 99,69 |
| **TB** | **37,63** | **36,74** | **−0,89** | 4,63 | 3,18 | 98,78 | 99,67 |

- Tiêu chí 1: thắng 0/2 seed. **Không đạt.**
- Tiêu chí 2: chênh trung bình −0,89 điểm. **Không đạt.**
- Tiêu chí 3: FP real giảm 1,44 điểm, AUC tăng 0,89 điểm. Đạt, nhưng như đã ghi ở phần giới hạn, tiêu chí này nghiêng về phía nhánh 10k.
- Tiêu chí 4: không có lỗi số học. Đạt.

**Mô tả (không phải tiêu chí), tính tại epoch cuối của mỗi nhánh:**

| Seed | Nhánh | FP CelebA | Tỷ lệ CelebA bị gọi là fake | FP FFHQ | Tỷ lệ FFHQ bị gọi là fake | IoU trên 400 ảnh val cũ |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 5k | 1,10 | 1,0% | 6,78 | **24,2%** | 37,46 |
| 0 | 10k | 1,68 | 1,6% | 3,40 | 5,8% | 37,44 |
| 1 | 5k | 0,59 | 1,2% | 3,50 | **18,2%** | 37,12 |
| 1 | 10k | 1,26 | 1,6% | 3,09 | 6,8% | 37,43 |

- **Giả thuyết lối tắt CelebA/FFHQ được ủng hộ.** Nhánh 5k chỉ gọi nhầm 1% ảnh CelebA là fake, nhưng gọi nhầm 18–24% ảnh real FFHQ. Nghĩa là nó phân loại thật/giả phần lớn bằng cách nhận ra bộ ảnh. Thêm ảnh real FFHQ vào train giảm tỷ lệ này xuống 6–7%. Mức 6–7% vẫn cao hơn mức của CelebA, có thể vì ảnh real FFHQ giống hệt phần không bị sửa của ảnh Diffae.
- **Lối tắt này không làm hại IoU trên ảnh giả.** Trên 400 ảnh val cũ, hai nhánh gần như bằng nhau (37,4 so với 37,1–37,5). Vậy bỏ lối tắt giúp phân loại ảnh real, nhưng không giúp khoanh vùng sửa.
- **Đường IoU của nhánh 10k vẫn đang tăng.** Epoch 7 đạt 37,9 và 37,8, trong khi nhánh 5k ở epoch cuối đạt 37,8 và 37,0. Trung bình 3 epoch cuối bất lợi cho nhánh 10k vì nhánh này có ít epoch hơn. Đây chỉ là quan sát, không thay đổi kết luận.
- **Train không lặp lại chính xác giữa hai phiên Kaggle.** IoU của nhánh 5k trên 400 ảnh val cũ khác lượt Dice + FW 2 ngày 02/11 tới 1–2 điểm ở từng epoch, dù dữ liệu, khởi tạo và lịch batch giống nhau. Trung bình 3 epoch cuối thì gần nhau: seed 0 là 37,76 so với 37,33; seed 1 là 37,01 so với 37,42. Vì vậy chênh lệch dưới khoảng 1 điểm giữa các phiên chạy khác nhau cần được đọc thận trọng.

**Kết luận.** Gấp đôi dữ liệu train mà giữ gần nguyên số cập nhật không tăng IoU. Nút thắt quanh 37 không nằm ở lượng dữ liệu, ít nhất là ở quy mô 5k–10k với cùng số bước train. Việc trộn ảnh real FFHQ vẫn đáng giữ, vì nó sửa lỗi gọi nhầm ảnh real FFHQ là fake (18–24% xuống 6–7%) mà không làm giảm IoU trên 400 ảnh val cũ. Dữ liệu 10k không trở thành nền mới theo tiêu chí. Muốn biết nhiều dữ liệu hơn có giúp khi train lâu hơn hay không, cần một lượt riêng với số cập nhật lớn hơn.
