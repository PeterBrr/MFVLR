# Đối chứng nhánh language so khớp câu với từng patch (đăng ký trước 05/11/2026)

Viết trước khi chạy. Mục "Kết quả" sẽ được bổ sung sau. Đây là cải tiến **ngoài paper**: thay bộ giải mã sinh câu của MFVLR bằng kiểu so khớp câu–patch của VLAForge ([arXiv 2603.24454](https://arxiv.org/html/2603.24454), Eq. 9–10) và HiFi-Net++ ([arXiv 2410.23556](https://arxiv.org/html/2410.23556), LFLE).

## Câu hỏi

1. Bản đồ câu–patch có tăng IoU khoanh vùng so với baseline không?
2. Nếu có, phần tăng đến từ **câu CLIP**, hay chỉ từ việc có thêm một loss trên mask ở độ phân giải patch?

## Các nhánh

| Nhánh | Config | Khác baseline |
|---|---|---|
| A. Baseline | `hpc50k_dicefw2` | — |
| B. Câu CLIP | `hpc50k_dicefw2_textpatch_clip` | `text_patch: clip`, `text_patch_weight: 1` |
| C. Đối chứng | `hpc50k_dicefw2_textpatch_random` | `text_patch: random`, `text_patch_weight: 1` |

Mọi thứ khác giống hệt nhau: dữ liệu GenFace 50k, Dice + FW 2, 4 epoch, batch hiệu dụng 4. **Chỉ chạy seed 0** (quyết định 05/11/2026): mỗi lần người dùng chỉ được giữ GPU HPC tối đa 36 giờ, vì phải nhường cho người khác. Không có loss language cũ (LR, CMC, KL đều bằng 0).

## Thiết kế nhánh mới (`src/mfvlr_research/text_patch.py`)

- **Câu:** 3 lớp, mỗi lớp một câu, cố định trước khi chạy:
  1. "an unedited region of a real face photo"
  2. "a region of a face photo where the bangs above the forehead were edited"
  3. "a region of a face photo where the smile around the mouth was edited"
- **Nhánh B:** vector câu lấy từ text encoder CLIP ViT-L/14 của OpenAI, **đóng băng**. Vector được tính một lần bằng `tools/make_text_patch_embeddings.py` (checkpoint SHA256 `b8cca3fd…3836`) và lưu ở `src/mfvlr_research/assets/text_patch_clip_vitl14.json`. Lúc train không chạy CLIP. Cosine giữa 3 câu là 0,79–0,86.
- **Nhánh C:** 3 vector ngẫu nhiên cố định, cùng kích thước 768, chuẩn hóa về độ dài 1 (seed 0). Ba vector gần như vuông góc với nhau.
- **Bản đồ:** 196 token patch (14×14) của ảnh và của residual được cộng lại, qua LayerNorm và một lớp tuyến tính 512→768, rồi chuẩn hóa. Điểm của mỗi lớp là cosine với vector câu nhân hệ số nhiệt độ học được (khởi tạo 1/0,07, tối đa 100, như CLIP).
- **Loss:** cross-entropy mềm theo tỷ lệ pixel bị sửa trong mỗi patch, cộng soft Dice của xác suất "bị sửa" trên ảnh fake. Nhân với trọng số FL và `text_patch_weight` = 1, rồi gộp vào FL và total trong log.
- **Đưa vào mask:** log-odds "bị sửa" (14×14) được phóng to lên 224×224 và cộng vào logit foreground, qua một hệ số khởi tạo bằng 0. Lúc khởi tạo, B và C cho mask **giống hệt** baseline (đã kiểm tra bằng test). Encoder, decoder và các head có trọng số khởi tạo giống baseline ở cùng seed. B và C chỉ khác nhau ở 3 vector câu.
- **Chấm điểm** chỉ dùng ảnh, như mọi lượt trước. Bản đồ là một phần của model nên được tính cả lúc chấm.

## Tiêu chí (viết trước khi chạy)

Chỉ số chính: IoU fake macro trên val 10.062 ảnh, trung bình 3 epoch cuối (epoch 2–4), so ghép cặp ở seed 0. Không dùng test.

Vì chỉ có 1 seed, kết quả là **thăm dò**, không đủ để đổi baseline. Ở các lượt trước, chênh lệch giữa các seed khoảng 0,5–1 điểm IoU.

**B có triển vọng so với A** khi đồng thời:
1. B hơn A ít nhất +2 điểm IoU.
2. FP real (diện tích dự đoán trên ảnh real) không tăng quá 0,5 điểm.
3. AUC không giảm quá 0,5 điểm.
4. Không có giá trị không hữu hạn.

**Phần tăng có thể là nhờ language** chỉ khi B có triển vọng so với A, **và** B hơn C ít nhất +1 điểm IoU.

Một nhánh chỉ được nhận làm baseline mới sau khi chạy thêm seed 1 và 2, và đạt tiêu chí 3 seed của các lượt trước: tốt hơn ở cả 3 seed, trung bình ít nhất +2 điểm.

Cách đọc:
- B có triển vọng, B > C: câu CLIP có thể đóng góp. Chạy thêm seed để xác nhận.
- B và C đều tốt hơn A, B ≈ C: phần tăng đến từ loss trên patch, không phải language.
- B không có triển vọng: dừng hướng này, không chạy thêm seed.

## Giới hạn biết trước

- **Language chỉ đóng góp qua hình học của 3 vector.** Encoder ảnh train từ đầu, không nằm trong không gian CLIP, và lớp chiếu 512→768 là học được. Vì vậy thứ duy nhất B có mà C không có là độ giống nhau giữa 3 câu. Nhiều khả năng B ≈ C. Muốn language thật sự có tác dụng, như trong HiFi-Net++ (thay CLIP bằng ResNet-50 ImageNet thì AUC tụt từ 97,8 xuống 74,0), cần thêm encoder ảnh CLIP đóng băng. Đó là bước sau, chỉ làm nếu B > C.
- **Lối tắt JPEG/PNG:** mọi ảnh real là JPEG, mọi ảnh fake là PNG. Lối tắt này tác động lên cả 3 nhánh như nhau. Nó có thể làm tăng AUC, nhưng không chỉ ra được vùng bị sửa trong ảnh fake.
- **Mask GT rất vụn** (trung vị 373 mảnh mỗi mask). IoU khó vượt khoảng 40–45 với bất kỳ nhánh nào.
- Trọng số `text_patch_weight` = 1 được chọn trước và không được dò.

## Trạng thái

Code và test xong ngày 05/11/2026. Chưa chạy. Chưa có kết quả. Ba lượt chạy nối tiếp trong một job 36 giờ, theo thứ tự A, B, C. Nếu hết giờ, lượt đang chạy sẽ tiếp tục từ checkpoint khi submit lại.
