# Đối chứng encoder ResNet-50 pretrained trên 5.000 ảnh (đăng ký trước 02/11/2026)

Viết trước khi chạy. Mục "Kết quả" được bổ sung sau khi chạy xong. Đây là cải tiến **ngoài paper**: MFVLR train từ đầu (§II-B), còn đối chứng này khởi tạo phần CNN bằng trọng số ImageNet. Từ 02/11/2026 mục tiêu dự án là làm model tốt hơn, không còn là tái hiện đúng paper.

## Câu hỏi và lý do

**Thay phần CNN train từ đầu bằng ResNet-50 pretrained trên ImageNet có tăng IoU không?**

- Bằng chứng mạnh nhất đến nay cho thấy model thiếu dữ liệu: tăng từ 1.000 ảnh × 15 epoch lên 5.000 ảnh × 30 epoch cho +8,20 điểm IoU. Mọi thay đổi khác về loss, augmentation hay language đều chỉ cho chênh lệch dưới 2 điểm.
- Khởi tạo pretrained là cách thông dụng nhất để bù thiếu dữ liệu. [DeCLIP (WACV 2025)](https://arxiv.org/html/2409.08849v2) định vị vùng mặt bị sửa cục bộ bằng đặc trưng pretrained đóng băng cộng một decoder tích chập.
- ResNet-50 khớp đúng giao diện hiện có: conv1 cho 64×112, layer1 cho 256×56, layer2 cho 512×28, layer3 cho 1024×14. Vì vậy decoder skip, transformer, residual và các head đều giữ nguyên.

## Thiết lập

| | Baseline (`g5k_baseline`, 01/11) | **ResNet-50 (`g5k_r50`)** |
|---|---|---|
| CNN | U-Net 5 tầng, train từ đầu | ResNet-50 IMAGENET1K_V2 đến layer3. Thêm stem 2 conv 3×3 train từ đầu cho skip 224 (ResNet không có đặc trưng ở độ phân giải đầy đủ) và conv 1×1 nâng 64→128 kênh cho skip 112 |
| Chuẩn hóa đầu vào | Ảnh [0,1] | Ảnh [0,1] cho stem 224. Phần ResNet chuẩn hóa theo mean/std của ImageNet |
| Batch norm | Không có | Đóng băng (FrozenBatchNorm2d), vì microbatch 2 không ước lượng được thống kê batch |
| Learning rate | 1e-4 | 1e-5 cho trọng số pretrained, 1e-4 cho mọi phần còn lại |
| Số tham số | 42,4 triệu | 32,2 triệu |

- **Nơi chạy:** Kaggle T4, tài khoản `v5nhdf`, notebook riêng tư `mfvlr-r50-5000-20261102`, seed 0, 1, 2, mỗi GPU một lượt. Lúc kiểm tra quota: `nha4von` còn 9,09 giờ và đang chạy đối chứng Dice + FW 2, `v5nhdf` còn 20,38 giờ. Người dùng cho phép thêm `v5nhdf` làm người đọc (READER) của hai dataset riêng tư. Sau khi cập nhật, cả hai dataset vẫn ở chế độ riêng tư. Chạy local không khả thi: trên RTX 4050, ResNet-50 mất 2,8 giây mỗi bước, tức khoảng 45 giờ cho 3 seed, và GPU còn dùng chung với dự án khác.
- **Giữ nguyên:** dữ liệu (5.000 train, 400 validation, bundle `1a372662…dc1b`), không language, FW 4, FD 0,1, AR 1, FL 1, không Dice, batch hiệu dụng 4, 15 epoch (18.750 cập nhật), lịch batch theo seed. Transformer, decoder và head có khởi tạo trùng bit với baseline cùng seed (đã có test). Không dùng test.
- **Trọng số:** torchvision `resnet50-11ad3fa6.pth`, SHA256 `11ad3fa6…79ca`, được tải lên dataset riêng tư `nha4von/mfvlr-resnet50-imagenet` với sự đồng ý của người dùng. Notebook kiểm tra hash trước khi dùng.
- **Baseline dùng lại các lượt 01/11.** Code mới chỉ thêm nhánh `encoder`. Mặc định `scratch` cho model trùng bit với trước (đã có test), nhưng hash file `experiment.py` đã đổi. Đây là giới hạn đã biết.
- **Learning rate 1e-5 cho phần pretrained** được chọn trước theo thông lệ fine-tune (thấp hơn 10 lần), không dò.

## Tiêu chí

Giữ nguyên tiêu chí ngày 01/11. Chỉ số chính là IoU fake macro trên validation, trung bình epoch 13–15, so ghép cặp với baseline. **ResNet-50 đạt khi đồng thời:**
1. Tốt hơn baseline ở cả 3/3 seed.
2. Chênh lệch trung bình ít nhất +2 điểm IoU.
3. FP area trên ảnh real, trung bình epoch 13–15 và trung bình 3 seed, không tăng quá 0,5 điểm. AUC không giảm quá 0,5 điểm.
4. Train không có lỗi số học.

Nếu đạt, ResNet-50 thành baseline mới, và các thay đổi về loss (Dice, Lovász) sẽ được thử lại trên nền này.

## Giới hạn đã biết trước

- Đổi cùng lúc trọng số pretrained, kiến trúc CNN, batch norm đóng băng và learning rate riêng. Nếu đạt, không tách được phần đóng góp của riêng việc pretrain. Muốn tách cần thêm nhánh ResNet-50 train từ đầu.
- Encoder dùng chung cho ảnh và ảnh residual |tái tạo − ảnh|. ImageNet không có loại ảnh như residual, nên đường residual có thể được lợi ít hơn.
- ImageNet không phải dữ liệu mặt người, và chỉ có một backbone, một mức learning rate được thử.
- 3 seed và 400 ảnh validation không đủ để kết luận ý nghĩa thống kê. Mọi ảnh giả đều từ Diffae.

## Kết quả (bổ sung 02/11/2026)

Cả 3 lượt `COMPLETED`, đủ 18.750 cập nhật, 32.158.984 tham số, không có giá trị không hữu hạn, không dùng test. Kết quả lấy từ `artifacts/ablation_r50_kaggle_20261102`. Mỗi lượt mất khoảng 2,9 giờ T4, cả notebook 5,3 giờ. Epoch 0 không trùng baseline, đúng như dự kiến vì encoder khác.

Trung bình epoch 13–15, điểm phần trăm:

| Seed | IoU baseline | IoU ResNet-50 | Chênh | FP real baseline | FP real ResNet-50 | AUC baseline | AUC ResNet-50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 34,30 | 36,95 | +2,65 | 1,24 | 0,03 | 99,94 | 100,00 |
| 1 | 35,54 | 36,64 | +1,10 | 1,03 | 0,17 | 99,98 | 99,83 |
| 2 | 35,76 | 37,27 | +1,51 | 0,76 | 0,08 | 99,98 | 100,00 |
| **TB** | 35,20 | **36,95** | **+1,75** | 1,01 | **0,09** | 99,97 | 99,94 (−0,02) |

**Không đạt, vì trượt tiêu chí 2** (+1,75 < +2). Các tiêu chí còn lại đều đạt: tốt hơn ở 3/3 seed và ở 24/24 cặp epoch-seed từ epoch 8–15; FP real giảm 0,91 điểm; AUC −0,02; không lỗi số học. Theo quy tắc đã đăng ký, ResNet-50 không thành baseline mới.

Mô tả, không phải tiêu chí:
- **Học nhanh hơn nhiều:** IoU epoch 1 là 31,2–32,0 ở cả 3 seed, so với 9,6–18,9 của baseline. ResNet-50 đạt mức cuối của baseline (khoảng 35) từ epoch 4.
- **Nhưng mức trần gần như không đổi:** từ epoch 5 trở đi IoU dao động quanh 35–38, ngang Dice FW 2 (37,30). Giả thuyết "thiếu dữ liệu nên thiếu đặc trưng tốt" chỉ giải thích được tốc độ học, không giải thích được mức trần. Nút thắt nhiều khả năng nằm ở chỗ khác: decoder, GT vụn, hoặc loss.
- **FP real gần như bằng 0** (0,03–0,17%), thấp hơn baseline khoảng 10 lần.
- Dao động giữa các epoch lớn (ví dụ seed 0: 38,0 → 35,1 → 37,9), nên trung bình 3 epoch vẫn còn nhiễu đáng kể.
