# Tham khảo tiền xử lý DeepfakeBench cho MFVLR

Đọc ngày29/10/2026, commit `f188b1c105465e2e5377eb536a95022ae0e4522d` của SCLBD/DeepfakeBench. Chỉ đọc mã nguồn, chưa chạy pipeline ngoài hoặc thay source/gói Kaggle đang train. Bản file tham chiếu lưu tại `artifacts/reference_deepfakebench_f188b1c/`.

## Quy trình thực tế và khác biệt với MFVLR

DeepfakeBench có hai giai đoạn:

1. Offline: từ video lấy frame, chọn khuôn mặt lớn nhất bằng dlib, dùng5landmark để ước lượng similarity transform, align/crop có margin, mặc định ra256×256. Nếu có mask thì dùng cùng ma trận biến đổi. Lưu ảnh, landmarks và mask. Đây là pipeline cho dữ liệu video được hỗ trợ, không có nghĩa mọi ảnh GenFace cần crop lại.
2. Dataloader: đọc ảnh RGB, resize cubic về resolution trong config; chỉ augmentation khi mode=train và bật cờ; ToTensor rồi Normalize theo mean/std của model.

Nguồn offline: [preprocess.py L97–189](https://github.com/SCLBD/DeepfakeBench/blob/f188b1c105465e2e5377eb536a95022ae0e4522d/preprocessing/preprocess.py#L97-L189), [lấy frame/lưu mask L253–314](https://github.com/SCLBD/DeepfakeBench/blob/f188b1c105465e2e5377eb536a95022ae0e4522d/preprocessing/preprocess.py#L253-L314).

| Thành phần | DeepfakeBench đã đọc | MFVLR hiện tại |
|---|---|---|
| Face alignment | Offline dlib/5điểm, lấy mặt lớn nhất | Không thêm alignment/crop mới |
| Kích thước | Theo detector; Xception/UCF/RECCE config256 | 224×224 |
| RGB resize | OpenCV cubic | PIL bilinear |
| Tensor ảnh | ToTensor rồi mean/std config | float32 chia255, RGB[0,1] |
| Augmentation | Tùy detector và train-only | Chưa bật trong các lượt đối chứng hiện tại |
| Mask | Có đường biến đổi đồng bộ; mask có thể là float | GT nhị phân int64, nearest resize |

Nguồn loader: [load_rgb L283–315](https://github.com/SCLBD/DeepfakeBench/blob/f188b1c105465e2e5377eb536a95022ae0e4522d/training/dataset/abstract_dataset.py#L283-L315), [train-only/normalize L507–520](https://github.com/SCLBD/DeepfakeBench/blob/f188b1c105465e2e5377eb536a95022ae0e4522d/training/dataset/abstract_dataset.py#L507-L520).

## Augmentation: đọc cả config và code gọi

Pipeline chung trong `init_data_aug_method()` gồm flip, rotate, GaussianBlur, OneOf3kiểu IsotropicResize, OneOf brightness/contrast hoặc FancyPCA hoặc HueSaturationValue, rồi image compression. Không có RandomCrop trong danh sách này; dù file helper có class crop, sự tồn tại của helper không chứng minh loader này sử dụng nó.

| Tham số | Xception | UCF | RECCE |
|---|---:|---:|---:|
| use_data_augmentation | true | true | false |
| Horizontal flip p | 0,5 | 0,5 | Không chạy khi flagfalse |
| Rotate p | 0 | 0,5 | Không chạy khi flagfalse |
| Rotate limit | ±10độ (đang tắt) | ±10độ | — |
| Blur p / kernel limit | 0,5 / [3,7] | 0,5 / [3,7] | — |
| Brightness/contrast limit | ±0,1 | ±0,1 | — |
| JPEG quality range | 40–100 | 40–100 | — |
| mean/std mỗi kênh | 0,5 / 0,5 | 0,5 / 0,5 | 0,5 / 0,5 |

Color OneOf có p0,5 hardcode trong loader; JPEG cũng p0,5 hardcode. `brightness_prob` có trong YAML nhưng không được loader này đọc làm xác suất riêng cho RandomBrightnessContrast. IsotropicResize không phải random downsample theo scale: max_side đặt đúngresolution. Vì load_rgb đã đưa ảnh thành vuông đúngresolution, helper có thể trả ảnh nguyên trạng trong đường nạp chung này.

Nguồn:
- [Loader augmentation L113–131](https://github.com/SCLBD/DeepfakeBench/blob/f188b1c105465e2e5377eb536a95022ae0e4522d/training/dataset/abstract_dataset.py#L113-L131).
- [Xception config L26–47](https://github.com/SCLBD/DeepfakeBench/blob/f188b1c105465e2e5377eb536a95022ae0e4522d/training/config/detector/xception.yaml#L26-L47).
- [UCF config L73–88](https://github.com/SCLBD/DeepfakeBench/blob/f188b1c105465e2e5377eb536a95022ae0e4522d/training/config/detector/ucf.yaml#L73-L88).
- [RECCE config L26–49](https://github.com/SCLBD/DeepfakeBench/blob/f188b1c105465e2e5377eb536a95022ae0e4522d/training/config/detector/recce.yaml#L26-L49).
- [IsotropicResize helper L10–46](https://github.com/SCLBD/DeepfakeBench/blob/f188b1c105465e2e5377eb536a95022ae0e4522d/training/dataset/albu.py#L10-L46).

RECCE tắt augmentation là quan sát config tại commit này, không phải lời giải thích của tác giả rằng augmentation gây hại cho reconstruction. Không suy ra MFVLR nên luôn tắt hay luôn bật từ riêng config đó.

## Mask: phần nên học và phần cần điều chỉnh

Loader truyền image và mask có foreground vào cùng Albumentations Compose, nên các phép hình học dùng chung tham số. Mask rỗng không cần biến đổi hình học vì vẫn rỗng. Helper IsotropicResize có apply_to_mask dùng nearest. Đây là mẫu tổ chức phù hợp cho augmentation ảnh–GT của MFVLR.

Tuy nhiên, load_mask resize bằng cv2 mặc định rồi chia255 ra float, không áp dụng nearest trong bước này. Offline warpAffine/resize mask cũng không chỉ định nearest rồi sau đó threshold ở1/255 để nhị phân hóa. Không sao chép các lựa chọn nội suy này vào GT class-index của MFVLR: nội suy có thể tạo nhãn trung gian/đổi biên. Tiếp tục dùng nearest cho mask nhị phân và kiểm tra chỉ có0/1, cùng kích thước/tọa độ với ảnh.

Loader còn có fallback mask thiếu về rỗng và ảnh đọc lỗi sang item0. MFVLR nên giữ kiểm tra lỗi/hash hiện tại, không âm thầm biến mask fake thiếu thành GT toàn nền hoặc thay mẫu.

Nguồn: [load_mask L318–358](https://github.com/SCLBD/DeepfakeBench/blob/f188b1c105465e2e5377eb536a95022ae0e4522d/training/dataset/abstract_dataset.py#L318-L358), [data_aug L407–454](https://github.com/SCLBD/DeepfakeBench/blob/f188b1c105465e2e5377eb536a95022ae0e4522d/training/dataset/abstract_dataset.py#L407-L454), [đọc lỗi L488–494](https://github.com/SCLBD/DeepfakeBench/blob/f188b1c105465e2e5377eb536a95022ae0e4522d/training/dataset/abstract_dataset.py#L488-L494).

## Đề xuất áp dụng cho MFVLR — chưa triển khai vào lượt đang chạy

1. Sau khi kết thúc đối chứng foreground weight, chọn một cấu hình để thử horizontal flip p0,5, giữ RGB224×224/[0,1], cùng split/seed/budget. Biến đổi image và GT đồng bộ, AR tái tạo đúng ảnh sau augmentation. Giữ validation/test cố định.
2. Nếu flip có lợi, đối chứng thêm rotate nhẹ hoặc brightness/contrast nhẹ từng nhóm. Mức khởi đầu đề xuất riêng cho MFVLR: rotate±5độ,p0,25 hoặc brightness/contrast±0,05,p0,25; không nhận đây là thông số của DeepfakeBench.
3. JPEG/blur làm một nhánh robustness riêng: bắt đầu JPEG80–100 thay vì sao chép ngay40–100; kiểm tra GT còn quan sát được sau suy giảm chất lượng. Hình học GT không đổi với JPEG/blur, nhưng độ hiển lộ dấu vết giả có thể đổi. Đây là suy luận cần thử nghiệm, không phải kết quả đã có.
4. Chưa tự động align/crop lại GenFace. Cần audit ảnh gốc, tỷ lệ foreground bị crop mất, tỷ lệ detector thất bại, và biến đổi mask bằng cùng ma trận. Mọi preprocessing quyết định theo ảnh, không phụ thuộc label; không bỏ mẫu âm thầm. Không dùng GT để chọn crop lúc inference.
5. Mean/std0,5 tương ứng z=2x−1. Nếu chuyển cả target và reconstruction sang miền đó thì MSE(zA,zB)=4MSE(A,B), residual|zB−zA|=2|B−A|. Vì vậy đổi normalize có thể đồng thời đổi scaleAR/residual và cân bằng loss. Nếu thử normalize encoder riêng, phải thiết kế rõ miền input/reconstruction/residual, giữ RGBtarget riêng; không chỉ thay một dòng loader rồi giữ mọi giả định cũ.

Sửa augmentation giữa chừng sẽ phá đối chứng FW4/FW2/FW1 hiện tại. Chỉ thêm ở lượt mới, ghi config/hash/RNG và kiểm tra resume tái lập phép biến đổi, nhất là khi chạy nhiều phiên Kaggle. Các runner hiện tại lưu RNGtorch/CUDA; nếu augmentation dùng random hoặc numpy thì cần lưu/khôi phục thêm trạng thái RNG tương ứng hoặc dùng lịch seed augmentation xác định theo epoch/sample.

Tham khảo repo này giúp chọn các đối chứng tiền xử lý có căn cứ, không xác nhận recipe nguyên bản MFVLR hoặc bảo đảm tăng IoU.
