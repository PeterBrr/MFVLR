# Chạy trên HPC: GenFace 50k và nhánh language mô tả vùng (03/11/2026)

## Dữ liệu

`genface50k.zip` (13,5 GB, một file, đã kiểm tra CRC) do `tools/prepare_genface_hpc50k.py` tạo. Không nằm trong git, vì giấy phép GenFace hạn chế phân phối lại.

| Split | Diffae | CelebA-HQ real | FFHQ real |
|---|---:|---:|---:|
| Train | 39.947 | 24.181 | 17.965 |
| Val | 5.031 | 2.993 | 2.038 |
| Test | 5.022 | 2.823 | 0 |
| **Tổng** | **50.000** | **29.997** | **20.003** |

- Split theo danh sách chính thức của GenFace. Diffae lấy ngẫu nhiên (seed 0) trong mỗi split, theo tỷ lệ kích thước split.
- Ảnh real gồm mọi ảnh CelebA-HQ có trên đĩa, cộng ảnh FFHQ lấy trong dải chỉ số của chính split đó (train 0–55.999, val 56.000–62.999). Vì vậy không khuôn mặt FFHQ nào xuất hiện ở hai split.
- Test chỉ có real CelebA, vì trên đĩa không có ảnh FFHQ nào cho test mà không trùng khuôn mặt. Phần real còn thiếu của test được chuyển sang train.
- Gói chứa trọn bộ 10k/2k/2k đã dùng, nên các kết quả trước vẫn so được.
- Mọi ảnh và mask đều được đọc thử. Một ảnh hỏng (`Diffae_smile_08464.png`) bị loại và có ảnh khác bù vào.
- Thêm ảnh real FFHQ là để giảm lối tắt CelebA/FFHQ. Ở thí nghiệm 10k, model chỉ train với real CelebA gọi nhầm 18–24% ảnh real FFHQ là fake ([đối chứng 10k](doi_chung_genface10k_20261103_vi.md)).

## Các bước trên HPC

```bash
git pull
unzip genface50k.zip -d data/                      # tạo data/genface50k/{train,val,test}.csv và các thư mục ảnh
python tools/pack_genface50k_npz.py --extracted data/genface50k --output artifacts/genface50k/package_001
PYTHONPATH=src python tools/run_experiment.py \
    --configs configs/experiments/hpc50k_dicefw2.json configs/experiments/hpc50k_dicefw2_region_language.json \
    --seeds 0 1 2 --output runs/hpc50k
```

- `pack_genface50k_npz.py` tạo tensor giống hệt `PilotFaces`: RGB resize bilinear về 224, mask `> 0` resize nearest. Kết quả là `train.npz` khoảng 16 GB và `val.npz` khoảng 2 GB, không nén. Test không bao giờ được đóng gói.
- Runner giữ ảnh ở dạng uint8 và chỉ chuyển sang float khi lấy từng mẫu. Train 80k ảnh cần khoảng 16 GB RAM, thay vì khoảng 64 GB nếu chuyển cả mảng sang float.
- Chạy lại cùng lệnh thì lượt đang dở được tiếp tục từ checkpoint; lượt đã xong được bỏ qua.
- Hai config dùng 4 epoch, tức khoảng 80.000 cập nhật mỗi lượt. Chỉnh `training.epochs` theo thời gian GPU có được, nhưng **hai nhánh phải dùng cùng số epoch**. Tiêu chí đạt/không đạt phải viết trước khi chạy.

## Nhánh language mô tả vùng (`hpc50k_dicefw2_region_language`)

Hai lựa chọn mới trong `model`. Giá trị mặc định giữ nguyên hành vi cũ:

| Khóa | Mặc định (cũ) | Giá trị mới |
|---|---|---|
| `language_prompt` | `label`: hai prompt cố định theo nhãn real/fake | `region`: 4 câu mô tả thuộc tính bị sửa, vị trí và độ lớn vùng sửa |
| `language_encoder_input` | `prompt`: LE đọc chính prompt cần tái tạo | `neutral`: LE đọc câu trung tính "A photo of a face"; LD vẫn được teacher-force trên prompt đích |

Prompt `region` cho ảnh fake:
1. "A photo of a fake face"
2. "The bangs above the forehead / smile around the mouth of this face is edited", lấy từ tên file Diffae
3. "Most edited pixels are in the upper / middle / lower part of the image": phần ba theo chiều dọc chứa nhiều pixel mask nhất
4. "The edited area is small / medium / large": diện tích mask dưới 10%, 10–17%, hoặc từ 17% trở lên. Hai ngưỡng này lấy từ tứ phân vị trên val 10k và được cố định trước khi chạy.

Ảnh real có một prompt cố định: "No attribute of this face is edited" …

**Lý do.** Báo cáo đối chứng language5000 của @Lucincin cho thấy hai lối tắt. Thứ nhất, LR được giải bằng cách chép nhãn từ prompt, vì LD nhìn thấy prompt qua bộ nhớ của LE. Thứ hai, các block sâu gần như không học do double residual. Với `neutral`, LD chỉ có thể biết câu đích từ ảnh. Với `region`, câu đích mang thông tin về vị trí và thuộc tính bị sửa, chứ không chỉ một bit nhãn. Config dùng thêm `vim_residual: single`, `language_normalization: pre_norm` và `language_visual_gradients: coupled`. Với `stop_all`, language không thể ảnh hưởng tới đặc trưng ảnh. Hệ số loss giữ như language5000 (LR 1e-4, CMC 0,01, KL 0,01) và không được dò.

**Giới hạn.**
- Lượt này đổi cùng lúc prompt, đầu vào của LE, residual và đường gradient. Nếu đạt, không tách được phần nào đóng góp bao nhiêu.
- Language chỉ dùng lúc train; lúc chấm, model chỉ chạy với ảnh.
- Mask GT rất vụn: trung vị 373 mảnh mỗi mask. Mask GT làm mờ Gaussian σ = 3 px chỉ còn IoU 45,6 với chính nó. Language không sửa được giới hạn này.
- Chưa chạy. Chưa có kết quả.
