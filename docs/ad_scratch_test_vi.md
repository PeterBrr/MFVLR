# Thử AD từ trọng số mới trên GenFace — 25/10/2026

**Đã chạy từ đầu, không nạp trọng số đã train.** Với bốn ảnh train và 200
cập nhật, AR riêng học tái tạo rõ hơn; thêm FD/FL làm MSE cuối cao hơn.
Nhánh kết hợp ba loss dao động mạnh. Kết quả hỗ trợ việc kiểm tra tương tác
mục tiêu/độ ổn định tối ưu, chưa chứng minh một nguyên nhân duy nhất của lỗi AD.

## Thiết lập và kiểm chứng

- Kiến trúc resize-conv đang có, bottleneck-only; giữ head AD tuyến tính.
  Toàn bộ nhánh ảnh được khởi tạo mới, seed 0. Không nạp pretrained weights.
- Bốn nhánh độc lập: AR, AR+FD, AR+FL, AR+FD+FL; cùng hash trọng số ban đầu
  và đầu ra ban đầu bằng nhau chính xác. Mỗi nhánh có optimizer Adam mới.
- Cùng bốn ảnh **train**, đúng nhóm đã dùng trong phép thử từ checkpoint:
  `CelebA_HQ/3903.jpg`, `Diffae/Diffae_Bangs_05733.png`,
  `CelebA_HQ/8025.jpg`, `Diffae/Diffae_smile_05970.png`.
- Mỗi nhánh 200 optimizer updates, effective batch 4 bằng hai microbatch 2;
  encoder/decoder đều được học. Không chạy language losses.
- Adam LR `1e-4`, betas `(0.9,0.999)`, eps `1e-8`, weight decay 0; hệ số
  các objective 1; FL weighted mean với class weights `[1,4]`.
- Ảnh RGB `[0,1]`, bilinear 224×224; mask nearest; FP32 deterministic,
  TF32 tắt. Không thêm sigmoid, skip, augmentation hoặc cơ chế cân gradient.
- Checkpoint cũ chỉ cung cấp **config và provenance của dữ liệu/source**.
  Nhánh `--initialization scratch` tạo model bằng constructor và không gọi
  `load_state_dict`. Report ghi `trained_weights_loaded=false`.
  Checkpoint tham chiếu được kiểm hash trước/sau, không thay đổi.

Đây là thử nghiệm học thuộc nhóm nhỏ. Các số dưới đây đều đo trên chính bốn
ảnh train, không phải validation/test hoặc kết quả tái hiện đầy đủ paper.

## Kết quả ở bước 200

| Nhánh | AR/MSE | FD CE | Weighted FL | MSE / AR riêng |
|---|---:|---:|---:|---:|
| Khởi tạo chung, trước cập nhật | 3,039310 | 3,457405 | 0,970724 | — |
| AR riêng | **0,0053204** | 1,955465 | 1,257880 | 1,00× |
| AR+FD | 0,0080293 | 0,00011461 | 0,738538 | 1,51× |
| AR+FL | 0,0135684 | 0,338818 | 0,207117 | 2,55× |
| AR+FD+FL | 0,2320181 | 0,00023067 | 0,650034 | 43,61× |

Các loss không thuộc objective của một nhánh vẫn được đo để đối chiếu,
nhưng không tham gia backward của nhánh đó.

![So sánh AD sau 200 cập nhật](../artifacts/genface_ad_scratch_20261025/comparison_final.png)

Từ trái: input, AR riêng, AR+FD, AR+FL, AR+FD+FL. AR riêng phục hồi được các
bộ phận mặt và màu/bố cục; ảnh vẫn mờ. AR+FD còn hình mặt; AR+FL có nhiều vân
màu hơn; nhánh kết hợp ở bước cuối mất nhiều cấu trúc mặt.

PNG chỉ clip giá trị về `[0,1]` để hiển thị, không normalize theo từng ảnh.
NPZ giữ đầu ra float32 chưa clip; đã đối chiếu MSE từ NPZ với report.

## Nhánh ba loss chưa ổn định

Không diễn giải tỷ lệ 43,61× như một mức suy giảm cố định của phương pháp:

- MSE ở các bước 75/100/125: `0,06546 / 2,15830 / 0,08492`.
- Bước 175 đạt `0,03384`, nhưng bước 200 tăng lên `0,23202`.
- Log tiến trình còn ghi nhận đợt tăng lỗi gần cuối; mọi tensor được kiểm tra
  vẫn hữu hạn. Không gặp lỗi NaN/Inf, nhưng quá trình tối ưu dao động mạnh.
- Ảnh xuất là trạng thái **bước 200 đã quy định**, không chọn ảnh ở bước đẹp
  nhất. Chưa thử giảm LR/cân hệ số hoặc chạy dài hơn để đánh giá hội tụ.

![Đường loss trên cùng bốn ảnh train](../artifacts/genface_ad_scratch_20261025/loss_curves.png)

Biểu đồ dùng thang log, ghi nhận mỗi 25 cập nhật. Thời gian vòng cập nhật và
đo định kỳ lần lượt khoảng 77/117/86/119 giây; không bao gồm toàn bộ nạp,
khởi tạo, hash và xuất ảnh.

## Gradient ban đầu không mặc định xung đột

Ở microbatch real/fake đầu tiên, trước học:

| Tham số | Cosine AR–FD | Cosine AR–FL |
|---|---:|---:|
| Encoder conv đầu | +0,526 | +0,689 |
| Decoder U1 projection | +0,795 | +0,127 |
| Decoder U4 conv2 | +0,540 | +0,057 |
| Appearance head | +0,573 | Không có đường FL |

Khác với gradient âm đo ở checkpoint đã học, các cặp này ban đầu cùng hướng.
Không thể dùng một lần đo để khẳng định các loss luôn nghịch nhau hoặc luôn
thuận nhau. Chưa đo cosine xuyên suốt các nhánh; gradient thô cũng không
tương đương cập nhật Adam.

## Kết luận trong phạm vi phép thử

1. AD hiện tại có thể học giảm lỗi tái tạo từ trọng số mới trên nhóm nhỏ;
   kết quả cải thiện không chỉ nhờ tiếp tục trọng số cũ.
2. Ở cùng 200 cập nhật, thêm riêng FD hoặc FL làm MSE cuối cao hơn AR riêng,
   đồng thời giảm mạnh loss của nhiệm vụ tương ứng so với nhánh AR riêng.
   Đây là sự đánh đổi trong cấu hình đã thử, không phải bằng chứng các loss
   đó luôn gây hại ở mọi giai đoạn hoặc mọi recipe.
3. Nhánh ba loss cho thấy độ bất ổn đáng kiểm tra. Chưa phân biệt đầy đủ
   tương tác gradient, thang/reduction loss và độ nhạy optimizer/LR.
4. Vì AR riêng vẫn mờ, chưa loại trừ hạn chế bottleneck/thiếu skip hoặc chưa
   đủ cập nhật. Chưa chứng minh cần đổi kiến trúc mới tái tạo được.
5. Chưa đo IoU hay khả năng học ảnh mới. Loss thấp trên bốn ảnh không thay
   thế đánh giá detection/localization; không chọn hyperparameter bằng test.

Hướng tiếp theo nếu cần làm rõ hơn: theo dõi gradient theo thời gian, thử LR
nhỏ hơn cho nhánh kết hợp và so decoder có skip dưới cùng objective/budget.
Những lượt này chưa được chạy trong báo cáo này.

## Tệp và tái chạy

- [Report số liệu, hash khởi tạo và lịch sử](../artifacts/genface_ad_scratch_20261025/report.json).
- [Ảnh đầy đủ gồm trạng thái khởi tạo](../artifacts/genface_ad_scratch_20261025/comparison.png).
- [Đầu ra gốc](../artifacts/genface_ad_scratch_20261025/raw_reconstructions.npz).
- [Phép thử trước: tiếp tục trọng số cũ](ad_overfit_test_vi.md).

```powershell
python tools/test_ad_overfit.py --checkpoint artifacts/diffae_pilot/runs/pilot_resize_conv_v1_001/best.pt --initialization scratch --trials ar_only ar_fd ar_fl ar_fd_fl --output artifacts/genface_ad_scratch_repeat --steps 200 --lr 0.0001
```

Chọn output mới khi tái chạy. Không tạo checkpoint triển khai hoặc sửa model,
loss/config train chính trong thí nghiệm này.
