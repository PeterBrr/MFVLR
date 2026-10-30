# Kiểm tra khả năng tái tạo AD trên GenFace — 25/10/2026

Phép thử trong tài liệu này tiếp tục trọng số đã train. Kết quả **khởi tạo
mới từ đầu**, với bốn nhánh tách AR/FD/FL, lưu tại
[ad_scratch_test_vi.md](ad_scratch_test_vi.md).

AD có thể học tái tạo tốt hơn trên một nhóm ảnh nhỏ khi chỉ tối ưu AR. Cộng
FD/FL làm lỗi tái tạo cao hơn trong cùng ngân sách, đồng thời giảm loss hai
nhiệm vụ đó. Đây là bằng chứng về sự đánh đổi trong cấu hình đã thử; chưa xác
định một nguyên nhân duy nhất của mọi lỗi AD/mask.

## Thiết lập

- Checkpoint nguồn: `artifacts/diffae_pilot/runs/pilot_resize_conv_v1_001/best.pt`.
- Chọn bốn ảnh trong chính subset 1.000 ảnh train đã dùng: real đầu/cuối và
  Diffae đầu/cuối theo thứ tự manifest, ghép hai cặp real/fake; không chọn theo metric.
- IDs: `CelebA_HQ/3903.jpg`, `Diffae/Diffae_Bangs_05733.png`,
  `CelebA_HQ/8025.jpg`, `Diffae/Diffae_smile_05970.png`.
- Hai lượt độc lập: AR riêng và AR + FD + FL; cùng checkpoint ban đầu, fresh
  optimizer và cùng thứ tự ảnh. Đã kiểm tra đầu ra ban đầu bằng nhau chính xác.
- Mỗi lượt 200 optimizer updates; effective batch 4, cộng gradient hai
  microbatch 2. Encoder và decoder đều không đóng băng. Không chạy language.
- Adam LR `1e-4`, betas `(0.9, 0.999)`, eps `1e-8`, weight decay 0; hệ số các
  objective bằng 1; FL weighted mean với class weights `[1,4]` như biến thể.
  Đây là recipe overfit chẩn đoán, khác lượt train gốc LR `1e-5`/decay `1e-3`.
- RGB bilinear 224×224 trong `[0,1]`; AR MSE toàn tensor; head AD tuyến tính,
  không sigmoid. Seed 0, FP32, deterministic, TF32 tắt.

## Kết quả trên chính bốn ảnh train

| Trạng thái | AR/MSE | FD CE | Weighted FL |
|---|---:|---:|---:|
| Trước cập nhật, chung hai lượt | 0,0522895 | 0,211529 | 0,595253 |
| Sau 200 updates, AR riêng | 0,00440862 | 1,61556 | 0,687350 |
| Sau 200 updates, AR + FD + FL | 0,0117923 | 0,000004739 | 0,200278 |

AR riêng giảm MSE **91,57%**. MSE của lượt kết hợp cao hơn lượt AR riêng
**2,675 lần** ở bước 200. Cả hai lượt đều cải thiện AR so với ban đầu; không
diễn giải việc cộng FD/FL là ngăn hoàn toàn khả năng tái tạo.

AR riêng mất khoảng 77 giây, lượt kết hợp khoảng 119 giây cho vòng cập nhật
và các lần đo định kỳ; không tính toàn bộ thời gian nạp/check hash checkpoint.

![Đầu vào, trước cập nhật, AR riêng, AR+FD+FL](../artifacts/genface_ad_overfit_20261025/comparison.png)

Ảnh AR riêng phục hồi được bố cục và các bộ phận mặt hơn ban đầu nhưng vẫn
mờ. Lượt kết hợp có màu/vân rõ hơn và tái tạo kém hơn về MSE trong thử nghiệm.
PNG giới hạn `[0,1]` chỉ để hiển thị; không normalize từng ảnh. NPZ lưu đầu
ra float32 gốc, chưa clip. Đã đối chiếu MSE từ NPZ khớp report trong sai số số học.

## Gradient ban đầu

Đo riêng AR/FD/FL trên **microbatch real/fake đầu tiên** tại checkpoint nguồn,
trước optimizer update:

| Tham số | Cosine AR–FD | Cosine AR–FL |
|---|---:|---:|
| Encoder conv đầu | −0,694 | −0,747 |
| Decoder U1 projection | −0,116 | +0,118 |
| Decoder U4 conv2 | −0,774 | −0,212 |
| Appearance head | −0,839 | Không có đường FL |

Tại appearance head, norm gradient FD khoảng 3,675 và AR khoảng 0,477.
FD đi qua ảnh residual nên có thể cập nhật head tái tạo; FL dùng mask head
riêng, không đi qua appearance head nhưng vẫn cập nhật trunk chung.

Cosine âm là bằng chứng ngược hướng cục bộ, không tương đương từng bước Adam
thực tế làm AR tăng. Chưa đo nhiều batch/checkpoint; chưa tách riêng AR+FD
với AR+FL nên chưa xác định mức tác động độc lập của mỗi loss.

## Diễn giải và việc còn mở

1. Nhánh hiện tại có khả năng cải thiện tái tạo mà không đổi kiến trúc hay thêm
   sigmoid. Chưa có cơ sở xem việc thiếu sigmoid là nguyên nhân duy nhất.
2. Sự đánh đổi AR với FD/FL có bằng chứng trong phép so hai lượt cùng ngân sách;
   không cần language gradient để xuất hiện sự đánh đổi này.
3. AR riêng vẫn mờ ở bước 200. Chưa thể phân biệt hoàn toàn thiếu skip, thông
   tin encoder, LR/optimizer hoặc chưa đủ cập nhật. Không kết luận bắt buộc
   cần skip chỉ từ phép thử này.
4. FD/FL thấp trên bốn ảnh không chứng minh detection/localization tốt trên
   ảnh mới; chưa đo IoU hay validation/test trong thí nghiệm.
5. Nếu cần tách nguyên nhân tiếp: AR+FD và AR+FL riêng với cùng điều kiện;
   hoặc thử decoder có skip trong phép so AR riêng. Những lượt này chưa chạy.

Checkpoint gốc được kiểm hash trước/sau và **không đổi**; không sửa model/loss
chính, không lưu thí nghiệm này thành checkpoint triển khai.

## Tệp và tái chạy

- [Báo cáo số liệu, lịch sử và gradient](../artifacts/genface_ad_overfit_20261025/report.json).
- [Ảnh so sánh](../artifacts/genface_ad_overfit_20261025/comparison.png).
- [Tensor đầu ra gốc](../artifacts/genface_ad_overfit_20261025/raw_reconstructions.npz).
- Script: `tools/test_ad_overfit.py`. Chọn thư mục output mới khi tái chạy.

```powershell
python tools/test_ad_overfit.py --checkpoint artifacts/diffae_pilot/runs/pilot_resize_conv_v1_001/best.pt --output artifacts/genface_ad_overfit_repeat --steps 200 --lr 0.0001
```
