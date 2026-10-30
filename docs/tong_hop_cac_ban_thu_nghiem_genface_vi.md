# Tổng hợp các phiên bản đã thử nghiệm trên GenFace

> Báo cáo chính đã giới hạn theo yêu cầu ở các thử nghiệm từ 26/10/2026 trở đi, có lý do thực hiện và kết quả từng nhóm: [Báo cáo tổng hợp MFVLR](bao_cao_tong_hop_thu_nghiem_mfvlr_20261030.md). Nội dung dưới là bản tóm tắt lịch sử, không dùng thay cho báo cáo đã lọc theo ngày.

Cập nhật 29/10/2026. Phạm vi: GenFace thật + Diffae; tổng hợp các nhóm thử nghiệm có báo cáo lưu trong dự án. Không đánh số thành các bản phát hành phần mềm. Phân biệt overfit trên train, validation và test; các bảng khác dữ liệu/ngân sách không phải đối chứng trực tiếp.

## 1. Các lượt đánh giá trên dữ liệu tách biệt

IoU ở đây là foreground IoU tính từng ảnh giả rồi lấy trung bình. Accuracy dùng detection threshold0,5. Mỗi hàng là mốc cuối định trước, trừ hàng ghi rõ là mốc trong lượt30epoch.

| Phiên bản / nhóm | Train / val | Ngân sách, seed | Language | AUC val | IoU fake val | Nhận xét |
|---|---|---|---|---:|---:|---|
| Encoder chung, decoder resize-conv chưa skip | 32 / 64 | 200 updates, seed0 | Không | 52,54% | 0,1186% | Mask gần toàn nền |
| Tách CNN riêng cho mask, vẫn chung trunk decoder | 32 / 64 | 200 updates, seed0 | Không | 54,79% | 0,0016% | Chưa giúp mask; thêm18,84 triệu tham số |
| A: resize-conv không skip | 128 / 128 | 100 updates, seed0 | Không | 61,87% | 1,585% | Đối chứng trong queue6lượt |
| B: A + skip concat | 128 / 128 | 100 updates, seed0 | Không | 62,84% | 1,029% | MSE giảm khoảng48%, IoU chưa tăng |
| C: B + detach AD trước RE | 128 / 128 | 100 updates, seed0 | Không | 65,99% | 0,850% | Giảm MSE; chặn gradient qua residual về AD |
| D: C + lambda_FL=0,25 | 128 / 128 | 100 updates, seed0 | Không | 66,75% | 1,160% | Chỉ là đối chứng nhỏ, chưa chọn làm cấu hình cuối |
| E: D + language cũ | 128 / 128 | 100 updates, seed0 | Có | 40,77% | 1,201% | PAD pooling, LR có PAD, CMC raw-dot |
| F: D + language EOS/PAD/L2 | 128 / 128 | 100 updates, seed0 | Có | 64,14% | 1,328% | Tốt hơn E; chưa vượt D về detection/tái tạo |
| G: D + sigmoid AD | 128 / 128 | 100 updates, seed0 | Không | 75,39% | 0,610% | AUC tăng nhưng MSE/IoU kém D; không giữ sigmoid trong các lượt lớn |
| FD1, AR1, FL1 | 1.000 / 400 | 5epoch, seed0/1/2 | Không | 85,88% ±7,85 | 11,82% ±5,16 | Trung bình ±SD qua3seed |
| FD0,1, AR1, FL1 | 1.000 / 400 | 5epoch, seed0/1/2 | Không | 96,91% ±0,60 | 13,77% ±4,75 | Detection/MSE cải thiện; IoU chưa nhất quán từng seed |
| FD0,1, train lâu hơn | 1.000 / 400 | 15epoch, seed0/1/2 | Không | 99,42% | 26,20% | IoU SD2,22 điểm; seed0 IoU28,20% |
| Visual5.000, mốc15epoch trong lượt30epoch | 5.000 / 400 | 15epoch, seed0 | Không | 99,9075% | 33,5668% | Đối chứng cùng mốc với language15 |
| Visual5.000, kết thúc30epoch | 5.000 / 400 | 30epoch, seed0 | Không | 99,9700% | 36,4081% | Train IoU48,72%, còn khoảng cách với validation |
| Language5.000 scratch | 5.000 / 400 | 15epoch, seed0 | Có | 99,9600% | 35,5571% | EOS/PAD/L2 + pre-norm + VIM residual2lần |

Các lượt lớn đều dùng skip concat, nearest resize-conv, AD tuyến tính, detach reconstruction trước RE, pixel weight[1,4], effective batch4/microbatch2, Adam imageLR1e-4, weight decay0. Lượt language dùng thêm LR1e-4, CMC0,01, KL0,01 theo hệ số loss và language optimizerLR1e-5. Đây là lựa chọn thực nghiệm, không khẳng định là recipe nguyên bản của tác giả.

## 2. Các thử nghiệm chẩn đoán loss trên tập rất nhỏ

Các kết quả dưới là trên chính ảnh train, không chứng minh khả năng tổng quát hóa.

### Overfit16ảnh local, 400updates

| Loss bật | IoU fake train | MSE AD | Kết luận trong phép thử |
|---|---:|---:|---|
| FL_only | 1,620% | 0,725601 | Không tối ưu AD; mask riêng chưa hội tụ ổn định |
| AR_only | 13,848% | 0,000386 | Mask không được tối ưu, gần toàn foreground; IoU không thể hiện học mask |
| AR+FL | 48,810% | 0,003816 | Chứng minh có khả năng học mask trên tập nhỏ |
| AR+FD+FL | 37,333% | 0,005638 | Thêm FD1 làm mask/AD kém hơn trong cấu hình này |

### Quét lambda_FD trên Kaggle,16ảnh,400updates

| Lambda_FD, AR=FL=1 | IoU fake train | MSE AD |
|---:|---:|---:|
| 0 | 45,777% | 0,003529 |
| 0,1 | 45,338% | 0,003953 |
| 0,25 | 40,527% | 0,004708 |
| 1 | 42,480% | 0,005747 |

FD0,1 là ứng viên cân bằng và được xác nhận tiếp trên1.000ảnh/3seed. Quan hệ không đơn điệu; không kết luận giảm FD luôn cải thiện mask. Không so trực tiếp số local với Kaggle vì khác phiên bản torch.

Chẩn đoán trước đó trên4ảnh từ checkpoint resize-conv: AR riêng200updates đạt MSE0,004409, so AR+FD+FL0,011792. Đây là tiếp tục tối ưu checkpoint cũ, khác các lượt scratch bên trên. Gradient cục bộ có cả thuận/nghịch; không chứng minh mọi loss luôn xung đột.

## 3. Language: thay đổi và kiểm tra trước khi mở rộng

- Đã thử EOS pooling, bỏ PAD khỏi LR, L2 feature trong CMC, và opt-in pre-norm LE/LD để kiểm soát độ lớn activation/logits khi dùng residual VIM hai lần.
- Preflight double-residual chưa pre-norm có logitRMS khoảng1,57e8; thêm pre-norm giảm còn khoảng0,455 trong phép đo cố định. Đây là số ổn định số học, không phải metric test.
- Pilot readiness16ảnh/8updates so stop_all và coupled chỉ kiểm tra luồng gradient; không dùng để tuyên bố tăng chất lượng.
- Lượt5.000ảnh language được khởi tạo từ đầu, không fine-tune checkpoint visual30epoch. Hai phiên Kaggle tiếp tục checkpoint của chính lượt mới để hoàn thành15epoch.
- Khi test, visual_only chỉ nhận ảnh. Language đóng góp thông qua trọng số đã học trong train; không cấp prompt theo nhãn thật/giả cho model lúc test.

## 4. Hai checkpoint đã test trên400ảnh GenFace

| Chỉ số test | Visual30epoch | Language15epoch |
|---|---:|---:|
| AUC | 99,9350% | 99,9425% |
| Accuracy | 99,00% | 98,50% |
| IoU fake macro | 37,2796% | 36,1392% |
| Dice fake macro | 53,8778% | 52,6805% |
| Diện tích báo giả trên real | 0,4827% | 1,7858% |
| MSE AD | 0,00028363 | 0,00059043 |

Hai checkpoint khác số epoch, không dùng bảng test này để kết luận riêng tác động language. So cùng epoch15 trên validation: language tăng IoU1,99điểm và giảm MSE26,1%, nhưng vùng báo giả real tăng0,81%→1,48%. Chỉ một seed và baseline lịch sử, chưa đủ kết luận chắc chắn về lợi ích ổn định.

Output language: `artifacts/language5000_scratch_kaggle_20261028/test_001/`; có400ảnh mỗi thư mục input/AD/GT/mask/overlay,400mảng xác suất, comparison.png, preview.png, index.json và test_metrics.json. Mask là output thô, không gate theo detection. AD PNG clip[0,1] chỉ để hiển thị, MSE tính từ output thô.

## 5. Nhận định hiện tại

1. Model đã học detection tốt trong split GenFace/Diffae đang dùng; mask cải thiện rõ khi train đủ dữ liệu/cập nhật so với pilot rất ngắn.
2. Skip/detach giúp tái tạo trong đối chứng nhỏ; giảm FD về0,1 hữu ích hơn FD1 trong các recipe đã kiểm tra. Chưa chứng minh đã loại hết cạnh tranh gradient ở encoder/trunk.
3. Sigmoid đã thử nhưng chưa cho lợi ích đồng thời về tái tạo/mask; các checkpoint lớn đang dùng AD tuyến tính.
4. Language sửa lại giúp validation ở cùng mốc15epoch về IoU/MSE, kèm tăng báo giả real. Token accuracy cao trên prompt lặp không chứng minh hiểu ảnh.
5. Mask còn tô dư/thiếu vùng, AD một số ảnh còn vệt màu; detection thật không tự ép mask bằng0. Chưa xác định nguyên nhân duy nhất. Chưa xác minh chia theo danh tính; chưa đánh giá cross-generator. Test đã từng dùng trong lịch sử dự án.

Ưu tiên tiếp: đối chứng visual/language cùng15epoch qua nhiều seed và cùng runner; kiểm tra FP/FN theo kích thước vùng GT trước khi thay thêm kiến trúc hoặc trọng số. Chưa tự khởi chạy các lượt mới trong báo cáo này.

## Nguồn số liệu trong dự án

- `docs/ad_overfit_test_vi.md`, `docs/split_mask_encoder_pilot_vi.md`.
- `docs/genface_ablation_20261026_vi.md`, `docs/genface_sigmoid_20261026_vi.md`.
- `docs/genface_loss_overfit_local_vi.md`, `docs/genface_fd_weight_kaggle_vi.md`.
- `docs/genface_fd_validation_results_vi.md`, `docs/genface_fd_epoch15_vi.md`.
- `docs/genface5000_epoch30_vi.md`, `docs/genface5000_test_vi.md`.
- `docs/language_readiness_20261028_vi.md`, `docs/language5000_scratch_kaggle_vi.md`.

Các pilot A0/resize-conv đời đầu và kiểm tra decoder lịch sử xem thêm `docs/mask_decoder_audit.md`, `docs/decoder_root_cause_vi.md`; không gộp metric khác protocol vào bảng đối chứng trên.
