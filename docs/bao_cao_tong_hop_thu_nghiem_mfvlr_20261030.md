# Báo cáo tổng hợp các thử nghiệm MFVLR trên GenFace từ 26/10/2026

**Ngày cập nhật: 30/10/2026 — khoảng báo cáo: 26–30/10/2026, tính cả ngày 26/10. Phạm vi dữ liệu: ảnh thật và ảnh Diffae trong GenFace.**

Chỉ tổng hợp các phép đo và lượt train/test thực hiện từ ngày 26/10/2026. Checkpoint cũ có thể được dùng làm điểm khởi tạo cho một thử nghiệm trong khoảng này, nhưng kết quả của các lượt trước 26/10 không được đưa vào bảng hoặc kết luận của báo cáo. Mỗi nhóm nêu lý do thực hiện, thiết lập và kết quả; các lượt tiếp tục checkpoint không được tính thành một đối chứng kiến trúc mới.

## 1. Mục tiêu và kết quả chính

Các thử nghiệm tập trung vào ba câu hỏi: vì sao mask gần giống nhau hoặc gần toàn nền; vì sao AD tái tạo kém; và cách cân các mục tiêu ảnh/ngôn ngữ để cải thiện detection cùng localization.

Những kết quả chính đã có:

- Skip connection và detach reconstruction trước residual encoder cải thiện tái tạo trong đối chứng nhỏ.
- Model học được mask trên tập nhỏ: AR+FL đạt IoU train 48,81% trên 16 ảnh. Điều này không tương đương khả năng tổng quát hóa.
- Giảm lambda_FD từ 1 xuống 0,1 cải thiện detection và MSE ở cả 3 seed trong pilot 1.000 ảnh/5 epoch; lợi ích IoU không nhất quán từng seed.
- Image-only 5.000 ảnh/30 epoch đạt IoU validation 36,41%, test 37,28%. Language scratch 5.000 ảnh/15 epoch đạt IoU validation 35,56%, test 36,14%.
- So cùng epoch 15 trên validation, language hơn image-only về IoU 1,99 điểm phần trăm và MSE giảm 26,1%, nhưng báo giả trên ảnh thật tăng. Chưa đủ bằng chứng nhiều seed.
- Đối chứng foreground weight 4/2/1 chưa hoàn thành 15 epoch. Weight 2 thường bớt tô dư hơn 4 nhưng giảm recall; xếp hạng IoU thay đổi theo epoch, chưa chốt weight tối ưu.

Các kết quả là thực nghiệm nội bộ, chưa được coi là tái hiện đầy đủ paper hoặc đánh giá cross-generator.

## 2. Dữ liệu, cách đo và quy tắc so sánh

### 2.1 Dữ liệu và tiền xử lý

- Chỉ dùng GenFace real + Diffae trong các đối chứng của báo cáo.
- Mask GT dùng từ dataset do người dùng tải theo nguồn tác giả cung cấp; không tạo GT từ ảnh AD trong các lượt train đang báo cáo.
- RGB 3 kênh, resize 224×224 bằng PIL bilinear, float32 chia 255 về [0,1]. Mask nhị phân 0/1, nearest resize. Không thêm face alignment/crop hoặc augmentation trong các lượt đối chứng chính.
- Các lượt train chính trong phạm vi báo cáo dùng 16/128/1.000/5.000 ảnh; các phép đo gradient dùng nhóm mẫu riêng được ghi trong từng mục. Các lượt lớn dùng 400 validation và 400 test, cân bằng thật/giả. Tập 5.000 train có 2.500 thật/2.500 Diffae.
- Đã kiểm ID/hash ảnh và không trùng nội dung RGB sau resize giữa các split trong gói 5.000 ảnh. **Chưa xác minh subject-disjoint** vì thiếu ánh xạ danh tính/nguồn đầy đủ.

### 2.2 Ý nghĩa chỉ số

| Chỉ số | Cách đọc |
|---|---|
| AUC/accuracy/F1 detection | Đánh giá nhãn thật/giả cho toàn ảnh |
| IoU fake macro | Tính intersection/union foreground cho từng ảnh giả, rồi lấy trung bình |
| Dice fake macro | Tính Dice foreground từng ảnh giả, rồi lấy trung bình |
| Pixel precision/recall | Cộng TP/FP/FN toàn split rồi tính; không đồng nhất với trung bình IoU từng ảnh |
| FP area real | Trung bình tỷ lệ pixel bị dự đoán giả trên mỗi ảnh thật |
| MSE AD | Sai số tái tạo trên RGB thô, không clip trước khi tính |

FL trong báo cáo là objective localization đang dùng weighted pixel cross-entropy, không mặc định là Focal Loss. Các loss khác: AR tái tạo ảnh, FD phân loại thật/giả, LR tái tạo ngôn ngữ, CMC tương phản ảnh–ngôn ngữ và KL alignment.

Các lượt đánh giá detection/localization trong phạm vi báo cáo dùng detection threshold cố định 0,5, mask argmax; không gate mask theo nhãn detection. Test chỉ nhận ảnh; nhãn/mask GT dùng để tính metric sau dự đoán.

Không xếp hạng trực tiếp các hàng khác tập dữ liệu, số cập nhật, môi trường hoặc loss. Báo cáo dùng các mốc cuối định trước; các mốc giữa quá trình chỉ phục vụ xem đường học. Test đã từng được xem trong lịch sử dự án, không tuyên bố là một tập hoàn toàn chưa từng dùng trong suốt quá trình nghiên cứu.

## 3. Ngày 26/10: đối chứng skip, detach, language và sigmoid

### 3.1 Sáu lượt 128 ảnh từ đầu

**Lý do thực hiện:** cần tách ảnh hưởng của skip connection, đường gradient qua residual, hệ số FL và nhóm sửa language. Mỗi cặp liền nhau chỉ đổi thành phần đã ghi; tất cả dùng cùng dữ liệu và ngân sách để tránh nhầm lợi ích kiến trúc với việc train lâu hơn.

| Lượt | Lý do riêng của lượt thử |
|---|---|
| A | Làm đối chứng resize-conv không skip để có điểm so sánh cho các thay đổi sau. |
| B so với A | Kiểm tra đưa đặc trưng nhiều độ phân giải xuống decoder có giúp phục hồi chi tiết ảnh hay không. |
| C so với B | Kiểm tra chặn gradient qua ảnh AD vào residual encoder có giảm tác động bất lợi lên tái tạo hay không. |
| D so với C | Kiểm tra giảm lambda_FL về 0,25 có cải thiện cân bằng AR–FL hay chỉ làm mask yếu hơn. |
| E so với D | Kiểm tra bổ sung language theo cách cũ có giúp nhánh ảnh trong cùng ngân sách hay không. |
| F so với E | Kiểm tra gói EOS pooling, bỏ PAD khỏi LR và L2 CMC có cải thiện so với language cũ hay không. |

**Thiết lập:**

128 train/128 validation,100 updates/lượt, seed 0, effective batch 4/microbatch 2, Adam LR 1e-5/WD 1e-3; pixel weights[1,4]. A dùng resize-conv không skip, không phải baselineA0 transposed-convolution. E/F bật gradient language về visual.

| Lượt | Thay đổi | AUC val | IoU fake val | MSE AD |
|---|---|---:|---:|---:|
| A | Không skip, residual coupled | 61,87% | 1,585% | 0,083683 |
| B | A + skip concat 4cấp | 62,84% | 1,029% | 0,043693 |
| C | B + detach AD trước RE | 65,99% | 0,850% | 0,018745 |
| D | C + lambda_FL0,25 | 66,75% | 1,160% | 0,011966 |
| E | D + language cũ | 40,77% | 1,201% | 0,013010 |
| F | D + EOS/bỏ PAD/L2 CMC | 64,14% | 1,328% | 0,012300 |

**Kết quả thu được:** A→B giảm MSE khoảng 48%; B→C giảm thêm MSE từ 0,043693 xuống 0,018745. C→D giảm MSE nhưng IoU vẫn chỉ 1,16%. E làm AUC giảm còn 40,77%; F phục hồi lên 64,14%, chưa vượt D ở AUC/MSE. Mask vẫn gần toàn nền ở cả sáu lượt. F tốt hơn E nhưng chưa tách được đóng góp riêng của EOS/PAD/L2 vì thay ba yếu tố cùng lúc; 100 cập nhật chưa đủ kết luận về lợi ích language nói chung.

Nguồn: [sáu đối chứng](genface_ablation_20261026_vi.md).

### 3.2 Sigmoid tại đầu ra AD

**Lý do thực hiện:** AD tuyến tính có thể tạo RGB âm hoặc lớn hơn 1 và ảnh xuất còn vệt màu. Thử sigmoid để kiểm tra việc giới hạn output trong cùng miền [0,1] với input có giúp tái tạo và mask tốt hơn hay không.

**Thiết lập:**

So D tuyến tính và G sigmoid, cùng 128 train/128 val/100 updates. Chỉ AD thêm sigmoid, mask vẫn trả logits.

| AD | AUC val | IoU fake val | MSE AD |
|---|---:|---:|---:|
| Tuyến tính | 66,75% | 1,160% | 0,011966 |
| Sigmoid | 75,39% | 0,608% | 0,031468 |

**Kết quả thu được:** sigmoid giới hạn RGB trong [0,1] và nâng AUC 66,75%→75,39%, nhưng MSE tăng 0,011966→0,031468 và IoU giảm 1,160%→0,608%. Vì chưa cải thiện đồng thời tái tạo và mask, các lượt 5.000 ảnh giữ AD tuyến tính. Việc PNG đẹp hơn không chứng minh đã bật sigmoid; PNG chỉ clip để hiển thị.

Nguồn: [đối chứng sigmoid](genface_sigmoid_20261026_vi.md).

### 3.3 Gradient AR–FL sau skip/detach

**Lý do thực hiện:** sau khi detach AD trước RE, AR và FL vẫn cập nhật encoder/trunk chung. Cần đo trực tiếp norm và cosine để kiểm tra chúng đang hỗ trợ hay cạnh tranh tại những tham số này, thay vì suy từ trị số loss hoặc ảnh output.

**Thiết lập và kết quả đo:**

Đo 8 cặp train tại khởi tạo và sau 16 updates, không cập nhật trong phép đo. Cosine trung bìnhAR–FL tại CNN/trunk lần lượt−0,2476/−0,1994 lúc đầu và−0,1854/−0,1530 sau 16 bước. Tỷ lệ norm FL đã nhân 0,25 so AR tăng từ khoảng 0,078 lên 0,56 ởtrunk.

**Kết quả thu được:** các nhóm đã đo có cosine trung bình âm ở cả hai mốc, cho thấy cạnh tranh gradient cục bộ; độ lớn FL tương đối tăng sau 16 bước. Detach chưa loại tương tác AR–FL tại trunk/encoder. Phép đo chỉ bao phủ các batch/trạng thái đã chọn và không tương đương hướng cập nhật Adam thực tế.

Nguồn: [EOS/PAD/L2 và gradient](eos_pad_l2_va_gradient_ar_fl_vi.md).

## 4. Ngày 26–27/10: kiểm tra loss và trọng số FD

### 4.1 Overfit 16 ảnh local,400 updates

**Lý do thực hiện:** mask trên pilot lớn hơn còn gần toàn nền. Trước khi thay kiến trúc tiếp, cần kiểm tra model có học thuộc được mask trên 16 ảnh hay không và mục tiêu nào làm kết quả thay đổi.

| Objective | Lý do đưa vào đối chứng |
|---|---|
| FL_only | Kiểm tra khả năng học mask khi chỉ có giám sát pixel. |
| AR_only | Kiểm tra riêng khả năng tái tạo; mask của lượt này chỉ là số chẩn đoán vì không được huấn luyện. |
| AR+FL | Kiểm tra AR hỗ trợ hay làm giảm khả năng học mask so với FL_only. |
| AR+FD+FL | Kiểm tra ảnh hưởng khi đưa detection trở lại so với AR+FL. |

**Thiết lập:**

Scratch, cùng khởi tạo/lịch batch; skip/detach, AD tuyến tính, không language. Adam LR 1e-4/WD 0, pixel weights[1,4], loss bật có hệ số 1.

| Objective | IoU fake train | Dice fake train | MSE AD | FP area real |
|---|---:|---:|---:|---:|
| FL_only | 1,62% | 3,14% | 0,725601 | 0,257% |
| AR_only | 13,85%* | 24,23%* | 0,000386 | 99,997%* |
| AR+FL | 48,81% | 65,49% | 0,003816 | 0,000% |
| AR+FD+FL | 37,33% | 54,23% | 0,005638 | 0,428% |

**Kết quả thu được:** AR+FL đạt IoU train 48,81%, chứng minh model có thể học mask trên tập nhỏ; thêm FD với hệ số 1 làm IoU giảm còn 37,33%. FL_only cuối lượt chỉ đạt 1,62%, nên không thể kết luận AR chỉ gây hại. AR_only tái tạo tốt nhưng mask gần toàn foreground vì không được giám sát; IoU có dấu * không được dùng để đánh giá khả năng học mask. Đây là overfit train, chưa phải kết quả trên ảnh mới.

Nguồn: [overfit loss local](genface_loss_overfit_local_vi.md).

### 4.2 Quét lambda_FD trên 16 ảnh Kaggle

**Lý do thực hiện:** lượt local cho thấy thêm FD1 làm mask/AD kém hơn AR+FL. Cần thử các mức 0/0,1/0,25/1 để xem giảm ảnh hưởng detection có giữ được cả detection và localization hay không. Chạy lại toàn bộ đối chứng trên cùng môi trường Kaggle để tránh quy chênh lệch phiên bản torch cho trọng số loss.

**Vai trò từng mức:** FD0 là đối chứng bỏ detection; FD0,1 và FD0,25 kiểm tra mức giảm trung gian; FD1 giữ recipe ba loss hệ số bằng nhau.

**Thiết lập:**

Cùng 400 updates/seed 0; chạy lại cả đối chứng trên cùng môi trường Kaggle, không trừ trực tiếp số local khác phiên bản torch.

| LambdaFD, AR=FL=1 | IoU fake train | MSE AD | Precision pixel | Recall pixel |
|---:|---:|---:|---:|---:|
| 0 | 45,78% | 0,003529 | 47,10% | 90,80% |
| 0,1 | 45,34% | 0,003953 | 59,83% | 64,61% |
| 0,25 | 40,53% | 0,004708 | 51,86% | 64,55% |
| 1 | 42,48% | 0,005747 | 44,59% | 86,16% |

**Kết quả thu được:** FD0,1 đạt IoU 45,34%, gần FD0 là 45,78% và cao hơn FD1 là 42,48%; MSE cũng thấp hơn FD1. Các lượt có FD đều học detection trên 16 ảnh train. FD0,25 không nằm giữa hai mức về IoU, nên quan hệ không đơn điệu. Cosine FD–FL cuối tại CNN trung bình +0,369 ở FD0,1 so với −0,145 ở FD1, nhưng chỉ đo 4 cặp. Chọn FD0,1 làm ứng viên xác nhận trên dữ liệu lớn hơn, chưa coi là hệ số tối ưu.

Nguồn: [quétFD trênKaggle](genface_fd_weight_kaggle_vi.md).

### 4.3 Xác nhận trên 1.000 train/400 validation,3 seed

**Lý do thực hiện lượt 5 epoch:** kết quả overfit 16 ảnh có thể chỉ phản ánh ghi nhớ. Cần xác nhận FD0,1 so với FD1 trên validation riêng và ba seed để xem lợi ích có tồn tại khi đổi khởi tạo/dữ liệu batch hay không.

**Thiết lập lượt 5 epoch:**

5 epoch/lượt, khởi tạo theo cặp seed 0/1/2, giữAR=FL=1. Giá trị±SD là độ lệch chuẩn mẫu giữa 3 seed, không phải khoảng tin cậy.

| LambdaFD | AUC val | IoU fake val | MSE AD |
|---:|---:|---:|---:|
| 0,1 | 96,91% ±0,60 điểm | 13,77% ±4,75 điểm | 0,002303 |
| 1 | 85,88% ±7,85 điểm | 11,82% ±5,16 điểm | 0,002904 |

**Kết quả thu được sau 5 epoch:** FD0,1 cải thiện AUC trung bình khoảng 11,03 điểm phần trăm và giảm MSE; AUC/MSE tốt hơn ở cả ba seed. IoU trung bình tăng khoảng 1,94 điểm nhưng không cùng chiều ở mọi seed. Chưa khẳng định ý nghĩa thống kê hoặc FD là nguyên nhân duy nhất.

**Lý do thực hiện lượt 15 epoch:** detection đã học nhưng localization còn thấp sau 5 epoch; cần kiểm tra thiếu số cập nhật có phải một phần nguyên nhân hay không, giữ kiến trúc/loss và dữ liệu không đổi.

**Thiết lập:** train scratch lại FD0,1 tới 15 epoch, cùng ba seed. Mốc 5 epoch tái lập chính xác metric của lượt trước:

| Mốc | AUC val trung bình | IoU fake val trung bình | MSE AD |
|---|---:|---:|---:|
| 5 epoch | 96,91% | 13,77% | 0,002303 |
| 10 epoch | 99,04% | 26,39% | 0,001867 |
| 15 epoch | 99,42% | 26,20% | 0,001717 |

**Kết quả thu được sau 15 epoch:** IoU trung bình tăng 13,77%→26,20% so với epoch 5; AUC tăng tới 99,42%. IoU seed 0/1/2 là 28,20%/23,81%/26,57%. Tuy nhiên IoU trung bình không tăng từ epoch 10→15, nên tăng epoch có ích giai đoạn đầu nhưng chưa đủ sửa giới hạn của mask.

Nguồn: [FD5epoch](genface_fd_validation_results_vi.md), [FD15epoch](genface_fd_epoch15_vi.md).

## 5. Ngày 27–29/10: mở rộng 5.000 ảnh và bổ sung language

### 5.1 Image-only 5.000 ảnh,30 epoch

**Lý do thực hiện lượt train:** IoU trên 1.000 ảnh tăng chậm sau khoảng 10 epoch. Mở rộng lên 5.000 ảnh và 30 epoch để kiểm tra khả năng cải thiện khi tăng độ đa dạng ảnh và tổng ngân sách cập nhật, đồng thời tạo baseline visual trước khi thêm language.

**Thiết lập:**

Seed 0,37.500 updates, Adam LR 1e-4/WD 0, effective batch 4/microbatch 2, FP32/TF32 off. Skipconcat/nearestresizeconv/AD tuyến tính/detach AD trướcRE; FD 0,1+AR 1+FL 1, pixel weights[1,4]. Không language.

| Kết quả cuối | Train | Validation | Test |
|---|---:|---:|---:|
| AUC | ≈100% | 99,97% | 99,935% |
| Accuracy | 100% | 99,00% | 99,00% |
| IoU fake macro | 48,72% | 36,41% | 37,28% |
| Dice fake macro | 65,24% | 53,01% | 53,88% |
| FP area real | 0,0147% | 0,4084% | 0,4827% |
| MSE AD | 0,00029015 | 0,00027809 | 0,00028363 |

**Kết quả thu được từ train/validation:** IoU validation đạt 36,41%, so với 28,20% của seed 0 trên 1.000 ảnh/15 epoch. Vì cả dữ liệu và số cập nhật cùng tăng, chưa tách được đóng góp riêng của mỗi yếu tố. IoU train 48,72% cao hơn validation 12,31 điểm, cho thấy vẫn cần cải thiện khả năng tổng quát hóa mask.

**Lý do thực hiện test:** sau khi chốt checkpoint cuối epoch 30, dùng split test để xác nhận chất lượng trên nhóm ảnh khác với train/validation, giữ nguyên ngưỡng và không dùng test chọn cấu hình.

**Kết quả thu được từ test:** AUC 99,935%, accuracy 99,00%, IoU 37,28%, Dice 53,88%. Test cao hơn validation một chút có thể do khác mẫu; không phải bằng chứng không overfit. Tập test đã được xem trong lịch sử dự án nên giới hạn này vẫn áp dụng.

Nguồn: [visual5.000/30](genface5000_epoch30_vi.md), [testvisual](genface5000_test_vi.md).

### 5.2 Sửa language và kiểm tra ổn định số học

**Lý do thực hiện:** cần đưa language trở lại sau khi đã có baseline visual học được mask. Các điểm cần kiểm tra là pooling vào đúng EOS, PAD không chi phối LR, độ lớn feature trong CMC và sự khuếch đại activation khi dùng residual VIM hai lần.

**Thay đổi và lý do:**

| Thành phần | Lý do kiểm tra hoặc sửa |
|---|---|
| EOS pooling | Lấy feature tại EOS thực sự thay vì vị trí cuối có thể là PAD. |
| Bỏ PAD khỏi LR | Tránh dành phần lớn mục tiêu tái tạo cho token đệm không có nội dung. |
| L2 trong CMC | Giảm ảnh hưởng độ lớn vector lên similarity ảnh–ngôn ngữ. |
| VIM double so với single | Kiểm tra lựa chọn residual hai lần đã thống nhất và tác động đến scale activation. |
| Pre-norm/final LN | Kiểm tra khả năng kiểm soát logits/gradient khi bật double residual. |

Đã triển khai EOScuối thực sự, loại targetPAD khỏiLR nhưng giữ token-sum/batch-mean, L2feature choCMC; thêm tùy chọn VIMresidual 2x+Delta và pre-norm/finalLN trongLE/LD. Không khẳng định các chi tiết normalization là recipe đã được tác giả xác nhận.

**Kết quả thu được từ preflight:** trên 2 ảnh với language khởi tạo mới và feature visual đã học, RMS logit single/baseline khoảng 1.317; double/baseline khoảng 1,57e8; double/pre-norm khoảng 0,455. Pre-norm/final LN kiểm soát đầu ra trong phép đo này. Đây không phải bằng chứng tổng quát hóa; residual stream bên trong vẫn có thể lớn, còn singleton K/V và full LE memory chưa được giải quyết chỉ bằng LN.

**Lý do thực hiện pilot stop_all/coupled:** kiểm tra toàn bộ luồng backward có chạy hữu hạn và mức tác động lên visual thay đổi thế nào khi chặn hoặc cho phép gradient language; chưa nhằm chọn checkpoint tốt nhất.

**Kết quả thu được từ pilot:** 16 ảnh/8 updates, warm-start visual; stop_all có IoU train 48,27%→49,34%, coupled 48,27%→48,76%. Gradient và tham số hữu hạn. Không có image-only control cùng 8 bước nên chưa thể dùng chênh lệch này để kết luận language giúp mask. Stop_all vẫn cho visual học AR/FL/FD.

Nguồn: [language readiness](language_readiness_20261028_vi.md). Ghi chú trạng thái “chưa train dài” trong tài liệu đó là lịch sử tại thời điểm viết; kết quả mới ở phần kế tiếp.

### 5.3 Language 5.000 ảnh,15 epoch từ đầu

**Lý do thực hiện lượt train:** sau khi sửa language và kiểm tra số học, cần đánh giá đóng góp của nhánh này ở quy mô thực tế. Train từ đầu để tránh lợi thế của việc fine-tune checkpoint visual đã học 30 epoch; đối chiếu với mốc visual cùng 15 epoch.

**Thiết lập:**

Không fine-tune checkpointvisual 30 epoch. Khởi tạo visual ban đầu khớp baseline cùng torch; cùng seed 0/data/lịch batch. Language coupled, double VIM/pre-norm/EOS/PAD/L2; hệ sốFD 0,1/AR 1/FL 1/LR 1e-4/CMC 0,01/KL 0,01. Optimizer image LR 1e-4/language LR 1e-5. CMCbatch thực tế 2, accumulation không tạo 4 negatives.

| Validation cùng 15 epoch | Image-only | Có language |
|---|---:|---:|
| AUC | 99,9075% | 99,9600% |
| Accuracy | 94,75% | 99,00% |
| IoU fake macro | 33,57% | 35,56% |
| Dice fake macro | 49,80% | 52,08% |
| MSE AD | 0,00079936 | 0,00059086 |
| FP area real | 0,81% | 1,48% |

**Kết quả thu được trên validation:** language tăng IoU 33,57%→35,56%, Dice 49,80%→52,08% và giảm MSE khoảng 26,1%; đổi lại FP area real tăng 0,81%→1,48%. Baseline image-only là mốc lịch sử, chưa có control mới qua cùng runner/nhiều seed. Token accuracy 100% trên prompt lặp/teacher-forcing/full memory chưa chứng minh hiểu ảnh.

**Lý do thực hiện test và xuất ảnh:** kiểm tra checkpoint language trên split test, xem metric có nhất quán với validation và quan sát trực tiếp chất lượng AD/mask, đặc biệt là vùng báo giả trên ảnh thật.

**Kết quả thu được từ test:** AUC 99,9425%, accuracy 98,50%, IoU 36,14%, Dice 52,68%, FP area real 1,79%, MSE 0,00059043. Đã xuất input/AD/GT/mask/overlay cho 400 ảnh. Inference chỉ nhận ảnh, không dùng prompt/nhãn GT; parity validation local–Kaggle có AUC/IoU lệch 0, MSE lệch khoảng 6,55e-12.

Không dùng test visual 30 epoch 37,28% so test language 15 epoch 36,14% để kết luận riêng tác độnglanguage vì khác ngân sách. Cả hai checkpoint đã xuấtAD/mask. [Ảnh language](../artifacts/language5000_scratch_kaggle_20261028/test_001/comparison.png), [metricstest](../artifacts/language5000_scratch_kaggle_20261028/test_001/test_metrics.json), [báo cáo lượtlanguage](language5000_scratch_kaggle_vi.md).

## 6. Ngày 29–30/10: đối chứng foreground weight 4/2/1 — chưa hoàn thành

**Lý do thực hiện:** test language cho precision pixel khoảng 39,12% nhưng recall khoảng 72,76%, mask còn tô dư và báo giả trên ảnh thật. Cần kiểm tra giảm foreground weight có thu hẹp vùng dư và cải thiện IoU hay chỉ làm tăng bỏ sót.

| Weight pixel giả | Lý do đưa vào đối chứng |
|---:|---|
| 4 | Giữ baseline đã hoàn thành để biết ảnh hưởng của việc giảm weight. |
| 2 | Thử mức trung gian, kỳ vọng giảm tô dư mà vẫn giữ đủ phản hồi với pixel giả. |
| 1 | Bỏ ưu tiên tương đối cho foreground để kiểm tra giới hạn đánh đổi precision–recall. |

**Thiết lập:**

Mục tiêu giữ cấu hình language 5.000 ảnh/15 epoch, chỉ thay foreground weight. FW4 là đối chứng scratch đã hoàn thành; FW2/FW1 train scratch riêng, cùng khởi tạo/lịch batch/data. Không fine-tuneFW 4 và không dùngtest chọn weight. Weighted CE vẫn `sum(w[target]*NLL)/sum(w[target])` trongmicrobatch, vì vậy đổiweight thay cả cân bằng lớp và scalegradient thực tế.

### 6.1 Validation cùng epoch 5, hoàn toàn trên Kaggle 1

**Lý do theo dõi mốc đầu:** có đối chứng cùng 5 epoch để kiểm tra phản ứng ban đầu khi đổi weight trước khi dùng thêm quota cho mục tiêu 15 epoch.

**Kết quả thu được:** weight 2 tăng IoU nhẹ so với 4 và giảm báo giả real; weight 1 giảm recall mạnh, IoU thấp. Đây là mốc đầu để xem hướng học, chưa phải kết quả 15 epoch.

| Weight | IoU fake | Precision pixel | Recall pixel | FP area real |
|---:|---:|---:|---:|---:|
| 4 | 29,99% | 33,79% | 65,10% | 1,96% |
| 2 | 30,68% | 46,07% | 44,28% | 1,07% |
| 1 | 12,84% | 57,00% | 12,69% | 0,36% |

### 6.2 Validation cùng epoch 9, hoàn toàn trên Kaggle 1

**Lý do theo dõi mốc tiếp:** kiểm tra ưu thế ban đầu của weight 2 có duy trì khi train lâu hơn hay không, so tại cùng ngân sách.

**Kết quả thu được:** weight 4 có IoU cao hơn weight 2, trái với thứ hạng ở epoch 5; weight 2 vẫn ít báo giả và precision cao hơn. Weight 1 còn bỏ sót nhiều foreground.

| Weight | IoU fake | Precision pixel | Recall pixel | FP area real |
|---:|---:|---:|---:|---:|
| 4 | 34,43% | 37,52% | 71,95% | 2,24% |
| 2 | 33,03% | 49,00% | 48,60% | 1,05% |
| 1 | 20,78% | 58,65% | 22,96% | 0,40% |

### 6.3 Validation cùng epoch 10, FW2/FW1 đã tiếp tục trên local

**Lý do tiếp tục đánh giá:** các lượt mới chưa đủ mục tiêu 15 epoch và đã chuyển sang local sau khi hết quota Kaggle1; cần theo dõi cùng mốc epoch và ghi rõ thay đổi môi trường.

**Kết quả thu được:** weight 2 có IoU 34,73%, vượt weight 4 là 31,66% tại mốc này, kèm precision cao hơn và báo giả real thấp hơn; recall và AUC vẫn thấp hơn. Weight 1 tăng IoU nhưng chưa đạt mức của hai weight còn lại.

| Weight | IoU fake | Dice fake | Precision pixel | Recall pixel | FP area real | AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 31,66% | 47,57% | 30,67% | 80,83% | 4,22% | 99,9750% |
| 2 | 34,73% | 51,15% | 42,30% | 59,46% | 1,80% | 99,6075% |
| 1 | 26,01% | 40,64% | 59,23% | 30,22% | 0,35% | 99,8850% |

FW2 validation epoch 11 mới nhất có IoU 31,24%, Dice 46,92%, AUC 99,7375%, FP area real 1,03%, MSE 0,0014184. FW1 chưa có mốc 11, nên không dùng metric này so với FW1 epoch 10. IoU FW2 tăng giảm theo epoch; không chọn weight 2 chỉ vì thắng ở epoch 10.

Nhận định: FW4 thiên về recall, FW1 thiên về precision nhưng bỏ sót nhiều GT; FW2 là ứng viên cân bằng, chưa là weight tối ưu. Các lượt mới chưa đủ 15 epoch/nhiều seed. Sau khi chuyển local: T4/torch 2.11→RTX 4050/torch 2.7; giữ source/config/optimizer/data nhưng không bảo đảm bitwise tương đương. Đây là giới hạn của đối chứng vớiFW 4 train hoàn toàn Kaggle.

### 6.4 Trạng thái tại thời điểm chốt báo cáo

**Lý do chuyển local/Kaggle2:** tiếp tục ngân sách train đã định trước khi Kaggle1 hết quota hoặc local gián đoạn. Đây là thay đổi nơi thực thi, không phải thử nghiệm thay kiến trúc/loss.

**Kết quả thực thi đã có:** checkpoint của hai lượt đã được giữ để resume; chưa có kết quả cuối 15 epoch cho FW2/FW1 và chưa có kết quả train Kaggle2 tại mốc chốt dưới đây.

- FW4: hoàn thành 15 epoch và test như mục 5.3.
- FW2: checkpoint local đã lưu bước 14.600; log có bước 14.699 nhưng không coi 99 updates sau checkpoint đã lưu. Có đánh giá cuối cùng epoch 11.
- FW1: checkpoint local đã lưu 12.700; log sau đó 12.743 chưa lưu. Có đánh giá cuối cùng epoch 10.
- Local không còn tiến trình train tại lần kiểm tra trước khi chuyển nền tảng; không kết luận nguyên nhân gián đoạn từ riêng log.
- Người dùng đã xác nhận tài khoản Kaggle 2 v5nhdf và chấp thuận gói 4,36 GB riêng tư. Pipeline đang UPLOADING, sẽ kiểm dataset ready/privacy/quota rồi gửi hai notebook resume. **Chưa có kết quả train Kaggle 2 trong báo cáo này**, không tính việc đổi nền tảng thành một ablation mới.

Nguồn: [đối chứngweight](language_mask_weights_20261029_vi.md), [chuyểnKaggle2](language_mask_weights_kaggle2_20261030_vi.md), [trạng tháipipeline](../artifacts/language_mask_weights_kaggle2_20261030/pipeline_status.json).

## 7. Những gì đã triển khai, đã thử và chưa thử

| Hạng mục | Trạng thái và giới hạn |
|---|---|
| Resize-conv, skip concat, detach AD→RE | Đã triển khai/đối chứng; không bảo đảm loại hết artifact hoặc cạnh tranh loss |
| Sigmoid AD | Đã thử; các checkpoint lớn hiện dùng tuyến tính |
| Weighted CE, quét lambda FD và foreground weight | Đã thử; FW2/FW1 còn dở |
| EOS pooling, bỏ PAD trong LR, L2 CMC | Đã triển khai, thử gộp; chưa có ablation riêng từng yếu tố ở quy mô 5.000 ảnh |
| VIM double/pre-norm | Đã kiểm số học và train language; chưa xác nhận trùng recipe tác giả |
| Gradient norm/cosine | Đã đo nhiều phép nhỏ; chưa theo dõi đủ xuyên suốt mọi epoch/mọi objective |
| Dice/boundary/Lovasz loss, PCGrad | Mới đề xuất, chưa có kết quả train để báo cáo |
| Augmentation/face alignment theo DeepfakeBench | Đã đọc nguồn và đối chiếu; chưa áp dụng vào các lượt đang so |
| Threshold mask/gate theo detection | Chưa dùng để cải thiện các bảng metric hiện tại |
| Cross-generator/identity-disjoint | Chưa xác nhận/đánh giá đầy đủ |

Tham khảo [DeepfakeBench](tham_khao_deepfakebench_tien_xu_ly_vi.md) là nghiên cứu mã nguồn, không phải thí nghiệm đã cho kết quả IoU. Các smoke/unit tests kiểm đúng luồng/hash/shape/gradient/determinism không thay thế validation/test.

## 8. Kết luận và ưu tiên tiếp theo

**Kết luận có căn cứ:** model có khả năng học mask; chất lượng AD/detection đã cải thiện qua nhiều thay đổi; các objective có tương tác tùy batch/trạng thái; giảm FD về 0,1 là recipe đối chứng hữu ích; tăng dữ liệu/ngân sách đem lại cải thiện nhưng vẫn còn FP/FN và khoảng cách train/val. Language có lợi bước đầu ở một seed/cùng mốc 15 epoch, kèm đánh đổi báo giả real.

**Chưa kết luận được:** một nguyên nhân duy nhất khiến IoU thấp; weight pixel giả tối ưu; việc tách decoder chắc chắn có ích; sigmoid hoặc thêm language luôn tốt hơn; normalization/decoder hiện tại đã tái hiện đầy đủ paper. AD tái tạo sát input không tự chứng minh residual mất mọi dấu vết giả, và MSE thấp không đảm bảo IoU cao.

Ưu tiên theo thứ tự:

1. Hoàn thành FW2/FW1 đến 15 epoch, báo cáo cùng mốc; giữ test cho xác nhận sau khi chọn trên validation.
2. Xác nhận ứng viên bằng nhiều seed và cùng môi trường/runner trước khi khẳng định cải thiện ổn định. Báo cáo cả IoU, precision, recall, FP area real, AUC và MSE.
3. Phân tích lỗi theo ảnh, kích thước GT và biên mask; phân biệt tô dư nền với bỏ sót chi tiết mảnh. Không gate mask để che lỗi khi chẩn đoán.
4. Thử augmentation nhẹ trong lượt mới, bắt đầu flip ngang đồng bộ ảnh–GT; giữ validation/test cố định. Không thay giữa chừng đối chứng weight.
5. Nếu vẫn hạn chế, đối chứng CE+Dice hoặc tách trunk AD/MD từng phương án riêng. Đây là kế hoạch, chưa có kết quả.

## 9. Tài liệu và artifact để kiểm chứng

- Kiến trúc/language pilot: [genface_ablation_20261026_vi.md](genface_ablation_20261026_vi.md), [genface_sigmoid_20261026_vi.md](genface_sigmoid_20261026_vi.md), [eos_pad_l2_va_gradient_ar_fl_vi.md](eos_pad_l2_va_gradient_ar_fl_vi.md).
- Loss/FD: [genface_loss_overfit_local_vi.md](genface_loss_overfit_local_vi.md), [genface_fd_weight_kaggle_vi.md](genface_fd_weight_kaggle_vi.md), [genface_fd_validation_results_vi.md](genface_fd_validation_results_vi.md), [genface_fd_epoch15_vi.md](genface_fd_epoch15_vi.md).
- Quy mô lớn: [genface5000_epoch30_vi.md](genface5000_epoch30_vi.md), [genface5000_test_vi.md](genface5000_test_vi.md), [language_readiness_20261028_vi.md](language_readiness_20261028_vi.md), [language5000_scratch_kaggle_vi.md](language5000_scratch_kaggle_vi.md).
- Kết quả FW mới: `artifacts/language_mask_weights_kaggle_20261029/weight_<1 hoặc2>_stage_2/downloaded/language5000_results/report.json`; local `weight_2/segment_002`, `segment_003`, `segment_004` và `weight_1/segment_001` dưới `artifacts/language_mask_weights_local_20261029/run_001/`.

Các bảng là số làm tròn từ report/tài liệu thực nghiệm. Không sửa checkpoint, không khởi chạy ablation mới, không commit/push trong bước viết báo cáo. Trạng thái job có thể thay đổi sau mốc chốt; xem pipeline/report mới để cập nhật.
