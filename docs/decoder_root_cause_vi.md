# Điều tra nguyên nhân mask lặp — 18/10/2026

Checkpoint tham chiếu: `pilot_005/best.pt`, step 2000, SHA-256
`187f27253849c7bae9de501f2fb2929a2da3561614311a3e501b95b2bc5b202e`.

Đã chạy đối chứng cấu trúc, đo gradient sáu objective trên hai cặp real/fake
thuộc train và hai thử nghiệm 100 bước trên bản sao decoder. Không sửa checkpoint,
model/loss/config train chính hoặc dùng test để chọn hyperparameter. Artifact ở
`artifacts/decoder_root_cause_pilot005/`. Kiểm tra hash trước/sau xác nhận checkpoint
gốc không đổi trong mỗi thử nghiệm.

## Kết luận có bằng chứng

**Mẫu lặp xuất phát từ các pha không gian trong decoder và bị quá trình tối ưu
hiện tại duy trì/khuếch đại. Tín hiệu localization yếu hơn rất nhiều gradient
ngôn ngữ tại trunk dùng chung; loss pixel không cân bằng còn cho phép nghiệm
gần toàn nền.** Các phép thử chưa chứng minh một yếu tố duy nhất giải thích mọi
lần train, nhưng đã xác định được các cơ chế cụ thể thay cho suy đoán từ ảnh.

### 1. Trọng số bốn pha upsampling tạo mẫu lưới

Mỗi `ConvTranspose2d(kernel=2,stride=2)` có bốn bộ trọng số cho các vị trí chẵn/lẻ.
Kernel bằng stride không buộc các bộ trọng số này bằng nhau. Bốn tầng ×2 tạo
các pha ở chu kỳ 2/4/8/16 pixel.

Đối chứng trên bốn ảnh train cố định: tạm buộc bốn pha của **cả bốn tầng** dùng
trọng số trung bình của chúng, đồng thời giữ Frobenius norm mỗi weight tensor.
Sau phép thử, khôi phục trọng số và kiểm tra đầu ra logits bằng chính xác baseline.

| Đối chứng | Phần biến thiên không gian của xác suất trung bình giải thích bởi chu kỳ 16 |
|---|---:|
| Decoder checkpoint gốc | 99.48% |
| Buộc cả bốn tầng có trọng số đồng nhất giữa các pha | 0.64% |
| Encoder khởi tạo + decoder đã train | 99.41% |

Giữ decoder đã train và thay encoder bằng encoder mới vẫn giữ pattern mạnh:
nguồn tạo pattern nằm rõ trong decoder. Từng tầng riêng lẻ chưa đủ loại mẫu,
vì các tầng còn lại vẫn tạo pha khác nhau.

Tuy nhiên, phép buộc pha làm output gần toàn nền, fake IoU=0; đó là phép chẩn đoán
nguồn artifact, **không phải cách sửa checkpoint để dùng ngay**. Decoder mới khởi
tạo cũng có thành phần tuần hoàn, nhưng biên độ xác suất nhỏ hơn nhiều (spatial
std khoảng 0.015–0.020 so với 0.270 ở decoder đã train). Không nên dùng duy nhất
tỷ lệ variance để kết luận chất lượng hoặc độ mạnh artifact.

### 2. Gradient language lấn át localization tại trunk dùng chung

Giữ nguyên sáu objective, đo `autograd.grad` riêng từng loss tới ba tensor
decoder, không optimizer update. Hai cặp train có hai ảnh fake khác thuộc tính:
Bangs và smile.

| Tensor | Cặp 1: norm LR / norm FL | Cặp 2: norm LR / norm FL | Cosine gradient tổng với FL, cặp 1 / cặp 2 |
|---|---:|---:|---:|
| U1 upsample weight | 6,668× | 9,190× | -0.648 / -0.564 |
| U4 conv2 weight | 2,734× | 2,760× | -0.061 / -0.102 |
| Mask head weight | LR không có đường gradient | LR không có đường gradient | 1.0 / 1.0 |

Đây là gradient thô trên hai minibatch, chưa tính preconditioning/moment/weight
decay của Adam. Cosine âm chứng tỏ xung đột cục bộ trong không gian gradient,
không tương đương chứng minh từng bước Adam thực tế tăng FL. G01 lịch sử không
được xét lại hay sửa ngưỡng ở đây; không đo additivity so với total backward.

Đường ảnh hưởng tồn tại trong code:

`language loss → visual-conditioned language → fused visual → residual encoder
→ |reconstruction - input| → appearance head → shared decoder trunk`.

Vì trunk được chia sẻ giữa appearance/mask, loss language có thể điều khiển
feature dùng cho mask dù mask head chỉ nhận FL. Giải thích trước đây dựa riêng
trên AR lớn hơn FL là chưa đầy đủ: gradient LR còn lớn hơn AR rất nhiều ở hai
tensor này. Mask head không bị ngắt gradient; vấn đề là feature nền mà head phải
phân loại đang bị các objective khác chi phối.

### 3. Nghiệm nền được ưu tiên khi chỉ dùng CE pixel không trọng số

Hai thử nghiệm dùng cùng bốn ảnh train, cùng encoder features đã freeze, cùng
decoder checkpoint làm điểm xuất phát, Adam LR=1e-4 và 100 steps. Chỉ bản sao
decoder được cập nhật; không tạo checkpoint dùng để benchmark.

Trên bốn ảnh, foreground chiếm 6.03% tổng pixel (hai real có mask rỗng; hai fake
lần lượt khoảng 13.06% và 11.07%).

| Thử nghiệm | Kết quả sau 100 steps trên chính bốn ảnh |
|---|---|
| FL nguyên bản (CE pixel không trọng số) | Loss 0.575 → 0.154, fake IoU 0; mọi xác suất foreground <0.5 |
| CE cân bằng pixel, weight [1,15.575] | Fake IoU khoảng 0.180; tỷ lệ variance chu kỳ 16 từ 99.48% xuống 3.18%; mask không còn toàn nền |

CE cân bằng vẫn có false-positive area real khoảng 47.35%, chưa học tốt và chưa
hội tụ. Chỉ chứng minh thay objective có thể thay đổi nghiệm và làm yếu lưới
ngay trên **cùng kiến trúc**. Hai loss khác nhau về chuẩn hóa nên không so sánh
trực tiếp trị số loss giữa hai hàng. Việc học trên bốn ảnh không là kết quả
validation/test và không xác nhận generalization.

## Hướng xử lý hợp lý

1. Giữ A0/checkpoint thất bại làm baseline. Không tiếp tục tăng epoch với cùng
   recipe và kỳ vọng mẫu lặp tự biến mất.
2. Với nhánh thí nghiệm riêng, trước hết xác nhận overfit nhỏ cho segmentation,
   đo chất lượng mask/target và khả năng phân biệt từng ảnh; sau đó mới đưa các
   objective khác trở lại và theo dõi norm/cosine gradient tại trunk.
3. Kiểm tra scale/reduction loss language và cân bằng đa mục tiêu có kiểm soát.
   Đổi hệ số/reduction là thay recipe so với baseline, cần ghi rõ; không tự sửa
   `SixObjectives` hoặc tuyên bố đó là paper recipe.
4. So sánh resize+conv với transposed-convolution như một ablation độc lập.
   Không bắt buộc đổi decoder trước mọi bước khác: cùng decoder đã giảm pattern
   trong đối chứng CE cân bằng, nên cần phân biệt thiết kế upsampler và tối ưu hóa.

Các report: `structure.json`, `gradients.json`, `gradients_pair1.json`,
`fit.json`, `fit_balanced_FL.json`. Công cụ tái lập:
`tools/decoder_root_cause.py` với các mode `structure`, `gradients`, `fit`.
