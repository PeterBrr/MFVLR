# Phân tích bổ sung 30/10/2026: residual VIM double, thông tin nhãn trong prompt và trạng thái 5 vấn đề

Phạm vi: đọc các tài liệu và code mới đến commit `e2e1b57` (thử nghiệm GenFace 26–30/10), đối chiếu với [báo cáo 5 vấn đề](bao_cao_5_van_de_decoder_loss_gradient_ngon_ngu.md), và chạy probe tại khởi tạo trên CPU. Máy phân tích không có checkpoint đã train hay ảnh GenFace (`data/` trống; các `report.json` trong `artifacts/` không nằm trong git), nên mọi số đo mới dưới đây là **probe tại khởi tạo**, không phải đo trên `final.pt`. Các số train/validation/test được dẫn lại từ tài liệu thực nghiệm, không được kiểm lại từ JSON gốc.

## 1. Kết quả chính

- **`double` + `pre_norm` làm độ sâu hiệu dụng của LE/LD suy giảm theo 2^(−j).** Đây là cấu hình của lượt language 5.000 ảnh/15 epoch. Bỏ hẳn block LE12 chỉ làm `T_l` đổi 0,04%, trong khi bỏ LE1 làm `T_l` đổi 105%. Gradient LR tại LE12 nhỏ hơn LE1 khoảng 2.057 lần. Với `single` + `pre_norm`, mười hai block LE đóng góp gần như bằng nhau.
- **Với hai prompt real/Diffae, LR chỉ cần thông tin nhãn ở đúng 1 vị trí.** Trong 32 (real) và 44 (Diffae) token non-PAD, chỉ vị trí 5 (`real`/`fake`) không suy ra được từ tiền tố causal, và chính token đó có sẵn trong LE memory cùng vị trí. Token accuracy 99,74% sau epoch 1 và LR = 0,0065 ở epoch 15 khớp với việc LR gần như không cần đến ảnh.
- **49,26% microbatch train chứa hai prompt giống hệt nhau**, nhưng CMC vẫn coi chúng là negative của nhau. CMC validation (0,00094) chỉ đo trên cặp thật–giả, vì validation được xếp xen kẽ nhãn, nên không phản ánh trường hợp này.
- Vì vậy lợi ích IoU +1,99 điểm của lượt language chưa thể quy cho “ngữ nghĩa ngôn ngữ”. Với hai prompt cố định theo nhãn, nhánh language về bản chất cung cấp thêm một tín hiệu giám sát nhãn qua KL/CMC. Cần một control thay text bằng hai vector nhãn học được (mục 5).

## 2. Residual VIM double dưới pre_norm

### 2.1 Cơ chế

Trong [pilot_language.py](../src/mfvlr_research/pilot_language.py), dòng 51–96, mỗi block `pre_norm` tính `factor*x + Delta(LN(x))` với `factor = 2` khi `vim_residual='double'`. Attention/FF trước đó cũng nhận `LN(x)`. Do đó mọi sublayer trả output cỡ O(1) bất kể độ lớn của stream, còn stream nhân đôi sau mỗi VIM: 12 lần trong LE (×4.096) và 7 lần trong LD (×128). Final LN ở cuối LE/LD loại bỏ thang chung. Kết quả là đóng góp tương đối của block j bị nhân với khoảng 2^(−j):

`2x + Delta = 2·(x + Delta/2)`, tức tương đương residual single với nhánh update của các block phía sau bị giảm theo cấp số nhân.

Nếu không có norm, cùng phép nhân đôi này đi thẳng vào các sublayer; đó là lý do [tài liệu readiness](language_readiness_20261028_vi.md) đo được RMS logit khoảng 1,57×10⁸ ở `double`/baseline. `pre_norm` chặn được bùng nổ ở đầu ra, nhưng thay vào đó làm các block sau gần như không còn tác dụng.

### 2.2 Số đo tại khởi tạo

Điều kiện: `EosFineGrainedLanguageTransformer(seed=0, language_normalization='pre_norm')`, token của hai prompt real/Diffae, visual token ngẫu nhiên có RMS 9,92 (bằng RMS visual mà tài liệu readiness ghi cho feature đã train), seed 0, CPU FP32. “Bỏ block” nghĩa là thay output của block bằng `factor × input`, tức giữ thang residual và chỉ bỏ các update của block đó.

| Block | RMS stream single | RMS stream double | Grad LR single | Grad LR double | ΔT_l khi bỏ block, single | ΔT_l khi bỏ block, double |
|---|---:|---:|---:|---:|---:|---:|
| LE1 | 10,10 | 10,14 | 3,52 | 20,71 | 30,6% | 104,8% |
| LE3 | 17,03 | 45,75 | 3,37 | 5,00 | 28,1% | 21,1% |
| LE6 | 24,11 | 368,28 | 3,53 | 0,64 | 31,0% | 2,94% |
| LE9 | 30,12 | 2.944,96 | 3,47 | 0,081 | 31,0% | 0,36% |
| LE12 | 34,65 | 23.558,41 | 3,34 | 0,010 | 30,2% | 0,04% |

| Block | Grad LR single | Grad LR double | ΔLogits khi bỏ block, single | ΔLogits khi bỏ block, double |
|---|---:|---:|---:|---:|
| LD1 | 43,41 | 97,53 | 39,0% | 96,5% |
| LD4 | 41,69 | 11,77 | 36,7% | 10,1% |
| LD7 | 41,41 | 1,47 | 37,4% | 1,3% |

RMS stream ở `double` (LE12 khoảng 23.558, LD7 khoảng 806) khớp với số tài liệu readiness ghi khi dùng visual feature của checkpoint 5.000/30 epoch và language khởi tạo mới (khoảng 23.981 và 742). Như vậy probe với visual ngẫu nhiên tái hiện đúng thang đã quan sát tại khởi tạo language.

### 2.3 Ý nghĩa và giới hạn

- Adam chuẩn hóa độ lớn gradient theo từng tham số, nên tham số của block sau vẫn được cập nhật khoảng `lr` mỗi bước. Gradient nhỏ tự nó không làm block đứng yên. Nhưng tác động của các cập nhật đó lên output vẫn bị nhân 2^(−j). Để LE12 có ảnh hưởng ngang LE1, output sublayer phải lớn lên khoảng 2.000 lần. Với language LR 1e-5 trong 18.750 bước, mỗi trọng số thay đổi tối đa khoảng 0,19, trong khi biên Xavier ban đầu khoảng 0,077. Đây là cận trên thô, nhưng cho thấy khó bù được hệ số hàng nghìn.
- Hệ quả là lượt language 5.000 ảnh nhiều khả năng đã train với LE hiệu dụng khoảng 3–4 block và LD khoảng 3 block. VIM ở các block đầu là nơi gần như duy nhất đưa ảnh vào text embedding.
- Đây là probe tại khởi tạo. Cần đo lại cùng phép “bỏ block” trên `final.pt` để biết trạng thái sau train; tool ở mục 7 có thể mở rộng để nạp checkpoint.
- `2x+Delta` là cách diễn giải người dùng đã chốt; probe này không xác định được tác giả dùng cách nào. Nó chỉ cho thấy: dưới `pre_norm`, chọn `double` tương đương chọn trọng số giảm theo cấp số nhân cho các block sau. Đó là một quyết định kiến trúc lớn, không chỉ là chi tiết cộng residual.

## 3. Hai prompt cố định theo nhãn

[run_language5000.py](../tools/run_language5000.py), dòng 64–66, khẳng định `tokens.shape == (2, 308)` và gán `token_ids = tokens[label]`. Toàn bộ supervision ngôn ngữ vì vậy là một hàm của nhãn nhị phân.

```text
real:   "A photo of a real face" × 4
Diffae: "A photo of a fake face" / "A photo of an attribute manipulated face" /
        "A photo generated by the diffusion-based model" /
        "The source generative model of this photo is Diffae"
```

Input LD tại vị trí t là token t−1; target là token t. Hai prompt có cùng tiền tố đến hết `A photo of a`, nên vị trí 5 là vị trí duy nhất mà target khác nhau trong khi input causal giống nhau. Sau vị trí đó, label đã nằm trong chính tiền tố. Ở vị trí 5, đáp án có sẵn trong LE memory cùng vị trí, vì memory attention không mask (vấn đề 5 của báo cáo 5 vấn đề).

Hệ quả:

- LR có thể đạt gần 100% token accuracy mà không cần ảnh; số liệu thực nghiệm khớp điều này (99,74% sau epoch 1, 100% ở epoch 15, LR = 0,0065). Cộng với hệ số LR 1e-4, đóng góp của LR lên nhánh ảnh sau giai đoạn đầu gần như bằng 0.
- Tác động của language lên visual vì vậy đến chủ yếu từ KL và CMC (hệ số 0,01), với text embedding được quyết định bởi nhãn cộng phần phụ thuộc ảnh qua VIM. Ở epoch 15, KL = 0,00039 và CMC = 0,00094 trên validation, nên gradient language ở cuối quá trình train nhiều khả năng đã nhỏ; tác động, nếu có, tập trung ở các epoch đầu. Cần xác nhận bằng `history` của report train.

## 4. CMC với prompt trùng trong microbatch

Train có nhãn xen kẽ `[0,1,…]` và lịch batch là hoán vị `np.random.default_rng(0)`. Tái lập đúng lịch đó:

| Đại lượng | Giá trị |
|---|---:|
| Microbatch có hai prompt giống hệt nhau | 49,26% |
| Bước mà cả hai microbatch đều có prompt trùng | 24,36% |
| Cặp validation trong `language_eval` có prompt trùng | 0% (luôn là thật–giả) |

CMC dùng positive trên đường chéo. Với hai ảnh cùng nhãn, CMC đẩy `cos(I_v^1, T_l^2)` xuống dù hai prompt giống hệt nhau. Cách duy nhất để giảm loss là làm `T_l` mang thông tin riêng của từng ảnh qua VIM. Khi đó CMC trở thành bài toán ghép ảnh với chính nó qua nhánh text, không còn là alignment ảnh–ngôn ngữ. CMC validation không đo được hiện tượng này vì cặp validation luôn khác nhãn.

Kiểm tra cụ thể: tách `history[].losses.cmc` của report language 5.000 theo loại bước. Danh sách bước có cả hai microbatch trùng prompt tái lập được từ lịch batch, và tool ở mục 7 đã tính tỉ lệ. Nếu CMC ở các bước trùng prompt cũng gần 0, `T_l` đã trở nên đặc trưng theo từng ảnh.

## 5. Hệ quả cho việc đánh giá lượt language 5.000

So sánh hiện có (validation epoch 15: IoU 33,57% → 35,56%; MSE −26,1%; FP area real 0,81% → 1,48%) có bốn giới hạn chồng lên nhau: một seed; baseline image-only là lượt lịch sử từ runner khác; language coupled thay đổi gradient vào encoder; và nhánh language về bản chất là giám sát nhãn. Thứ tự control đề xuất, mỗi control chỉ đổi một yếu tố:

1. `--image-only` cùng runner, cùng seed 0. Runner đã hỗ trợ; đây là đối chứng rẻ nhất và tách được ảnh hưởng của runner.
2. **Control prototype nhãn:** thay `T_l` bằng một trong hai vector học được theo nhãn, giữ KL/CMC, hệ số và optimizer. Nếu kết quả tương đương lượt language, phần lợi ích đến từ giám sát nhãn bổ sung chứ không từ LE/LD.
3. `single` so với `double` dưới `pre_norm`, giữ mọi thứ khác. Kết hợp đo “bỏ block” trên checkpoint cuối.
4. Nhiều seed cho ứng viên tốt nhất trước khi kết luận.

Kết quả lượt foreground weight 4/2/1 không bị ảnh hưởng trực tiếp bởi các điểm trên, nhưng cũng dùng cùng cấu hình language. Mọi chênh lệch giữa các weight đều được đo trên một nhánh language có độ sâu hiệu dụng thấp và supervision trùng nhãn.

## 6. Trạng thái 5 vấn đề sau các thử nghiệm 26–30/10

| Vấn đề | Đã xử lý trong code thử nghiệm | Còn lại hoặc mới phát sinh |
|---|---|---|
| 1. Vision Decoder | Skip concat, resize-conv, detach AD trước RE; MSE giảm mạnh (0,0837 → 0,0120 ở pilot 128 ảnh; khoảng 2,8e-4 ở 5.000 ảnh) | Cosine AR–FL vẫn âm (−0,15 đến −0,25). Với MSE khoảng 2,8e-4, RMS residual mỗi phần tử khoảng 0,017, nên RE gần như nhận một ảnh đen; đóng góp của `I_g^r` vào `I_v` chưa được đo. Đề xuất: thay `I_g^r` bằng trung bình của nó trên validation khi inference, đo lại AUC |
| 2. LD không LN | `pre_norm` + final LN cho LE/LD; logits RMS khoảng 0,455 | Kết hợp với `double` gây suy giảm độ sâu hiệu dụng (mục 2) |
| 3. Thang loss | Bỏ PAD khỏi LR; hệ số FD 0,1, LR 1e-4, CMC/KL 0,01; quét FD và foreground weight | Additivity FAIL lịch sử chưa được chạy lại trên recipe mới; LR không cần ảnh nên hệ số của nó gần như không còn ý nghĩa (mục 3) |
| 4. VIM singleton | Giữ nguyên có chủ đích | Q/K vẫn không nhận task gradient; dưới `double`, VIM các block sau còn bị giảm thêm 2^(−j) |
| 5. Nhánh ngôn ngữ | EOS cuối thực sự (thay slot 307 luôn là PAD); L2 cho CMC (thay raw dot bão hòa) | Đường tắt qua full memory; hai prompt cố định theo nhãn; 49% microbatch prompt trùng; CMC validation chỉ đo cặp khác nhãn |

## 7. Tái lập

```powershell
python tools/probe_language_depth_and_prompts.py --output artifacts/language_depth_probe_20261030
```

Tool không dùng ảnh, checkpoint hay GPU; ghi `report.json` gồm số liệu từng block (single/double), các vị trí cần nhãn trong prompt và tỉ lệ microbatch trùng prompt theo đúng lịch batch của `run_language5000.py`. Thư mục output phải chưa tồn tại. Probe chạy với torch 2.12.1+cpu, Python 3.14.3; tokenizer dùng dependency trong `.phase2c1_deps` hoặc `.pilot_deps`.

Những gì cần checkpoint/dữ liệu và chưa làm trong bước này: đo “bỏ block” và ablation `I_g^r` trên `final.pt`; tách CMC theo loại bước từ `history`; các control ở mục 5. Dataset GenFace trên Kaggle (`v5nhdf/genface-dataset`) chỉ chứa ảnh gốc, không có checkpoint, nên chưa được tải về cho các probe tại khởi tạo này.
