**BÁO CÁO: 5 VẤN ĐỀ QUAN TRỌNG NHẤT VỀ DECODER, LOSS/GRADIENT VÀ NHÁNH NGÔN NGỮ CỦA MFVLR**

Ngày rà soát: **25/10/2026, Asia/Saigon**. Source tại commit `fe2358fde3682483dc8af5173a6796ebaa35e79a`; working tree sạch trước khi thêm báo cáo.

Cập nhật: **30/10/2026**. Đã đối chiếu lại toàn bộ nhận định với source/paper/artifact và chạy các probe chẩn đoán ở trạng thái khởi tạo; các đoạn **Kiểm chứng bổ sung (30/10/2026)** ghi kết quả và hiệu chỉnh tương ứng. Không có nhận định nào của bản 25/10 bị phát hiện sai; các bổ sung chủ yếu làm rõ nguyên nhân và liên kết giữa các vấn đề.

**Nhận định chính:** source hiện tại triển khai một baseline kỹ thuật A0 có hợp đồng khá rõ, nhưng chưa đủ cơ sở gọi là khôi phục chính xác kiến trúc tác giả hoặc mô hình đã sẵn sàng huấn luyện. Năm vấn đề dưới đây ảnh hưởng trực tiếp đến thông tin decoder nhận được, độ lớn gradient, cơ chế tương tác ảnh–văn bản và ý nghĩa của supervision ngôn ngữ.

Nguồn chính là [paper MFVLR](../papers/MFVLR.pdf), bản 18 trang, SHA-256 `2afed0516c32121513463d326720153887586ceaa328f26b96263bd39da45307`. Đã trích xuất văn bản và xem trực tiếp trang 5–8, 14–15 để kiểm tra hình, công thức và bảng liên quan. Các vị trí code dưới đây trỏ đến source đang có; số dòng tính tại thời điểm rà soát.

Quy ước: **Paper** là điều văn bản/hình/công thức nêu trực tiếp; **code** là hành vi đọc được từ implementation; **suy luận** là hệ quả toán học hoặc nhận định cần kiểm chứng; **số liệu lịch sử** là kết quả trong artifact có sẵn, không phải thí nghiệm mới của lần báo cáo này; **probe 30/10** là số đo mới, CPU FP32, trọng số khởi tạo seed 0, input tổng hợp, không optimizer update, chỉ có giá trị chẩn đoán tại khởi tạo. “Lý do chọn” gồm cả lý do đưa vấn đề vào top 5 và lý do baseline chọn cách triển khai đó.

| Vấn đề | Tác động chính | Độ chắc chắn của nhận định |
|---|---|---|
| 1. Vision Decoder chỉ dùng bottleneck, dùng chung trunk và truyền gradient qua residual ảnh | Chi tiết localization và bản chất tín hiệu residual | Cấu trúc code rõ; ưu thế so với skip decoder chưa đo |
| 2. Language Decoder không chuẩn hóa qua 7 block | Khuếch đại activation, thang logits LR và độ nhạy gradient | Có số liệu activation lớn; probe 30/10 cho thấy thiếu norm trước vocab projection giải thích phần lớn thang LR; chưa chứng minh là nguyên nhân duy nhất của additivity FAIL |
| 3. Sáu loss có trọng số 1 nhưng khác thang đo; kiểm tra cộng gradient chưa đạt | Hướng tối ưu và độ tin cậy số học | Có artifact định lượng; LR và CMC phồng do thang activation/norm; chưa có kết quả training |
| 4. VIM một visual token làm attention suy biến | Q/K không học qua task loss; khác ý nghĩa tương tác theo từng từ | Hệ quả toán học chắc chắn với softmax theo key đang dùng |
| 5. Nhánh ngôn ngữ tái tạo prompt có điều kiện và tự căn chỉnh với ảnh | Nguy cơ học đường tắt, chất lượng embedding và contrastive supervision | Đường thông tin rõ; mức độ đường tắt khi training chưa đo |

Trạng thái hiện hành được lấy từ [REPOSITORY_STATUS](../REPOSITORY_STATUS.md) và [D02B11](phase2d2_d02b11_authorization_checklist.md), không lấy mô tả giai đoạn sớm trong README làm trạng thái hiện tại. Phase 2D.2 vẫn **BLOCKED**. Lần rà soát 25/10 chỉ đọc source, paper và artifact; không chạy forward/backward toàn mô hình, optimizer hay benchmark, không thay đổi baseline. Lần cập nhật 30/10 có chạy forward/backward chẩn đoán tại khởi tạo (mô tả trong từng mục), vẫn không có optimizer, training hay benchmark, và không thay đổi baseline.

---

**1. Vision Decoder: chỉ dùng bottleneck, trong khi chất lượng residual và mask cùng phụ thuộc một trunk**

**Hiện trạng.** AD (tái tạo ảnh) và MD (dự đoán mask) dùng một decoder trunk, thực thi một lần rồi tách hai head. Decoder chỉ nhận feature `[B,1024,14,14]`, không sử dụng bốn skip feature mà encoder đã tạo. Ảnh tái tạo tiếp tục đi vào residual encoder, vì vậy AD vừa chịu loss tái tạo vừa nằm trên đường gradient từ các mục tiêu phía sau.

**Paper mô tả.** Trang 5, Hình 4 và trang 6, §III-C: AD/MD là U-Net decoder với convolution đầu ra riêng; hai decoder dùng chung kiến trúc và trọng số. Paper viết `I_pre = AD(I_loc)`, `M_pre = MD(I_loc)`. Trang 5–6 mô tả `I_r = |I_pre − I|`, IE/RE chia sẻ trọng số và `I_v = I_g + I_g^r`. Paper không liệt kê chính xác skip connections, nội bộ upsampling, activation đầu ra hay chính sách detach. Tên “U-Net” và công thức đầu vào một tensor chưa đủ để khẳng định tác giả có hoặc không dùng skips.

**Code hiện tại.** [vision_decoder.py](../src/mfvlr/vision_decoder.py), dòng 28, 42, 62, 90 và 110:

- Bốn stage `ConvTranspose2d(k=2,stride=2)` và hai `Conv2d(3×3)+ReLU` mỗi stage; kênh `1024→512→256→128→64`, không gian `14→28→56→112→224`.
- `skip_policy='bottleneck_only'`; truyền `skips` vào sẽ bị từ chối. `decode_appearance()` chỉ đọc `appearance.local`.
- Head `1×1` riêng: `64→3` cho reconstruction và `64→2` cho mask logits. Không sigmoid/clamp cho reconstruction; đầu ra có thể ngoài `[0,1]`.
- [mve.py](../src/mfvlr/mve.py), dòng 260–279, dùng `torch.abs(reconstruction-image)` rồi gọi lại `shared_encoder`, không detach. [integration.py](../src/mfvlr/integration.py), dòng 54, nối toàn bộ chuỗi này.

Sơ đồ phụ thuộc thực tế:

```text
Ảnh → IE → I_loc → trunk VD → head AD → I_pre → abs(I_pre−I) → RE
              │       └────→ head MD → mask logits              │
              └→ visual transformer → I_g ──────────────────────┴→ I_v
```

IE và RE là hai lần gọi cùng encoder; AD và MD dùng cùng feature trunk. Đây là graph tuần tự, không phải vòng lặp tái tạo lặp vô hạn. Tại `I_pre=I`, phép abs dùng gradient 0 theo quy ước autograd; ở vị trí khác, dấu residual tham gia gradient.

**Lý do chọn.** Đây là quyết định kiến trúc ảnh tác động đồng thời đến detection, localization và tất cả loss dùng `I_v`. [Hợp đồng Phase 2B](phase2b_decoder_contract.md) chọn bottleneck-only để bám giao diện `AD(I_loc)/MD(I_loc)` được công bố và hạn chế thêm đầu vào chưa được mô tả. Đây là lựa chọn kỹ thuật có chủ đích, không phải bằng chứng rằng nó trung thành với tác giả hơn mọi biến thể có skips.

**Ưu điểm.** Một trunk giữ đúng chia sẻ AD/MD và tránh nhân đôi tham số; bottleneck làm giao diện đơn giản, buộc tái tạo qua feature nén. Decoder hiện có **9.055.365** tham số. Không detach cho phép tín hiệu detection/ngôn ngữ tác động đến cách hình thành residual. Head tuyến tính giữ gradient ngoài miền `[0,1]`, tránh saturation do sigmoid.

**Nhược điểm.** Bỏ skip có thể làm mất biên và texture nhỏ; residual có thể chứa lỗi tái tạo thông thường thay vì dấu vết giả mạo. Ngược lại, tăng khả năng tái tạo cũng có thể làm residual giả mạo yếu đi vì AR áp dụng cả ảnh thật và giả. Hai head và các mục tiêu phía sau cùng tác động vào trunk nên có khả năng xung đột gradient. Reconstruction không bị chặn miền giá trị cũng có thể làm residual lớn. Các nguy cơ này là suy luận, chưa phải kết quả so sánh có/không skips. MVE vẫn giữ các skip feature trong record, nên “decoder không dùng skips” không đồng nghĩa đã loại hết bộ nhớ lưu chúng.

**Kiểm chứng bổ sung (30/10/2026).** Số dòng và số tham số đã được kiểm lại: tính độc lập từ topology cho đúng **9.055.365** (bottleneck) và **12.188.805** (Option A nối skip), khớp [hợp đồng Phase 2B](phase2b_decoder_contract.md); 25/25 test trong `tests/decoder` pass. Paper trang 6 xác nhận RE “shares the same architecture and weights with IE”, và Eq. (17) áp dụng cho mọi mẫu trong batch, không phân biệt thật/giả. Hai bổ sung:

- *Residual tại khởi tạo gần như bằng chính ảnh.* Probe `MVE(seed=0)` + `VisionDecoder(seed=0)`, một ảnh `torch.rand` (seed 1): reconstruction nằm trong `[−0,473; 0,298]`, trung bình −0,016, 52,2% pixel ngoài `[0,1]`; residual `|I_pre−I|` trung bình 0,520, max 1,424, tức xấp xỉ `I`. Nguyên nhân là head AD có bias 0 và Xavier init nên `I_pre≈0`. Vì vậy ở đầu training `I_v ≈ IE(I) + RE(I)` với IE/RE cùng trọng số: nhánh residual mã hóa lại ảnh gốc và chỉ mang nghĩa “dấu vết giả mạo” sau khi AR đã học tái tạo. Rủi ro thực tế lúc khởi tạo là residual trùng thông tin với ảnh, không phải residual lớn bất thường.
- *Trunk nhận gradient từ cả sáu loss.* Backward riêng một đại lượng tính trên `I_v` cho gradient trunk có norm khoảng 106; head mask không nhận gradient (`None`), tức chỉ FL huấn luyện head mask. Trong [integration.py](../src/mfvlr/integration.py), dòng 97–99 và 111–116, `I_v` đi vào FLT/VIM (LR), adapter (KL), CMC và detector (FD). Do đó FD, KL, CMC và LR đều truyền ngược về trunk qua `I_v → RE → |I_pre−I| → head AD`, bên cạnh AR/FL trực tiếp. Theo Vấn đề 3, LR có norm gradient lớn hơn nhiều loss khác, nên LR có thể chi phối cả trunk của decoder ảnh.

**Đề xuất cải tiến.** Giữ A0 làm đối chứng; khảo sát riêng biến thể dùng appearance skips, giữ nguyên loss, heads, dữ liệu và seed. Theo thiết kế Option A đã có trong hợp đồng, VD sẽ tăng lên **12.188.805** tham số; đây là số của phương án thiết kế, không phải decoder production đã triển khai. Nếu thử gated skips, phải coi đó là biến thể khác. Đánh giá đồng thời reconstruction MSE, phân bố residual real/fake, AUC và mIoU/độ chính xác biên mask; chỉ MSE thấp chưa đủ để chọn decoder. Theo dõi norm và cosine gradient trên trunk cho AR–FL, AR–FD, AR–LR, AR–KL và AR–CMC để kiểm tra xung đột, thay vì giả định chia sẻ luôn có lợi. Ghi lại phân bố residual theo bước training để biết khi nào residual bắt đầu khác ảnh gốc; nếu muốn rút ngắn giai đoạn `I_r≈I`, khởi tạo bias head AD gần 0,5 hoặc warm-up AR là các biến thể riêng. Không thay sigmoid, skips, detach hay khởi tạo head trong cùng một phép so sánh.

---

**2. Language Decoder: 7 block không LayerNorm, với mức activation tăng mạnh đã được ghi nhận**

**Hiện trạng.** LD production là bảy block độc lập, mỗi block có bốn phép cộng residual và không có LayerNorm đầu vào, giữa block hoặc cuối stack. LE chỉ có LN trước self-attention ở mỗi block. Các biến thể nghiên cứu P3/Q3 **đã có source**, nhưng chưa có bằng chứng đủ để thay baseline.

**Paper mô tả.** Trang 6 mô tả rõ LN trước MHA của LE. Trang 7, Eqs. (13)–(16), LD có thứ tự masked self-attention → attention với LE memory → FF → VIM, cộng residual sau từng bước. Các phương trình LD không hiển thị LN. Tuy nhiên, Hình 4 cũng không vẽ LN của LE dù phần chữ có nêu; vì vậy sự vắng mặt trên hình/công thức không chứng minh tác giả cấm mọi normalization. Trang 8 đặt `E=12, D=7, d=512`; head count, FF width, initializer cụ thể chưa được xác định đầy đủ.

**Code hiện tại.** [language_decoder.py](../src/mfvlr/language_decoder.py), dòng 12–37:

```python
masked = state + self_attention(state, causal=True)
communicated = masked + memory_attention(masked, memory)
fed = communicated + ff(communicated)
out = fed + vim(fed, visual)
```

[language_encoder.py](../src/mfvlr/language_encoder.py), dòng 37, 88 và 100, định nghĩa attention 8 heads, FF `512→2048→512` với GELU, cùng LN trước LE attention. [flt.py](../src/mfvlr/flt.py), dòng 69, chiếu LD hidden qua vocabulary weight dùng chung, không final norm.

Artifact B=2 `real_efs`, seed 0, CPU FP32 ghi nhận RMS sau các block như sau; đây là giá trị đã lưu, không phải số đo mới:

| Tensor | RMS |
|---|---:|
| LE12 / memory | 44,56 |
| LD01 | 51,10 |
| LD03 | 172,15 |
| LD05 | 492,15 |
| LD07 | 1.252,64 |

Max absolute tại LD07 là **4.529,80**. Nguồn: trường `activations` trong [artifact total.json](../artifacts/phase2d2_n31/g2/final_real_efs/total.json), đã đọc trực tiếp. Các giá trị vẫn hữu hạn; đây là bằng chứng tăng thang activation trong fixture, chưa chứng minh training sẽ phân kỳ. Artifact đánh số block từ 0 (`LE11`, `LD0`, `LD2`, `LD4`, `LD6`); bảng trên đánh lại từ 1.

**Kiểm chứng bổ sung (30/10/2026).** Các số trên khớp artifact. Ba bổ sung:

- *LE cũng tăng thang dù có LN.* Cùng artifact ghi RMS LE từ 4,57 (block đầu) lên 44,56 (block 12), khoảng 10 lần. LN của LE chỉ chuẩn hóa input của self-attention; VIM, FF và đầu ra cuối LE nhận residual stream chưa chuẩn hóa ([language_encoder.py](../src/mfvlr/language_encoder.py), dòng 108–111). Vì vậy memory đưa vào LD đã có RMS 44,6; P3/Q3 chỉ sửa LD, không đụng đến LE.
- *Thiếu norm trước vocab projection là nguyên nhân chính của thang LR.* Probe `MFVLRIntegration()` (seed 0), B=2 với ảnh `torch.rand` (seed 0), một prompt thật và một prompt EFS/diffusion/DDPM: hidden ra khỏi LD có RMS 1.244,0; nhân với vocabulary dùng chung (khởi tạo std 0,02) cho logits std 561,3, max abs 2.648,9. NLL trung bình là **806,6 nats/token**, so với `ln(49408)=10,81` của phân phối đều; LR mỗi ảnh là 240.590 và 256.260. Áp một LayerNorm không học lên chính hidden đó trước phép chiếu (chỉ để chẩn đoán) cho logits std 0,45 và NLL **9,83/token**. Ước lượng `1.244×0,02×√512 ≈ 563` khớp std đo được.
- *P3 chưa kiểm soát thang logits.* Probe `FineGrainedLanguageTransformer(seed=0)` và `build_variant(..., 'P3'/'Q3')`, cùng token IDs như trên, `I_v` ngẫu nhiên có norm khoảng 96 (bằng norm đo được của `I_v` thật), không tính gradient:

| Biến thể | RMS đầu ra LD | Logits std | NLL/token |
|---|---:|---:|---:|
| A0 | 1.450,4 | 658,86 | 2.401,56 |
| P3 (pre-LN) | 123,9 | 56,37 | 179,94 |
| Q3 (post-LN) | 4,3 | 1,96 | 14,57 |

NLL của A0 ở đây cao hơn probe phía trên vì `I_v` là vector ngẫu nhiên, không phải `I_v` từ MVE; so sánh chỉ có ý nghĩa giữa ba dòng của bảng. P3 không có norm sau residual cuối nên residual stream vẫn lớn; Q3 kết thúc bằng LN nên logits gần thang đơn vị, chỉ cộng thêm VIM delta.

**Lý do chọn.** LD tạo logits cho loss LR và truyền gradient ngược về LE, vocabulary và visual branch. Độ nhạy của nó có thể ảnh hưởng cả mô hình. Baseline chọn không thêm LD LN vì paper không ghi rõ, đồng thời giữ đúng thứ tự các phép cộng được công bố. Lựa chọn này dễ truy vết nhưng không phải xác nhận kiến trúc tác giả.

**Ưu điểm.** Graph đơn giản, thứ tự operation nhất quán với Eqs. (13)–(16). API VIM trả delta để caller cộng đúng một lần, tránh cộng text residual hai lần do cách trình bày Hình 5 và Eq. (16) chưa rõ ranh giới module. Việc giữ A0 giúp phân biệt hiệu ứng của normalization khi làm ablation.

**Nhược điểm.** Nhiều residual cộng dồn cùng memory/visual injection có thể khuếch đại activation và Jacobian, tăng độ nhạy với sai số FP32. `finite()` chỉ phát hiện NaN/Inf, không kiểm soát conditioning. Không thể suy từ số liệu trên rằng “thiếu LN là nguyên nhân duy nhất” của lỗi gradient. Ngoài ra, full model hiện có **154.156.232** tham số, khác khoảng **141M** paper báo ở Bảng IX/XV; chênh lệch này cho thấy nội bộ FF/visual/counting scope còn chưa được xác nhận, không tự chỉ ra lớp nào sai.

**Đề xuất cải tiến.** Tận dụng [decoder_a11.py](../src/mfvlr_research/decoder_a11.py), `NormalizedDecoderBlock` và `build_variant`, thay vì viết lại: P3 thêm pre-LN cho ba sublayer thông thường; Q3 thêm post-LN sau ba residual tương ứng. VIM vẫn nằm ngoài các norm, LE memory giữ nguyên. Mỗi phương án thêm **21.504** tham số, nên riêng LN không giải thích chênh lệch khoảng 13,16M với paper. [Snapshot A1.1](phase2d2_a11_acceptance_snapshot.md) xác nhận implementation/module tests đã tồn tại; không được hiểu các tài liệu A1 cũ ghi “chưa triển khai” là trạng thái mới nhất.

Phép kiểm chứng phù hợp là so A0/P3/Q3 từ cùng common weights, theo dõi RMS từng block, phân bố attention, gradient norm và sai số VJP trên fixture độc lập. Norm giúp số học tốt hơn vẫn cần kiểm chứng detection/localization sau training. Không chọn Q3/P3 chỉ vì activation nhỏ hơn, cũng không đổi initializer hoặc loss cùng lúc rồi quy toàn bộ lợi ích cho LN.

Theo probe 30/10, “LN cuối trước vocab projection” nên là một yếu tố riêng trong ablation, chẳng hạn P3 + final LN, tách khỏi vị trí LN trong block. Nếu không, so sánh A0/P3/Q3 sẽ trộn hai hiệu ứng: ổn định residual stream và thang logits của LR. Thang activation của LE cũng cần được ghi lại cùng lúc, vì memory đưa vào LD không thay đổi giữa ba biến thể.

---

**3. Loss và gradient: trọng số bằng nhau che khuất chênh lệch thang đo, trong khi additivity vẫn FAIL**

**Hiện trạng.** Sáu loss đã được nối vào `MFVLRIntegration`, có kiểm tra sample/token pairing. Tất cả hệ số bằng 1 nhưng cách reduction khác nhau. LR cộng trên 308 vị trí, kể cả padding; AR/FL lấy trung bình theo pixel. Artifact G01-A0 B=1 cho thấy gradient LR lớn hơn nhiều các loss khác. Kiểm tra gradient tổng so với tổng gradient riêng vẫn không đạt tiêu chuẩn hiện hành.

**Paper mô tả.** Trang 7–8, Eqs. (17)–(27), có AR, FL, KL, CMC, LR và FD; Eq. (27) cộng sáu loss với hệ số 1. Eq. (25) cộng token rồi chia batch. Nhưng Eq. (17) chưa chỉ rõ cách đưa các chiều ảnh về scalar; Eq. (18) thiếu chi tiết reduction/class encoding; các phương trình log của đầu ra tuyến tính cần diễn giải thành xác suất để tính CE hợp lệ. Vì vậy code phải chọn một cách thực thi cụ thể, không thể coi mọi reduction là được paper quy định đầy đủ.

**Code hiện tại.** [objectives.py](../src/mfvlr/objectives.py), các hàm bắt đầu tại dòng 66, 83, 89, 125, 134, 146 và 192:

| Loss | Cách tính trong code | Hệ quả |
|---|---|---|
| FD | CE hai lớp qua `log_softmax`, mean batch | Scalar phân loại thông thường |
| LR | CE vocabulary; sum 308 tokens, mean batch; không ignore PAD/BOS/EOS | Phụ thuộc chiều dài và số vị trí padding |
| CMC | Dot product chưa L2-normalize; CE hai chiều, positive trên đường chéo; `tau=exp(theta)`, ban đầu 0,07 | Phụ thuộc norm embedding; B=1 luôn bằng 0 |
| FL | CE hai lớp, mean batch và không gian | Không tăng tuyến tính theo số pixel như sum |
| AR | Mean squared error trên batch, RGB và pixel | Khác image-sum về hệ số `3×224×224` |
| KL | `KL(text || adapted visual)`, T=0,5, sum 512 features rồi mean batch; không detach | Gradient đi cả hai phía; không phải teacher cố định |

`unit_weight_sum()` cộng đúng thứ tự FD→LR→CMC→FL→AR→KL, không chia 6. [integration.py](../src/mfvlr/integration.py), dòng 92 và 102, nối dự đoán với `SixObjectives`.

**Bằng chứng định lượng.** [loss_rankings.json](../artifacts/phase2d2_d02b8/loss_rankings.json), đã đọc trực tiếp, ghi norm L2 gradient riêng trên toàn roster của G01-A0:

| Loss | Norm gradient L2 |
|---|---:|
| LR | 37.142.368,66 |
| KL | 1.444,87 |
| FD | 1.046,97 |
| FL | 1,609 |
| AR | 1,573 |
| CMC | 0 |

LR lớn khoảng **35.476 lần FD** và **25.706 lần KL**. Theo [forensics D02B8](phase2d2_d02b8_gradient_forensics.md), scalar LR là **248.989,578125**; 308 vị trí gồm 276 PAD, 24 content, 4 BOS và 4 EOS. PAD chiếm khoảng **75,16%** tổng NLL khi cộng nhóm bằng FP64. Con số này không chứng minh PAD cũng chiếm 75,16% gradient, vì các vector gradient có thể cộng hoặc triệt tiêu.

Ngay cả khi logits đồng đều, LR theo chính reduction đang dùng đã là `308×ln(49408) ≈ 3328,82`, trong khi FD là `ln(2) ≈ 0,693`. Đây là mốc toán học minh họa thang scalar, không phải ước lượng gradient thực tế hay lời giải thích đầy đủ cho LR ≈249 nghìn.

**Kiểm chứng bổ sung (30/10/2026).** Phần chênh lệch còn lại, khoảng 75 lần so với mốc 3.328, đến từ thang logits: probe ở Vấn đề 2 đo được NLL 806,6 nats/token tại khởi tạo, và giảm còn 9,83/token khi chuẩn hóa hidden trước vocab projection. Như vậy độ lớn của LR, và nhiều khả năng cả norm gradient 3,7×10⁷, chủ yếu là hệ quả của Vấn đề 2 chứ không chỉ của reduction token-sum. Cùng probe B=2 (một thật, một EFS), PAD chiếm 70,7% tổng NLL, cùng bậc với 75,16% ở G01.

Artifact B=2 [total.json](../artifacts/phase2d2_n31/g2/final_real_efs/total.json) ghi thêm `cmc = 206,23` và `kl = 30,71`, bên cạnh `lr = 255.077,56` và `fd = 3,39`. Với hai lớp, CE của CMC ở phân phối đều chỉ là `ln 2 ≈ 0,69`; giá trị 206 cho thấy softmax đã bão hòa do raw dot chia τ = 0,07 (chi tiết ở Vấn đề 5). CMC vì vậy là scalar lớn thứ hai ở batch có hơn một mẫu, dù bằng 0 ở G01 B=1.

[summary.json](../artifacts/phase2d2_d02b8/summary.json) ghi **293/508 tensor** và **8.772.234 tọa độ** không đạt đối chiếu additivity, với ngưỡng lịch sử `atol=1e-6, rtol=1e-5`. Ý tưởng kiểm tra là:

`grad(sum_i L_i)` đối chiếu với `sum_i grad(L_i)`.

Trong số học chính xác, đạo hàm tuyến tính theo tổng; hai cách backward FP32 có thể khác thứ tự tính/cộng và sai số làm tròn. FP64 ở đây dùng để cộng các gradient riêng vốn đã tính bằng FP32; nó không biến chúng thành đạo hàm ground truth chính xác hoặc full FP64 autodiff. [module_rankings.json](../artifacts/phase2d2_d02b8/module_rankings.json) ghi lỗi ở FLT 179/387 tensor, MVE 88/88 và VD 26/28; norm tương đối toàn module nhỏ không loại trừ nhiều tọa độ yếu bị sai ngưỡng. Các kiểm tra lịch sử cùng-forward cũng có sai khác; chưa thể quy mọi lỗi cho việc tạo graph riêng.

**Lý do chọn.** Đây là vấn đề ưu tiên vận hành cao nhất: tính đúng công thức scalar không bảo đảm các mục tiêu tác động cân bằng hoặc gradient đủ tin cậy để tiếp tục training. Baseline giữ token-sum và hệ số 1 vì Eq. (25)/(27) thể hiện rõ; dùng log-softmax để tạo CE hữu hạn có nghĩa. Cách này bảo toàn một đối chứng truy vết được, thay vì âm thầm điều chỉnh objective cho dễ chạy.

**Ưu điểm.** Loss tách thành hàm độc lập, reduction rõ, không log trực tiếp raw logits; giữ toàn bộ graph và không giấu lỗi bằng clipping hoặc nới tolerance. Kiểm tra pairing ngăn ghép sai ảnh, mask và prompt. Artifact lưu cả kết quả FAIL giúp phân biệt “kiểm chứng lại báo cáo thành công” với “gradient đạt”.

**Nhược điểm.** Hệ số 1 không có nghĩa ảnh hưởng bằng nhau. LR có thể lấn át tín hiệu detector/mask; padding làm mục tiêu tiêu tốn phần lớn supervision cho cấu trúc chuỗi ở fixture này. Norm toàn roster còn bao gồm tập tham số khác nhau cho từng loss, nên cần xem thêm norm/cosine trên cùng nhóm tham số chia sẻ. G01 chỉ có một ảnh thật: CMC bằng 0 và không có foreground giả, vì vậy không đại diện batch training. LR lớn là yếu tố rủi ro, chưa được chứng minh là nguyên nhân duy nhất của additivity FAIL.

**Đề xuất cải tiến.** Tách kiểm chứng A0 khỏi thay objective. Bước chẩn đoán hợp lý là control cùng một forward, giữ graph để tính total và sáu gradient riêng, không optimizer update, rồi báo cáo đầy đủ roster, tọa độ yếu và `None` so với zero gradient. D02B10/D02B11 đã chuẩn bị hướng này; trạng thái lịch sử vẫn chưa chạy lần control mới đó. Sau đó cần batch nhiều mẫu với real/EFS/local và prompt trùng/khác để kiểm tra CMC và foreground.

Vì thang LR phụ thuộc mạnh vào thang logits, ablation về reduction/masking của LR nên được đọc cùng ablation norm ở Vấn đề 2; giảm LR bằng token-mean khi logits vẫn có std hàng trăm chỉ chia nhỏ một scalar bão hòa. Ở nhánh cải tiến riêng, so lần lượt: LR token-mean với cùng token set; LR bỏ PAD nhưng giữ cách sum; rồi mới cân nhắc LR non-PAD mean hoặc hệ số `lambda_lr`. Không thay cả masking, reduction và weight trong một ablation. Theo dõi norm/cosine từng loss trên shared encoder/trunk, NLL tách content/PAD, cùng AUC/mIoU. Bỏ PAD có thể tập trung supervision hữu ích nhưng đổi objective; giảm LR có thể cải thiện cân bằng nhưng cũng làm mất lợi ích semantic mà paper báo. Không dùng hạ loss scale hoặc gradient clipping để tuyên bố lỗi baseline đã được giải quyết.

---

**4. VIM: một visual token khiến attention không phụ thuộc query và Q/K có gradient task bằng 0**

**Hiện trạng.** Có 19 VIM độc lập: 12 trong LE, 7 trong LD. Mỗi VIM lấy language tokens làm Q, đúng một fused visual token làm K/V, rồi softmax theo key. Với cấu hình đó, phần injection là một phép biến đổi visual vector được broadcast cho mọi vị trí từ.

**Paper mô tả.** Hình 5 trang 6 và Eqs. (4)–(11) trang 7 mô tả language query, class token làm key/value, chia heads và output projection. Eq. (10) in mẫu số `sqrt(r/d)`; paper diễn giải đây là tương tác ảnh–ngôn ngữ ở mức từ. Eq. (8)–(9) ghi rõ `K_{j,i}, V_{j,i} ∈ R^{1×d/r}`, tức key/value chỉ có một token theo chính công thức đã công bố; trạng thái suy biến dưới đây là tính chất của phương pháp như được viết, không phải do code diễn giải sai. Bảng XV trang 15 báo lựa chọn class-token tốt hơn các phương án all/patch trong thí nghiệm của tác giả. Paper không ghi rõ trục softmax bằng một tensor API.

**Code hiện tại.** [vim.py](../src/mfvlr/vim.py), dòng 28–85: `visual.shape[1]` phải bằng 1; Q/K/V/O là bốn linear không bias; `denominator=sqrt(8/512)=1/8`; logits `[B,8,N,1]`; `softmax(dim=-1)`.

Với logits hữu hạn, suy ra chính xác:

```text
softmax([a]) = [1]
Delta_t = W_out(W_v(I_v))     cho mọi vị trí t
d Delta / d Q = 0; d Delta / d K = 0
```

Ký hiệu trên dùng quy ước hàm linear; không hàm ý thứ tự nhân matrix khác code. Giá trị Delta không chọn visual feature theo từng query. Text hidden sau residual vẫn khác nhau giữa các từ; V/O và đường gradient về `I_v` vẫn học được. Do đó không được kết luận cả nhánh vision injection vô ích hoặc tất cả gradient VIM đều mất.

Q/K của 19 VIM gồm `19×2×512×512 = 9.961.472` tham số không nhận gradient từ task loss qua attention này. Khi có graph kết nối, chúng có thể có tensor gradient zero; đường không kết nối có thể là `None`. Nếu sau này có weight decay phù hợp, tham số vẫn có thể thay đổi vì regularization. [D02B8](phase2d2_d02b8_gradient_forensics.md) ghi nhóm 38 Q/K tensor singleton có zero gradient và không có tọa độ additivity FAIL; không nên nhầm nhóm này với Q/K của attention nhiều token hoặc sai số triệt tiêu key bias.

**Kiểm chứng bổ sung (30/10/2026).** Backward LR trong probe toàn mô hình ở Vấn đề 2 cho `max|grad|` của Q và K **đúng bằng 0** ở cả 19 VIM, trong khi gradient V nhỏ nhất vẫn có norm khoảng 3,6×10⁴. Kết quả này khớp hệ quả toán học ở trên và với nhóm 38 tensor của D02B8.

**Lý do chọn.** Đây là điểm khác biệt giữa cơ chế được diễn giải và cơ chế toán học thực thi, đồng thời giải thích một nhóm gradient bằng 0. Baseline giữ class token và mẫu số được in trong paper để không tự sửa phương pháp; softmax theo key và API delta-only là các diễn giải kỹ thuật được ghi rõ.

**Ưu điểm.** Tiêm ngữ cảnh ảnh vào mọi token với attention map nhỏ; không cần ma trận text×patch lớn. Có thể vẫn tạo embedding ngôn ngữ phụ thuộc ảnh qua residual/FF và các block sau. Giữ Q/K đầy đủ giúp kiểm chứng công thức và phơi bày overflow thay vì lược đồ thị rồi bỏ qua nó.

**Nhược điểm.** Không có lựa chọn visual key theo từ; gần 9,96M tham số Q/K không được task supervision sử dụng. Việc tính Q/K vẫn tốn tài nguyên và có thể overflow trước softmax dù kết quả lý tưởng là hằng 1. Mẫu số `1/8` tạo logits lớn gấp 64 lần so với mẫu số chuẩn `sqrt(64)=8` khi Q/K giữ nguyên. Với singleton hữu hạn, thay scale không đổi output attention, nên sửa riêng mẫu số không khôi phục khả năng chọn theo từ. Với nhiều key thì thay scale có tác động thực sự.

**Đề xuất cải tiến.** Có hai hướng khác mục tiêu. Nếu ưu tiên giữ hành vi singleton và giảm chi phí, thử module chỉ gồm `V→O` rồi broadcast; phải kiểm chứng cả forward, V/O gradient và khác biệt xử lý nonfinite/checkpoint vì đây không phải thay thế bitwise đầy đủ cho graph hiện hành. Nếu ưu tiên tương tác theo từ, thử nhiều visual tokens, chẳng hạn một tập patch đã pooling, để softmax có nhiều key. Hướng này cần quyết định rõ token lấy từ appearance hay residual và cách fuse, thêm chi phí attention, đồng thời là kiến trúc mới so với baseline class-token.

Thiết kế kiểm chứng nên tách số key và attention scale thành các yếu tố riêng; đo gradient Q/K, sự thay đổi attention khi đổi query, AUC/mIoU và bộ nhớ. Nếu thêm gate phụ thuộc text vào global injection thì cũng phải ghi thành biến thể riêng. Không diễn giải lợi ích của nhiều key như bằng chứng rằng tác giả đã dùng cấu hình đó.

---

**5. Nhánh ngôn ngữ: LD nhìn được prompt qua memory, global token luôn là PAD và alignment không có teacher độc lập**

**Hiện trạng.** FLT dùng tokenizer tương thích CLIP nhưng embedding/transformer khởi tạo từ đầu. Bốn prompt được đóng thành 4 đoạn 77 token; LE nhìn toàn bộ 308 vị trí, không padding mask. LD có causal self-attention nhưng cross-attention đọc toàn bộ LE memory, vốn được tính từ chính target prompt. Global language feature lấy vị trí vật lý cuối cùng. KL và CMC căn chỉnh ảnh với text embedding đã nhận thông tin ảnh qua VIM.

**Paper mô tả.** Trang 6–8, §III-D, Hình 6 và Eqs. (12)–(16), (24)–(25): dùng prompt phân cấp, lấy token cuối làm embedding ngôn ngữ, dịch input LD, nhận LE features để tái tạo prompt, dùng lại vocabulary matrix ở output. Trang 8 dùng KL/CMC để alignment; phần CMC vừa gọi similarity là cosine vừa diễn giải `sim` là dot product. Hình 4 tách training và inference; language chỉ phục vụ training. Các chi tiết EOS/PAD, memory mask, teacher detach và cách giải quyết prompt trùng không được nêu đủ để suy ra một cấu hình duy nhất.

**Code hiện tại.**

- [tokenizer.py](../src/mfvlr/tokenizer.py), `encode_levels()` dòng 132: mỗi đoạn `[BOS, content, EOS, PAD...]`, ghép thành 308 vị trí; trả `padding_mask` và `eos_positions` nhưng FLT không dùng chúng.
- [language_encoder.py](../src/mfvlr/language_encoder.py), dòng 165 và 197: `padding_idx=None`, LE bidirectional, global feature là `hidden[:,307:308,:]`. Nếu prompt L4 chưa lấp đầy 77 vị trí thì slot 307 là PAD; với registry hiện tại điều này luôn xảy ra (xem kiểm chứng bổ sung). Hidden ở đó vẫn chứa ngữ cảnh sau attention, không phải vector PAD thô hay zero.
- [flt.py](../src/mfvlr/flt.py), dòng 55–79: shift một lần trên toàn chuỗi, không shift từng đoạn; output projection dùng đúng `vocabulary.weight`, không có pretrained CLIP weights.
- [language_decoder.py](../src/mfvlr/language_decoder.py), dòng 20: self-attention causal, memory-attention không mask. [objectives.py](../src/mfvlr/objectives.py), dòng 104–122 và 146–159: CMC dùng raw dot/diagonal positives; KL không detach text.
- [integration.py](../src/mfvlr/integration.py), dòng 86 và 119: `visual_only()` không gọi language; `ImageOnlyMFVLR` không khởi tạo FLT/adapter/objectives. Hành vi inference chỉ dùng ảnh đã được thể hiện đúng ở ranh giới module.

**Lý do chọn.** Hiệu quả của nhánh ngôn ngữ phải thể hiện bằng visual representation tốt hơn khi bỏ text ở inference. Nếu reconstruction chủ yếu dựa vào prompt memory hoặc cấu trúc padding, LR giảm chưa chắc biểu thị học tương tác ảnh–ngôn ngữ hữu ích. Baseline chọn full memory để thực thi LE→LD communication của paper, slot cuối để diễn giải “last token”, tied weight theo Eq. (24), và raw dot theo phần định nghĩa `sim`; đây là các quyết định cần giữ minh bạch.

**Ưu điểm.** Prompt phân cấp cung cấp supervision từ real/fake đến loại thao tác, họ generator và generator cụ thể. Tied vocabulary tránh một ma trận output độc lập lớn. Full memory làm bài toán tái tạo khả thi; VIM cho phép cùng một prompt có embedding khác nhau theo ảnh. Inference loại nhánh ngôn ngữ nên không cần biết generator label/prompt lúc triển khai. Physical visual model có **41.135.303** tham số theo `ownership_report`, ít hơn full model 154.156.232.

**Nhược điểm.** Causal mask của LD không làm toàn bộ hệ thống thành language model tự hồi quy: vị trí hiện tại có thể nhận thông tin target/tương lai qua bidirectional LE memory. Đây là đường tắt trong nhiệm vụ reconstruction, không phải bằng chứng rò rỉ label ở inference ảnh-only. Global slot PAD và việc không mask có thể khiến representation nhạy với cách đóng gói prompt. KL hai phía căn chỉnh hai nhánh cùng học; nhánh text đã phụ thuộc ảnh nên không phải teacher semantic độc lập, dù tên hàm là knowledge distillation.

CMC raw dot nhạy với norm embedding. Positive chỉ theo đường chéo khiến hai mẫu cùng ngữ nghĩa/prompt vẫn là negative của nhau; real prompts trong [prompts.py](../src/mfvlr/prompts.py) còn lặp cùng một câu ở cả bốn level. VIM làm embedding của chúng có thể khác nhau, nhưng không loại bỏ xung đột ngữ nghĩa của diagonal-only target. B=1 không kiểm tra được hiện tượng này. Ngoài ra, registry generator hữu hạn đòi hỏi metadata mapping chuẩn; repo chưa có pipeline dataset hoàn chỉnh chứng minh đủ coverage.

**Kiểm chứng bổ sung (30/10/2026).**

- *Global token luôn là PAD.* Token hóa prompt cho ảnh thật và cả 9 generator trong `GENERATOR_REGISTRY` cho payload dài nhất theo level là 6, 9, 10 và 12 token; đoạn L4 dài nhất chỉ chiếm 14/77 vị trí kể cả BOS/EOS. Slot 307 là PAD (`id 0`) ở mọi mẫu. Vậy `T_l` luôn là hidden tại một vị trí PAD, không phải trường hợp biên. Paper viết “the last token in `T^e_hig`”; slot vật lý cuối cùng là một cách diễn giải, EOS là cách diễn giải khác.
- *LD có thể sao chép trực tiếp target qua memory.* Input LD tại vị trí t là token t−1 (shift trong [flt.py](../src/mfvlr/flt.py), dòng 55–58), nhưng target là token t chưa shift, và memory tại vị trí t là hidden LE của chính token t. Memory attention không mask nên đường tắt không chỉ là “nhìn thông tin tương lai”: target nằm sẵn trong memory ở cùng vị trí. Tại khởi tạo, tỷ lệ argmax trùng target là 0, nên mức khai thác đường tắt chỉ đo được sau training.
- *CMC tại khởi tạo bị norm chi phối.* Cùng probe B=2: norm `I_v` khoảng 96,05, norm `T_l` khoảng 1.015. Raw dot giữa các cặp nằm trong 1.878–1.926; chia τ = 0,07 cho score khoảng 26.836–27.507, trong khi cosine của mọi cặp chỉ khoảng 0,019–0,020. Cột có `T_l` norm lớn hơn thắng ở cả hai hàng, nên softmax bão hòa phản ánh norm chứ không phản ánh việc ghép đúng cặp. Đây là cơ chế của `cmc = 206,23` trong artifact B=2 nêu ở Vấn đề 3.

**Đề xuất cải tiến.** Trước hết đo mức sử dụng thông tin bằng các control tách biệt: xáo trộn visual conditioning giữa mẫu, loại visual injection, che/đổi memory, và báo cáo LR trên content token. Đây là probe chẩn đoán; tác động của chúng cần đi kèm đánh giá visual-only sau training để kết luận lợi ích thật.

Sau đó khảo sát riêng từng thay đổi: global embedding tại EOS hoặc pooling trên token hợp lệ; padding mask; denoising/corrupted prompt ở encoder; normalized cosine cho CMC; multi-positive contrastive theo nhãn phân cấp. Mỗi thay đổi có đánh đổi: EOS/pooling đổi định nghĩa representation; denoising làm task khó hơn; cosine loại ảnh hưởng norm nhưng đổi similarity; multi-positive giảm negative sai ngữ nghĩa nhưng có thể mất phân biệt generator nếu gom quá thô. Chỉ mask memory theo vị trí vẫn chưa bảo đảm causality nếu LE token đã trộn thông tin tương lai; muốn mô hình causal thực sự phải xử lý cả encoder context. Nếu thử stop-gradient text trong KL, cần ghi rõ text vẫn nhận ảnh qua VIM và vẫn có thể được cập nhật bởi LR/CMC; detach một loss không tạo ra teacher độc lập cố định.

---

**Thứ tự xử lý đề nghị và tiêu chí đánh giá**

| Thứ tự | Công việc | Kết quả cần có trước khi kết luận |
|---|---|---|
| 1 | Hoàn tất chẩn đoán gradient A0 có giới hạn, kiểm soát cùng-forward và xem từng module/tọa độ | Phân biệt lỗi graph, độ nhạy số học và ảnh hưởng thang loss; giữ nguyên FAIL lịch sử |
| 2 | Chuẩn hóa hợp đồng dữ liệu/mask/prompt, batch nhiều mẫu và đánh giá visual-only | Có nguồn dữ liệu, pairing và protocol đủ để kiểm chứng hiệu quả; B1 real không thay thế bước này |
| 3 | So A0/P3/Q3 và final LN trước vocab như một yếu tố riêng, sau đó ablation LR masking/reduction tách biệt | Bằng chứng ổn định số học, thang logits và đóng góp gradient, rồi mới kết quả training |
| 4 | So bottleneck/skips và singleton/multi-token VIM theo từng yếu tố | AUC/mIoU cùng chi phí tài nguyên; reconstruction hoặc attention map đẹp chưa đủ |
| 5 | Kiểm chứng đường tắt ngôn ngữ, pooling và objective alignment | Lợi ích trên inference ảnh-only, qua domain/generator chưa thấy |

Đây là thứ tự đề xuất nghiên cứu, không phải các thí nghiệm đã thực hiện. Source nghiên cứu P3/Q3 có sẵn nhưng không nằm trên đường triển khai bắt buộc để hoàn thiện một A0 đầu tiên. [Báo cáo integration gaps D02B10](phase2d2_d02b10_integration_gaps.md) và [trainer hiện tại](../src/mfvlr_research/d02b9/trainer.py) cho thấy trainer vẫn là giao diện chuẩn bị, `fit/step/backward` bị vô hiệu hóa. Không có căn cứ từ các unit test hoặc artifact nêu trên để báo cáo accuracy, hội tụ hay ưu thế của một cải tiến chưa huấn luyện.

**Phạm vi xác minh của báo cáo.** Đã đọc trực tiếp source liên quan, kiểm tra hash paper, xem các trang công thức/hình/bảng, đối chiếu JSON tổng hợp gradient và activation với báo cáo lịch sử. Không tái cộng toàn bộ 2.392 gradient shards trong lần này; các kết luận về việc kiểm tra toàn bộ shard được dẫn lại từ D02B8. Không coi tài liệu lịch sử ghi PROPOSED/NOT IMPLEMENTED là trạng thái hiện tại nếu source và hồ sơ mới hơn đã cập nhật. Các đề xuất cải tiến là đánh giá kỹ thuật dựa trên graph và bằng chứng hiện có, chưa được chứng minh cải thiện benchmark.

Lần cập nhật 30/10/2026 bổ sung: tính lại số tham số VD (bottleneck và Option A) từ topology và khởi tạo full model để đếm 154.156.232 tham số; chạy 25 test `tests/decoder`; chạy các probe forward/backward tại khởi tạo đã mô tả trong từng mục (CPU FP32, Python 3.14.3, torch 2.12.1+cpu, tokenizer dùng dependency trong `.phase2c1_deps`). Các script probe chỉ dùng tạm, không được commit; mọi điều kiện cần để tái lập (seed, input, prompt, biến thể) đã ghi trong các đoạn tương ứng. Probe ở trạng thái khởi tạo không thay thế đo lường trong training, và không có kết luận nào về accuracy hay hội tụ được rút ra từ chúng.
