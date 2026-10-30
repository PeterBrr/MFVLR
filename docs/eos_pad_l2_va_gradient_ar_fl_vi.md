# EOS pooling, bỏ PAD trong LR, L2 CMC và đo gradient AR/FL — 26/10/2026

Đã triển khai ba lựa chọn đã thống nhất trong biến thể `skip_resize_conv_v2`.
A0 và `resize_conv_v1` giữ cách tính cũ để đối chứng. Không đổi tokenizer,
attention mask, residual VIM hoặc normalization LD trong lần này.

## Hành vi mới của nhánh ngôn ngữ

| Phần | Triển khai |
|---|---|
| Global language | Lấy hidden ở vị trí EOS cuối thực sự; giữ shape `[B,1,512]` |
| Nhiều prompt | GenFace lấy EOS L4; chuỗi chỉ có hai prompt lấy EOS L2, bỏ qua phần PAD sau đó |
| Thiếu EOS | Báo lỗi; không lấy token cuối hoặc argmax ID thay thế |
| LR | Loại target ID0, giữ tổng token rồi mean batch; vẫn tính BOS/EOS/nội dung |
| CMC | L2 từng vector512 chiều, eps1e-12; dot/temperature học được như trước |
| Các đường khác | Không normalize in-place; FD/KL nhận feature gốc, không đổi diagonal positives |

```text
p_b = max{i | token_ids[b,i] == 49407}
T_global[b,0,:] = T_high[b,p_b,:]

LR = sum_b sum_t 1[target[b,t] != 0] * NLL[b,t] / B
CMC_scores[i,j] = dot(normalize(v_i), normalize(t_j)) / tau
```

Chỉ mask target loss, không xóa PAD khỏi308 vị trí hoặc vocabulary, không
đóng băng PAD embedding và không thêm attention padding mask. LR chưa chuyển
sang mean theo số token hợp lệ. Việc bỏ PAD có thể làm scalar LR giảm cơ học;
chưa chứng minh nhánh language dùng ảnh tốt hơn.

Code: `src/mfvlr_research/pilot_language.py`, được nối vào `pilot_skip.py`.
Các config ảnh hiện có vẫn để LR/CMC/KL=0 để cô lập thí nghiệm nhánh ảnh.
Khi bật các loss này bằng `loss_weights`, hành vi mới được dùng tự động;
không có nghĩa đã train lại model đầy đủ với language.

## Kiểm tra đã thực hiện

- Bộ `tests/pilot` tại thời điểm kiểm tra: **45 test pass**. Có kiểm tra EOS nhiều đoạn/hai cấp,
  gradient pooling đúng vị trí, thiếu EOS, PAD target gradient0, giữ BOS/EOS,
  reduction token-sum/batch-mean và toàn PAD cho connected-zero.
- CMC bất biến khi nhân độ lớn từng phía bằng hệ số dương, không sửa tensor
  đầu vào, gradient qua hai feature và temperature vẫn hữu hạn/kết nối.
- Đã kiểm tra các hàm được gọi thực sự từ `ManualObjectives` và full FLT forward.
- Kiểm tra FLT thật trên CPU bằng prompt real/Diffae offline, batch2×308:
  EOS cuối ở238/243, global `[2,1,512]`,540 target PAD có gradient LR bằng0;
  EOS target vẫn có gradient. Backward LE/LD/visual/temperature đều hữu hạn.
- Fixture CPU dùng visual conditioning ngẫu nhiên cố định, không phải batch
  ảnh hay kết quả training. LR non-PAD~33.145,64, CMC~0,83665: LR vẫn lớn,
  nên ba sửa đổi chưa giải quyết normalization LD hoặc cân hệ số LR.

[Report kiểm tra language](../artifacts/eos_pad_l2_check_20261026/report.json).
Script: `tools/check_pilot_language.py --output <file-json-mới>`; không có
optimizer update trong kiểm tra này.

## Phép đo gradient AR và FL

- Model: skip concat, detach B trước RE, nearest resize, AD/MD chung trunk.
- Hai trạng thái: khởi tạo seed0 và checkpoint sau16 cập nhật của lượt
  `skip_detach_smoke_20261026/best.pt`.
- Cùng8 microbatch, mỗi batch1 real +1 Diffae:16 ảnh không lặp, chọn đều trong
  chính32 ảnh train của lượt tiny. Không dùng validation/test; không optimizer.
- AR mean squared error toàn ảnh; FL weighted pixel CE với class weights[1,4].
  Norm raw tính riêng; tỷ lệ có trọng số dùng lambda_AR=1, lambda_FL=0,25.
- Đo mọi tham số CNN/trunk và chi tiết từng stage; các head kiểm tra đường
  kết nối riêng. Gradient FP32, tính norm/dot bằng FP64 trên CPU; không phải
  full FP64 autodiff hoặc kiểm tra additivity lịch sử.
- Đã đối chiếu forward trực tiếp encoder/decoder với full image route, kết quả
  bằng nhau chính xác. Không cần chạy detector/RE để lấy đạo hàm của AR/FL.

| Trạng thái | Nhóm tham số | Cosine trung bình | Min..max | Batch âm | Mean norm(0,25g_FL)/norm(g_AR) |
|---|---|---:|---|---:|---:|
| Khởi tạo | CNN encoder | −0,2476 | −0,3311..−0,1364 | 8/8 | 0,0774 |
| Khởi tạo | Trunk decoder | −0,1994 | −0,2589..−0,1518 | 8/8 | 0,0782 |
| Sau16 bước | CNN encoder | −0,1854 | −0,3085..−0,0589 | 8/8 | 0,5563 |
| Sau16 bước | Trunk decoder | −0,1530 | −0,2892..−0,0596 | 8/8 | 0,5606 |

![Cosine AR/FL](../artifacts/ar_fl_gradients_20261026/ar_fl_cosine.png)

Chi tiết theo stage có khác biệt: ở checkpoint, một số stage có batch cosine
dương dù tổng CNN/trunk đều âm trong8 batch. Không suy mọi tọa độ/tầng luôn
nghịch hướng. Appearance head không có đường FL; mask head không có đường AR,
cosine ở các head được ghi null thay vì tạo số đo giả.

## Diễn giải và bước tiếp theo

1. Có bằng chứng AR/FL ngược hướng trên các nhóm chia sẻ trong16 ảnh đã đo,
   ở cả khởi tạo và sau16 bước. Chưa đại diện toàn GenFace hoặc mọi epoch.
2. Detach B đã ngắt FD về decoder qua residual; không ngắt AR/FL cùng tác động
   trunk. Bằng chứng mới phù hợp với việc cần kiểm tra cân loss hoặc tách trunk.
3. Nhân FL bằng hệ số dương đổi độ lớn tương đối, không đổi cosine AR–FL.
   Với lambda0,25, norm FL sau16 bước khoảng56% norm AR trên trunk; không thể
   kết luận nên giảm FL thêm chỉ từ con số này khi localization còn yếu.
4. Cosine âm không chứng minh gradient tổng có trọng số hoặc bước Adam làm
   một loss tăng. Cần theo dõi quá trình tối ưu và validation khi so phương án.
5. Hướng kiểm chứng: giữ data/init/budget, so lambda_FL có kiểm soát; hoặc
   so trunk AD/MD riêng để tách đường cạnh tranh decoder. Chưa chạy các lượt
   đó trong phép đo này, chưa tự áp PCGrad/GradNorm hoặc thay trọng số tối ưu.

Phép đo được chạy **trước khi nối ba thay đổi language** để giữ đúng source
pin của checkpoint. Ba thay đổi chỉ tác động FLT/LR/CMC, không đổi graph
AR/FL của nhánh ảnh. Source tại thời điểm đo được lưu theo đúng hash trong
`artifacts/ar_fl_gradients_20261026/source_snapshot/`; checkpoint gốc không đổi.
Script đo giữ strict source/manifest checks; muốn đo checkpoint khác phải dùng
đúng source tương ứng, không bỏ qua provenance để chạy với graph khác.

[Report gradient đầy đủ](../artifacts/ar_fl_gradients_20261026/report.json).
Script: `tools/measure_ar_fl_gradients.py --checkpoint <checkpoint-đúng-source>
--output <thư-mục-mới> --batches 8`.

## Cập nhật sau train đối chứng GenFace

Đã hoàn tất sáu lượt mới, mỗi lượt 128 ảnh train/128 validation và 100 cập nhật,
chỉ dùng GenFace real + Diffae. So language cũ với EOS/PAD/L2: AUC validation
40,77% → 64,14%; tuy nhiên cấu hình không language đạt 66,75%. Skip/detach
giúp tái tạo, nhưng mask vẫn gần toàn nền và IoU fake chỉ 0,85–1,58%.
Chưa thể kết luận ba thay đổi language riêng lẻ cải thiện localization.

[Bảng kết quả, ảnh đối chiếu và đề xuất tiếp theo](genface_ablation_20261026_vi.md).
