# Chuẩn bị nhánh language: residual, normalization và gradient

## Thay đổi đã triển khai

- `model.vim_residual`: `single` mặc định giữ các lượt cũ; `double` thực hiện đúng 2x+Delta tại VIM của cả LE/LD, không thành3x. Không chạy attention hai lần, không cộng Delta hai lần. Không đổi 12 block LE/7 block LD hoặc19 VIM riêng.
- `model.language_normalization`: `baseline` giữ cách cũ; `pre_norm` thêm LN trước VIM/FF của LE, trước self-attention/cross-attention/FF/VIM của LD, và LN cuối LE/LD. Norm cuối LE áp dụng trước EOS pooling và làm memory cho LD; norm cuối LD trước vocabulary projection. Giữ LN attention LE hiện có, thêm54 LN với55.296 tham số.
- Ở pre_norm, nhánh residual vẫn là factor*x; Delta nhận LN(x). Đây là thay đổi kiến trúc thực nghiệm được khai báo, không khẳng định chi tiết tác giả MFVLR. Không sửa A0 hoặc config/checkpoint cũ.
- EOS cuối, bỏ target PAD khỏi LR, L2 CMC đã có và được giữ. LR vẫn token-sum/batch-mean, không đổi sang token-mean.
- Singleton class-token K/V, trục softmax, scale VIM, full LE memory, causal self-attention LD, instance-positive CMC, KL hai phía cùng học được giữ. Normalization không tự giải quyết query-degeneracy hoặc đường tắt qua memory.

Config pilot: `configs/genface_language_readiness.json`. Các trọng số bắt đầu FD0,1/AR1/FL1/LR1e-4/CMC0,01/KL0,01, pixel weight[1,4]. Đây là trọng số thử, chưa hiệu chỉnh theo gradient. Default config chọn stop_all, có thể đối chứng coupled.

## Đo trước khi cập nhật

Dùng hai ảnh train cố định (một thật/một Diffae) trong tập16 cũ, feature ảnh lấy từ checkpoint visual5.000/30epoch đã hoàn thành. Language khởi tạo mới seed0; không optimizer, không validation/test. Visual RMS khoảng9,92. CMC batch thực sự2, không phải effective batch4.

| Biến thể | RMS logit | LR thô | Norm gradient LR tại visual token |
|---|---:|---:|---:|
| Single, baseline norm | 1.317,06 | 208.752,84 | 4.842,87 |
| Double, baseline norm | 157.138.516,53 | 22.746.787.840 | 676.032.748,78 |
| Double + pre_norm | 0,4547 | 414,43 | 0,3736 |

Double không normalization khuếch đại rất lớn, dù chưa có NaN/Inf ở cặp mẫu này. Pre_norm và final-LN kiểm soát đầu ra/gradient của phép đo; không coi đây là bằng chứng toàn bộ training ổn định. Residual2x vẫn làm hidden stream tăng: RMS block LE12 khoảng23.981 và LD7 khoảng742 trước final-LN. Không tuyên bố mọi activation đã được giới hạn quanh1.

Đổi ảnh nhất quán ở cả VIM LE và LD, giữ prompt:

- Double+pre_norm: matched LR414,43; zero visual409,82; swap visual413,69.
- RMSE logit so với matched: zero0,6148; swap0,4323.
- Vì output đổi khi đổi visual nên đường ảnh có ảnh hưởng. Language đang chưa train, matched LR không thấp hơn swap nên chưa chứng minh nó dùng ảnh đúng ngữ nghĩa. Chưa giải quyết nghi vấn đường tắt qua full LE memory.

Với trọng số hiện tại, norm gradient tại visual token của LR/CMC/KL lần lượt khoảng0,0000374 /0,000310 /0,01383. KL mạnh nhất trong ba đường này ở cặp đo, chưa đủ để chọn trọng số cho toàn dataset. Đây là gradient theo visual feature, không phải theo toàn bộ encoder và không bằng bước cập nhật Adam.

## Pilot train nhỏ đã hoàn thành

16 ảnh train8thật/8Diffae, warm-start nhánh ảnh từ final epoch30; language/adapter khởi tạo mới. Hai chế độ cùng initial-state hash, lịch batch, ảnh/prompt, source model và loss weights. AdamLR1e-5, WD0, microbatch2/effective4, FP32 deterministic, activation checkpointing. Mỗi chế độ8cập nhật; không validation/test và không lưu checkpoint lớn cho lượt smoke.

| Chế độ | IoU train trước → sau | MSE train sau | AUC train sau |
|---|---:|---:|---:|
| Stop_all | 48,27% → 49,34% | 0,00028270 | 100% |
| Coupled | 48,27% → 48,76% | 0,00028399 | 100% |

Tất cả loss, gradient, tham số được kiểm tra hữu hạn. Ở bước1 cùng model/batch, total-gradient norm encoder0,3969 với stop_all và2,7118 với coupled (gấp6,83). Decoder cùng0,1990 ở bước1: đường residual về AD vẫn detach, language không trực tiếp sửa trunk qua FD/residual; ảnh hưởng qua encoder chung vẫn có.

Stop_all chỉ chặn language → visual, không đóng băng encoder: encoder/decoder vẫn học AR/FL/FD. Do đó việc IoU stop_all tăng không phải bằng chứng language cải thiện mask. Không có control image-only cùng8 bước trong lượt này. Sau bước đầu model đi theo các trạng thái khác nhau; không diễn giải chênh lệch norm các bước sau là gradient language riêng lẻ. Hai chế độ đều AUC100% trên16ảnh đã thuộc train, không chứng minh tổng quát hóa.

Pilot001 bị ngắt khi coupled đạt3/8; đã đánh dấu INTERRUPTED, giữ control stop_all hoàn thành. Pilot002 chạy lại coupled từ đầu đủ8 bước và xác minh reference control, không ghép3 bước cũ vào lượt mới.

## Kiểm tra tự động và hướng tiếp

`tests/pilot`: **51 passed trong43,02giây**. Bao gồm kiểm tra 2x+Delta tại actual LE/LD call sites, idempotent không thành3x, bảo toàn trọng số cũ khi bật double, pre_norm giữ trọng số nền và có gradient hữu hạn. Không mở rộng test lịch sử toàn repo.

Chưa chạy lượt language dài trên Kaggle. Đề xuất giai đoạn đầu làm quen language với stop_all, sau đó đối chứng coupled với trọng số KL thấp hơn dựa trên gradient đo nhiều batch; giữ image-only control và cùng budget. Chưa chọn giá trị KL tối ưu từ hai mẫu. Cần pilot có validation trước khi chạy5.000ảnh dài; không dùng lại test để lựa cấu hình. Việc LR có phụ thuộc ảnh đúng ý nghĩa còn cần đánh giá sau khi language đã học, bằng perturbation cùng prompt và kiểm soát full LE memory.

## Artifact và lệnh

- Diagnostic3biến thể: `artifacts/language_readiness_20261028/run_002/report.json`.
- Control stop_all: `artifacts/language_readiness_20261028/pilot_001/report.json`.
- Coupled hoàn thành: `artifacts/language_readiness_20261028/pilot_002/report.json`.
- Preview16ảnh train (xuất4ảnh): `pilot_001/stop_all/preview.png`, `pilot_002/coupled/preview.png` trong artifact trên.

```powershell
python tools/diagnose_language_readiness.py --output artifacts/language_readiness_20261028/run_003
python tools/run_language_readiness_pilot.py --output artifacts/language_readiness_20261028/pilot_003
```

Các lệnh yêu cầu output mới để không ghi đè bằng chứng cũ. Tài liệu này không thay các metric validation/test của lượt image-only trước đó. Không commit/push trong bước này.
