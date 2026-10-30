# Đối chiếu báo cáo 5 vấn đề với bản tóm tắt giảng viên — 25/10/2026

Tài liệu đối chiếu:

- File ngoài workspace: `C:\Ổ đĩa D\Download\bao_cao_5_van_de_decoder_loss_gradient_ngon_ngu.md`.
- [Bản tóm tắt 30 phút](tom_tat_bao_cao_giang_vien.txt).

**Hai bản thống nhất ở phần lớn hành vi code.** File mới là bản kiểm toán
baseline A0 chi tiết; bản tóm tắt tổ chức ba nhóm phục vụ trình bày, có dẫn
pilot resize-conv và các quyết định người dùng đã chốt. Không nên thay nguyên
bản tóm tắt bằng file mới vì khác phạm vi và chưa cập nhật thực nghiệm gần đây.

## Ánh xạ nội dung

| File mới | Bản tóm tắt | Khác biệt đáng chú ý |
|---|---|---|
| 1. VD bottleneck, trunk chung, residual có gradient | Ưu tiên1 và2 | Cùng cơ chế; file mới giải thích rõ hợp đồng A0, còn tóm tắt có bằng chứng pha upsampling trên pilot_005 và biến thể resize-conv |
| 2. LD bảy block không LayerNorm | Mới đề cập gián tiếp trong ưu tiên3 | Đây là phần bổ sung quan trọng nhất: source xác nhận thiếu LN; P3/Q3 đã có code nghiên cứu |
| 3. Sáu loss khác thang, additivity FAIL | Ưu tiên2 | Cùng vấn đề reduction; file mới tập trung fixture G01/kiểm chứng số học, tóm tắt tập trung gradient trên ảnh train và khả năng học |
| 4. VIM một visual K/V | Ưu tiên3(d) | Thống nhất hệ quả softmax=1; file mới bổ sung ~9,96M tham số Q/K không nhận task gradient qua nhánh singleton |
| 5. Prompt memory, PAD pooling, KL/CMC | Ưu tiên3(a,b,e) và LD | Thống nhất nguy cơ đường tắt; file mới mở nhiều phương án, tóm tắt ghi quyết định EOS/bỏ PAD/L2 đã chốt nhưng chưa triển khai |

File mới có khoảng5.038 từ phân cách bằng khoảng trắng so với khoảng2.002
của bản tóm tắt; phù hợp làm phụ lục kỹ thuật hơn là đọc trọn trong30 phút.

## Những ý nên lấy từ file mới

1. **LD không có LayerNorm.** `src/mfvlr/language_decoder.py` xác nhận không LN
   trong bảy block; `flt.py` không final norm. Cần thêm vào phần language như
   một giả thuyết về thang activation/conditioning, không khẳng định là nguyên
   nhân duy nhất của lỗi train hoặc mask. P3/Q3 có tại
   `src/mfvlr_research/decoder_a11.py`, mỗi biến thể thêm21.504 tham số; chưa
   được dùng trong các pilot đang báo cáo.
2. **Gradient additivity khác gradient conflict.** Một bên so hai cách tính
   đạo hàm tổng trong FP32; một bên so hướng gradient các objective khác nhau.
   Không dùng FAIL additivity làm bằng chứng trực tiếp các task nghịch nhau,
   cũng không dùng giảm loss để tuyên bố FAIL lịch sử đã được sửa.
3. **VIM vẫn có ích dù singleton.** Q/K không được task gradient qua attention
   một key, nhưng V/output và residual/FF vẫn có thể tạo biểu diễn phụ thuộc ảnh.
4. **Causal LD không chặn thông tin qua full LE memory.** Đây là probe về cách
   dùng ảnh khi tái tạo prompt, không phải bằng chứng rò rỉ nhãn ở inference
   chỉ dùng ảnh. Kiểm tra phải thay visual conditioning ở cả LE và LD.

## Những điểm cần cập nhật hoặc phân biệt trước khi báo cáo

### 1. Trạng thái training và dataset

File mới dẫn trainer chuẩn bị `d02b9/trainer.py` và nói chưa có kết quả training/
pipeline dataset hoàn chỉnh. Các phát biểu đó cần giới hạn vào giai đoạn A0
lịch sử hoặc phạm vi fixture đang xét. Workspace hiện có `train.py`,
`evaluate.py`, loader/manifest và các lượt Diffae/FF++ đã chạy.

Ngay `REPOSITORY_STATUS.md` cũng ghi pilot nội bộ có phạm vi riêng, trong khi
Phase2D.2 vẫn BLOCKED. Hai trạng thái có thể cùng đúng; không đồng nhất BLOCKED
lịch sử với việc toàn dự án chưa từng train.

### 2. A0 khác resize-conv và các nhánh thí nghiệm

- A0: full154.156.232; nhánh ảnh41.135.303; VD9.055.365 tham số.
- Resize-conv đang thử: full152.067.272; nhánh ảnh39.046.343;
  VD6.966.405 tham số. Class weight/stop_all thuộc biến thể này.
- CNN mask riêng trong pilot mới: nhánh ảnh57.889.543; decoder vẫn chung.

Các số A0 trong file mới đúng với kiến trúc được mô tả, nhưng không đại diện
cho tất cả config/checkpoint hiện tại. Option skip12.188.805 chỉ là phương án
thiết kế được dẫn, không phải decoder đang train.

### 3. Residual VIM là khác biệt về lựa chọn

File mới ưu tiên API delta-only và caller cộng một lần để tránh cộng hai lần.
Đó cũng là code hiện tại. Bản tóm tắt đã ghi người dùng chốt cách diễn giải
`2*x+Delta`, nhưng **chưa triển khai**. Cần trình bày đây là hai cách diễn giải,
không gán lựa chọn một lần hoặc hai lần thành mã gốc tác giả đã xác nhận.

### 4. Những quyết định đã chốt không còn là danh sách phương án ngang nhau

Dự án đã chốt lấy EOS cuối thực sự, bỏ target PAD nhưng giữ token-sum/batch-mean,
L2 trước dot CMC; giữ KL hai phía cùng học và diagonal positives. Nhiều phương
án trong file mới như multi-positive, denoising hay detach teacher là đề xuất
nghiên cứu khác, chưa được chọn. Không tự đưa chúng vào recipe hiện hành.

### 5. Chưa có thực nghiệm normalization, nhưng đã có thực nghiệm nhánh ảnh

- [Bốn nhánh AR/FD/FL từ đầu trên4 ảnh](ad_scratch_test_vi.md): MSE cuối
  AR0,00532; AR+FD0,00803; AR+FL0,01357; AR+FD+FL0,23202 và dao động mạnh.
- [Tách CNN mask trên32 train/64 validation](split_mask_encoder_pilot_vi.md):
  MSE validation0,08991 ->0,18639; foreground IoU fake0,11857% ->0,001563%.
  Cô lập gradient đúng nhưng chưa cải thiện mask; trunk decoder vẫn chung.
- Không có language losses trong hai thử nghiệm trên. Không dùng chúng kết
  luận LN/PAD/VIM gây ra hoặc đã chữa lỗi; chưa phải benchmark nhiều seed.

Cả file mới và bản tóm tắt đều cần bổ sung các kết quả gần đây nếu dùng như
báo cáo hiện trạng. Các thử nghiệm mới không xóa FAIL số học A0 lịch sử.

## Khả năng kiểm chứng nguồn trong workspace hiện tại

- Commit HEAD trùng `fe2358fde3682483dc8af5173a6796ebaa35e79a` mà file mới ghi;
  working tree hiện có thay đổi, nên commit không bao quát các file pilot mới.
- Đã đọc source LD/P3/Q3 và tài liệu D02B8. D02B8 xác nhận các con số lịch sử
  293 tensor/8.772.234 tọa độ và LR norm~37,14 triệu được file mới dẫn.
- Các JSON gốc `artifacts/phase2d2_n31/g2/final_real_efs/total.json` và
  `artifacts/phase2d2_d02b8/loss_rankings.json` không có tại đường dẫn tương
  ứng trong workspace này. Chưa độc lập xác minh lại toàn bộ số activation/
  gradient từ JSON; cần giữ nhãn dẫn lại hồ sơ lịch sử.
- File ngoài đang dùng30 link tương đối; cả30 không trỏ tới tệp tồn tại khi
  giải từ thư mục Download hiện tại. Cần sửa link hoặc đóng gói đúng cây thư
  mục trước khi gửi giảng viên. Không thay đổi file ngoài trong lần đối chiếu.

## Cách dùng cho buổi30 phút

Giữ cấu trúc ba nhóm của bản tóm tắt. Bổ sung ngắn LD thiếu LN vào nhóm3;
thêm kết quả AR/FD/FL và tách encoder vào nhóm1–2. Đưa audit G01 chi tiết,
P3/Q3, công thức/roster và các phương án mở rộng vào phụ lục file5 vấn đề.
Giữ phân biệt rõ: code đang chạy, lựa chọn đã chốt, bằng chứng đã đo và giả
thuyết chưa được kiểm chứng.
