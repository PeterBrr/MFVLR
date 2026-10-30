# Xóa phần FF++ không còn sử dụng

Theo yêu cầu rõ ràng của người dùng, đã xóa vĩnh viễn ZIP/dữ liệu/checkpoint/output FF++ tại project và archive, tổng **32.058.048.304 byte (32,06 GB / 29,86 GiB)** theo kiểm kê trước xóa; 8.844 file. Dung lượng thực tế hệ điều hành thu hồi có thể khác do cách cấp phát file.

Đã bỏ hai config FF++, ba tool kiểm tra/chuẩn bị/train, test adapter và hai tài liệu chỉ dành cho FF++. Adapter `pilot_datasets.py` nay chỉ hỗ trợ GenFace real/Diffae, giữ nguyên giao diện được train/evaluate/export và runner sử dụng. Kiểm tra source grouping vẫn trả false vì GenFace chưa có ánh xạ danh tính được xác minh.

Không xóa nội dung trích paper hoặc số liệu thảo luận lịch sử trong tài liệu chung; chúng không phải dữ liệu train. Không sửa contract/snapshot học thuật hoặc bundle thí nghiệm lịch sử. Bundle Kaggle đã đóng băng có thể chứa snapshot adapter cũ: giữ nguyên hash để bảo toàn bằng chứng và lượt đang chạy, không có ảnh FF++ trong các gói GenFace này.

Giữ nguyên dữ liệu/manifest/checkpoint GenFace, code model, paper PDF và phiên Kaggle 5.000 ảnh × 30 epoch. Kaggle chạy từ bản đóng gói riêng nên việc bỏ adapter FF++ ở máy không thay phiên remote.

Archive restore manifest đã bỏ mục FF++ bị xóa; các ảnh mask GenFace cũ còn lưu trong archive vẫn có thể khôi phục. Log danh sách đường dẫn/số byte: `docs/unused_dataset_deletion_20261027.json`.

Xác minh sau thay đổi: `tests/pilot` còn **48 test pass (43,57 giây)** với plugin tự động/cache pytest tắt. Đã kiểm tra đường dẫn xóa không còn tồn tại và restore manifest chỉ chứa phần còn giữ. Không commit/push.
