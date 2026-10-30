# Dọn thư mục làm việc ngày 27/10/2026

**Cập nhật sau đó:** người dùng yêu cầu xóa vĩnh viễn phần FF++. ZIP, ảnh, checkpoint và output FF++ trong project/archive đã xóa; chúng không còn khôi phục được từ archive. Danh sách bên dưới mô tả lần lưu trữ trước khi xóa. `manifest.json` đã loại các mục FF++; script restore chỉ áp dụng cho phần còn lại. Xem `xoa_dataset_khong_dung_20261027.md`.

Đã chuyển **13.906 file, 23.206.737.195 byte (23,21 GB / 21,61 GiB)** ra ngoài MFVLR vào:

`D:\New folder (2)\_archive\MFVLR_cleanup_20261027_01`

Đây là lưu trữ có thể khôi phục trên cùng ổ đĩa, chưa giải phóng dung lượng ổ đĩa. Không xóa vĩnh viễn.

## Đã lưu trữ

- `FaceForensics++.zip` (khoảng 22,61 GB).
- Ảnh và mask FF++ đã trích xuất tại `artifacts/ffpp_pilot/preparation/images` và `masks`. Manifest, metadata, báo cáo và checkpoint vẫn giữ nguyên; cần khôi phục hai thư mục này trước khi chạy lại các config FF++ cũ.
- Ảnh xuất lẻ `comparison`, `ground_truth`, `overlay`, `predicted` của ba lượt Diffae cũ và hai lượt FF++ cũ. Giữ `preview.jpg`, `index.csv`, `export.json` và metric ở nơi cũ. Các đường dẫn ảnh lẻ trong index/export cũ nay cần đối chiếu archive hoặc khôi phục.
- `output/pdf/qa` là thư mục QA/phụ thuộc sinh ra khi làm PDF khác; giữ nguyên PDF cuối `output/pdf/Phan_cong_Goi_4_N09.pdf` và các script gốc.

Không thay source/config đã có, paper PDF, tài liệu thảo luận, checkpoint, manifest GenFace, bundle lịch sử, `.pilot_deps`, `tmp/kaggle_current_deps` hoặc dữ liệu nguồn `D:\BaiduNetdiskDownload`. Không thay artifact/log của lượt Kaggle đang chạy. Các cache có quyền truy cập hạn chế được bỏ qua.

## Tra cứu và khôi phục

Archive giữ nguyên cây đường dẫn tương đối; `manifest.json` ở gốc archive ghi vị trí cũ/mới, số file và số byte của từng mục (24 mục).

Xem kế hoạch khôi phục toàn bộ, chưa di chuyển:

```powershell
.\tools\restore_workspace_archive.ps1
```

Khôi phục toàn bộ khi cần:

```powershell
.\tools\restore_workspace_archive.ps1 -Apply
```

Khôi phục riêng một thư mục ảnh:

```powershell
.\tools\restore_workspace_archive.ps1 -RelativePath 'artifacts\diffae_pilot\runs\pilot_005\masks_test\predicted' -Apply
```

Script từ chối ghi đè đích đã tồn tại và kiểm tra đường dẫn nằm trong project/archive. Nếu đã khôi phục một phần, chọn từng mục còn lại bằng `-RelativePath` thay vì dùng lại lệnh toàn bộ. Không commit/push trong lần dọn này.
