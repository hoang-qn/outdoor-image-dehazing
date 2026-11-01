# Phạm vi nghiên cứu

## Bài toán

Nghiên cứu khử sương mù cho ảnh ngoài trời theo hướng supervised image-to-image, với đầu vào và mục tiêu là các cặp ảnh hazy/clear. Các thí nghiệm xem xét cả dữ liệu tổng hợp và khả năng tổng quát hóa sang ảnh sương mù ngoài trời thực tế.

## Dữ liệu và vai trò của các tập

- RESIDE-6K Outdoor cung cấp các split train, validation và test cho baseline.
- NH-HAZE2 và HD-NH-HAZE có thể được thêm vào train/validation trong các thí nghiệm đa nguồn; manifest trong data/ xác định các cặp và split.
- O-HAZE và NH-HAZE được dùng làm tập đánh giá ngoài miền.
- Nguồn dữ liệu và split đang hoạt động được khai báo trong từng tệp configs/experiment/*.yaml.

Các tập validation dùng để chọn checkpoint và cấu hình. Tập test nội bộ và các tập ngoài miền dành cho đánh giá sau khi cấu hình được cố định. Dữ liệu gốc và các biến thể augmentation được mô tả riêng trong [data/README.md](../data/README.md).

## Mô hình và yếu tố thí nghiệm

DehazeFormer-T là baseline chính. Các thí nghiệm mở rộng gồm các ablation kiến trúc M1–M9, DEA-Net, SLRNet và WLD-Net. Những yếu tố dữ liệu và huấn luyện được khảo sát gồm haze augmentation ngoại tuyến/trực tuyến, conditioning bằng depth map, tổng hợp haze MCBM và các hàm mất mát bổ trợ.

Cấu hình của từng thí nghiệm ghi rõ kiến trúc, dữ liệu, loss, optimizer, lịch học, seed và tiêu chí chọn checkpoint. Những phương pháp dùng cấu hình huấn luyện khác nhau được trình bày như các kết quả riêng, không xem là ablation có kiểm soát hoàn toàn.

## Đánh giá

PSNR và SSIM được tính trên ảnh trong dải [0, 1] và báo cáo riêng cho từng tập. Các báo cáo cũng ghi số ảnh đánh giá, cách suy luận tiled và giới hạn của phép đo GFLOPs/FPS. Kết quả pilot được phân biệt với các run dài hơn theo ngân sách huấn luyện.

## Tái lập

Mã nguồn, cấu hình và script nằm trong repository. Dataset, checkpoint, log chi tiết và ảnh dự đoán được tạo hoặc cung cấp riêng; các thư mục lớn này không được commit vào Git mặc định.