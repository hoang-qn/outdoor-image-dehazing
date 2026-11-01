# Quy trình thực nghiệm

## Định nghĩa thí nghiệm

Mỗi thí nghiệm được mô tả bằng một tệp YAML riêng trong configs/experiment/. Cấu hình xác định dữ liệu và split, kiến trúc, khởi tạo, loss, optimizer, scheduler, seed, kích thước crop, augmentation, tiêu chí dừng và đầu ra.

Các cặp hazy/clear được xác định bằng manifest hoặc quy tắc ghép cặp của dataset. Nguồn dữ liệu và split được ghi rõ để tránh trộn mẫu giữa train, validation và test.

## Huấn luyện và chọn checkpoint

Train metrics được ghi theo epoch cùng cấu hình hiệu lực của lần chạy. Checkpoint tốt nhất được chọn theo metric validation đã khai báo; dữ liệu test không tham gia chọn checkpoint hay siêu tham số.

Run outputs có thể bao gồm cấu hình hiệu lực, metrics, checkpoint và dự đoán. Seed và trạng thái xác định của bộ sinh số ngẫu nhiên được ghi trong cấu hình hoặc metadata khi có hỗ trợ.

## Đánh giá

Đánh giá cuối được thực hiện sau khi cấu hình được cố định. RESIDE test, O-HAZE và NH-HAZE được báo cáo riêng, cùng số lượng ảnh và cách xử lý độ phân giải. PSNR/SSIM sử dụng cùng dải ảnh và quy tắc xử lý trong các phương pháp được so sánh.

Các số đo hiệu năng ghi kèm phần cứng, precision, batch size, kích thước đầu vào hoặc tile, overlap, warm-up và phạm vi thời gian được tính. Ước lượng GFLOPs nêu rõ input size và các operator mà công cụ đo không hỗ trợ.

## Lưu trữ kết quả

Mỗi run ghi cấu hình, metrics, checkpoint và dự đoán dưới runs/<experiment_name>/. Dataset, checkpoint và output lớn là dữ liệu dẫn xuất hoặc đầu vào cục bộ, không được commit vào Git mặc định. Bảng và hình tóm tắt có thể được lưu trong results/ cùng với thiết lập đánh giá và các giới hạn liên quan.