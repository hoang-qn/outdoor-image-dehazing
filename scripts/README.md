# Scripts

Các script chuẩn bị dữ liệu, chạy thí nghiệm và phân tích kết quả nằm trong thư mục này. Logic huấn luyện và đánh giá dùng chung được triển khai trong src/dehazing/.

## Lệnh package

```bash
python -m dehazing.train --config configs/experiment/baseline.yaml
python -m dehazing.evaluate --checkpoint runs/dehazeformer_t/checkpoints/best.pth --dataset all
```

## PowerShell launchers

start_training.ps1 hỗ trợ chạy train nền và ghi log vào thư mục run tương ứng. training_status.ps1 hiển thị trạng thái và log; stop_training.ps1 yêu cầu dừng sau epoch hiện tại.

```powershell
./scripts/start_training.ps1 -Config configs/experiment/baseline.yaml -RunName dehazeformer_t
./scripts/training_status.ps1 -RunName dehazeformer_t
./scripts/stop_training.ps1 -RunName dehazeformer_t
```

Các launcher cho suite thí nghiệm và từng pipeline cụ thể nằm trong cùng thư mục. Cấu hình YAML và tên run là tham số đầu vào; log, metrics, checkpoint và trạng thái được lưu dưới runs/.

## Chuẩn bị dữ liệu

prepare_training_cache.py, generate_depth_maps.py và generate_white_haze_density.py tạo cache hoặc biến thể dữ liệu dẫn xuất. Các tệp được tạo lưu trong data/ và không được commit vào Git mặc định. Cách bố trí dataset và manifest được mô tả trong [data/README.md](../data/README.md).