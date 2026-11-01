# Đánh giá pilot 12 epoch: baseline và M1–M9

Báo cáo so sánh 10 checkpoint (baseline và M1–M9) trên ba tập test. Các checkpoint được lấy từ best.pth của từng run pilot. Pilot lấy mẫu cân bằng đã dừng sau epoch 1 nên không nằm trong báo cáo. Chữ đậm đánh dấu giá trị tốt nhất của từng chỉ số trong mỗi tập.

## Thiết lập đánh giá

- RESIDE-6K-Outdoor: split test, 500 ảnh; O-HAZE: 45 ảnh; NH-HAZE: 55 ảnh.
- PSNR/SSIM đo trên ảnh độ phân giải gốc. Suy luận tiled với tile 512×512, overlap 64, AMP, batch 1.
- FPS theo tập tính thời gian xử lý một ảnh độ phân giải gốc, gồm chuyển tile, model forward và ghép tile; không gồm đọc/ghi file và tính metric. Do độ phân giải ảnh giữa các tập khác nhau, FPS này có thể khác nhau.
- Benchmark FPS cố định: RTX 3060, FP32 (TF32 tắt), batch 16, input 512×512, 50 warm-up và 200 lượt đo; chỉ tính model forward, input nằm sẵn trên GPU.
- GFLOPs được ước lượng bằng fvcore tại input 256×256. fvcore bỏ qua operator không được hỗ trợ; M8/M9 có thêm operator của DCP positional embedding (ví dụ amin, max-pool, sin/cos, exp) không được đếm, vì vậy GFLOPs của hai model này đặc biệt là giá trị thiếu và không phản ánh đầy đủ chi phí tính toán.
- Đây là pilot 12 epoch dùng để sàng lọc biến thể, không phải so sánh cuối cùng ngang ngân sách với các run 50 epoch.

## RESIDE-6K-Outdoor — test (500 ảnh)

| Thí nghiệm | PSNR (dB) ↑ | SSIM ↑ | FPS theo tập ↑ |
| --- | ---: | ---: | ---: |
| Baseline DehazeFormer-T | 27.21836 | 0.93302 | 9.5291 |
| M1 — SCAF | 27.14766 | 0.93360 | 8.7214 |
| M2 — M1 + haze guidance | 27.33174 | 0.93487 | 8.8108 |
| M3 — M2 + Haar detail | 27.13049 | 0.93300 | 8.6671 |
| M4 — Haar wavelet, 2 levels | 27.16613 | 0.93252 | 8.7704 |
| M5 — M2 + haze-map skip guidance | **27.41231** | 0.93507 | 8.1813 |
| M6 — atmospheric-scattering consistency loss | 27.23476 | 0.93494 | **9.5524** |
| M7 — decoder-conditioned skip modulation | 27.35395 | **0.93514** | 9.3232 |
| M8 — DCP-based 3D positional embedding | 27.26082 | 0.93243 | 9.2781 |
| M9 — M7 + M8 | 27.31590 | 0.93328 | 9.0725 |

## O-HAZE — all (45 ảnh)

| Thí nghiệm | PSNR (dB) ↑ | SSIM ↑ | FPS theo tập ↑ |
| --- | ---: | ---: | ---: |
| Baseline DehazeFormer-T | 17.89437 | 0.63680 | 0.3241 |
| M1 — SCAF | 17.68900 | 0.64624 | 0.3001 |
| M2 — M1 + haze guidance | 18.05261 | 0.63896 | 0.2993 |
| M3 — M2 + Haar detail | 17.88408 | 0.64070 | 0.3018 |
| M4 — Haar wavelet, 2 levels | 18.26895 | 0.64172 | 0.3005 |
| M5 — M2 + haze-map skip guidance | 17.78255 | 0.64585 | 0.2805 |
| M6 — atmospheric-scattering consistency loss | 17.80309 | **0.64810** | **0.3244** |
| M7 — decoder-conditioned skip modulation | 17.76213 | 0.64706 | 0.3182 |
| M8 — DCP-based 3D positional embedding | **18.30199** | 0.63192 | 0.3157 |
| M9 — M7 + M8 | 17.93549 | 0.64618 | 0.3096 |

## NH-HAZE — all (55 ảnh)

| Thí nghiệm | PSNR (dB) ↑ | SSIM ↑ | FPS theo tập ↑ |
| --- | ---: | ---: | ---: |
| Baseline DehazeFormer-T | 12.83988 | 0.48385 | 1.7946 |
| M1 — SCAF | 12.77432 | 0.48733 | 1.6402 |
| M2 — M1 + haze guidance | 12.87507 | 0.48675 | 1.6430 |
| M3 — M2 + Haar detail | 12.87089 | 0.48697 | 1.6643 |
| M4 — Haar wavelet, 2 levels | 12.98028 | 0.48741 | 1.6577 |
| M5 — M2 + haze-map skip guidance | 12.79134 | 0.48607 | 1.5339 |
| M6 — atmospheric-scattering consistency loss | 12.84994 | 0.48882 | **1.7956** |
| M7 — decoder-conditioned skip modulation | 12.80122 | **0.48889** | 1.7468 |
| M8 — DCP-based 3D positional embedding | **13.04139** | 0.48388 | 1.7478 |
| M9 — M7 + M8 | 12.88062 | 0.48812 | 1.7149 |

## Kích thước mô hình và benchmark cố định

| Thí nghiệm | Tham số (M) | GFLOPs / 256²* | FPS FP32 (batch 16, 512²) |
| --- | ---: | ---: | ---: |
| Baseline DehazeFormer-T | 0.687404 | 6.64548 | **15.9170** |
| M1 — SCAF | 0.690884 | 6.74529 | 14.7807 |
| M2 — M1 + haze guidance | 0.711670 | 6.83033 | 14.7761 |
| M3 — M2 + Haar detail | 0.712127 | 6.84448 | 15.1114 |
| M4 — Haar wavelet, 2 levels | 0.700197 | 6.78390 | 15.0350 |
| M5 — M2 + haze-map skip guidance | 0.712246 | 6.84009 | 12.2393 |
| M6 — atmospheric-scattering consistency loss | 0.710592 | 6.64548 | 15.9013 |
| M7 — decoder-conditioned skip modulation | 0.693308 | 6.79648 | 15.3733 |
| M8 — DCP-based 3D positional embedding | 0.687410 | 6.64561 | 15.5383 |
| M9 — M7 + M8 | 0.693314 | 6.79661 | 15.0471 |

* GFLOPs là ước lượng fvcore; operator không được hỗ trợ không được tính. M8/M9 còn có các toán tử DCP không được đếm, như đã nêu ở phần thiết lập.

## Kết quả nổi bật

- RESIDE: M5 có PSNR cao nhất; M7 có SSIM cao nhất; M6 có FPS theo tập cao nhất.
- O-HAZE: M8 có PSNR cao nhất; M6 có SSIM và FPS theo tập cao nhất.
- NH-HAZE: M8 có PSNR cao nhất; M7 có SSIM cao nhất; M6 có FPS theo tập cao nhất.
- Benchmark cố định cho thấy baseline có FPS cao nhất trong nhóm 10 model. FPS này đo batch 16 ở 512×512 nên không đồng nhất với FPS theo tập.

## File kết quả nguồn

Các summary.json được tạo trong thư mục runs/ cục bộ và không được commit vào Git. Đường dẫn bên dưới nhận diện artifact nguồn; bảng trong tài liệu lưu bản tóm tắt các metric.

- Baseline DehazeFormer-T — summary.json — `runs/dehazeformer_t_reside_nhhaze2_hdnhhaze_pilot12/evaluation_pilot12_3tests/summary.json`
- M1 — SCAF — summary.json — `runs/dehazeformer_t_m1_pilot12/evaluation_pilot12_3tests/summary.json`
- M2 — M1 + haze guidance — summary.json — `runs/dehazeformer_t_m2_pilot12/evaluation_pilot12_3tests/summary.json`
- M3 — M2 + Haar detail — summary.json — `runs/dehazeformer_t_m3_pilot12/evaluation_pilot12_3tests/summary.json`
- M4 — Haar wavelet, 2 levels — summary.json — `runs/dehazeformer_t_m4_pilot12/evaluation_pilot12_3tests/summary.json`
- M5 — M2 + haze-map skip guidance — summary.json — `runs/dehazeformer_t_m5_pilot12/evaluation_pilot12_3tests/summary.json`
- M6 — atmospheric-scattering consistency loss — summary.json — `runs/dehazeformer_t_m6_pilot12/evaluation_pilot12_3tests/summary.json`
- M7 — decoder-conditioned skip modulation — summary.json — `runs/dehazeformer_t_m7_pilot12/evaluation_pilot12_3tests/summary.json`
- M8 — DCP-based 3D positional embedding — summary.json — `runs/dehazeformer_t_m8_pilot12/evaluation_pilot12_3tests/summary.json`
- M9 — M7 + M8 — summary.json — `runs/dehazeformer_t_m9_pilot12/evaluation_pilot12_3tests/summary.json`
