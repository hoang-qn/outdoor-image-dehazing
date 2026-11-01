# Tổng hợp đánh giá các checkpoint khử sương

Báo cáo tổng hợp kết quả đánh giá đã lưu trong project trên RESIDE-6K-Outdoor, O-HAZE và NH-HAZE. Chữ đậm đánh dấu giá trị tốt nhất của từng chỉ số trong mỗi bảng test.

## Thiết lập đánh giá

- Mỗi dòng tương ứng một thí nghiệm/checkpoint. Kết quả được tổng hợp từ các summary.json của từng run.
- M1, M2, M3 dùng best.pth sau khi train đủ 50 epoch trên RESIDE-6K-Outdoor + NH-HAZE2 + HD-NH-HAZE.
- M6 + MCBM train đủ 50 epoch trên cùng ba tập; đánh giá best.pth ở epoch 47 (validation PSNR 29.1137 dB). Loss là L1 + 0.05 × loss nhất quán tán xạ khí quyển. MCBM tạo thêm một mẫu tổng hợp cho mỗi cặp train: 2.818 cặp gốc + 2.818 mẫu MCBM = 5.636 mẫu/epoch; validation dùng 297 cặp gốc.
- M6 + MCBM dùng batch 16, crop 256, AMP, shuffle tự nhiên, seed 42; thời gian train và validation đủ 50 epoch là 3 giờ 13 phút 24 giây, chưa tính chuẩn bị depth/cache và tuning. Batch size của M6 + MCBM là 16, còn các pilot và M1–M3 dùng batch size 8; khác biệt này hạn chế so sánh trực tiếp thời gian train và mức sử dụng bộ nhớ.
- SLRNet và WLD-Net được train 50 epoch trên cùng split ba nguồn (batch 8); best.pth được chọn theo validation PSNR. Cả hai có kết quả test trên ba tập.
- RESIDE-6K-Outdoor: test, 500 ảnh. O-HAZE: toàn bộ 45 ảnh. NH-HAZE: toàn bộ 55 ảnh.
- PSNR/SSIM tính trên ảnh đầy đủ. Suy luận dùng AMP, batch 1, tile 512×512, overlap 64. FPS theo dataset trong summary tính thời gian chuyển tile, forward và ghép tile, không tính đọc/lưu ảnh hay metric.
- GFLOPs được fvcore ước lượng ở 256×256; một số operator không được fvcore đếm nên con số có thể thấp hơn FLOPs thực tế.
- FPS cố định đo trên RTX 3060, FP32 (TF32 tắt), batch 16, input 512×512, 50 warm-up và 200 batch đo. Input ở sẵn trên GPU; phép đo chỉ tính model forward.

## RESIDE-6K-Outdoor — test (500 ảnh)

| Thí nghiệm | PSNR (dB) ↑ | SSIM ↑ |
| --- | ---: | ---: |
| DehazeFormer-T baseline (RESIDE) | 28.3316 | 0.9443 |
| DEA-Net (50 epoch) | **29.5746** | **0.9553** |
| SLRNet (50 epoch) | 25.7201 | 0.9269 |
| WLD-Net (50 epoch) | 27.9401 | 0.9402 |
| Baseline (RESIDE + NH-HAZE2 + HD-NH-HAZE) | 28.5163 | 0.9484 |
| M1 — SCAF (50 epoch) | 28.5517 | 0.9488 |
| M2 — M1 + haze guidance (50 epoch) | 28.6895 | 0.9499 |
| M3 — M2 + Haar detail (50 epoch) | 28.7758 | 0.9505 |
| M6 + MCBM (RESIDE + NH-HAZE2 + HD-NH-HAZE, 50 epoch) | 28.2152 | 0.9437 |

## O-HAZE — toàn bộ (45 ảnh)

| Thí nghiệm | PSNR (dB) ↑ | SSIM ↑ |
| --- | ---: | ---: |
| DehazeFormer-T baseline (RESIDE) | 16.0728 | 0.6384 |
| DEA-Net (50 epoch) | 16.0991 | 0.6402 |
| SLRNet (50 epoch) | 15.8095 | 0.6321 |
| WLD-Net (50 epoch) | 19.4347 | 0.6466 |
| Baseline (RESIDE + NH-HAZE2 + HD-NH-HAZE) | **19.9787** | **0.6750** |
| M1 — SCAF (50 epoch) | 19.8262 | 0.6695 |
| M2 — M1 + haze guidance (50 epoch) | 19.8071 | 0.6711 |
| M3 — M2 + Haar detail (50 epoch) | 19.8545 | 0.6726 |
| M6 + MCBM (RESIDE + NH-HAZE2 + HD-NH-HAZE, 50 epoch) | 18.0230 | 0.6365 |

## NH-HAZE — toàn bộ (55 ảnh)

| Thí nghiệm | PSNR (dB) ↑ | SSIM ↑ |
| --- | ---: | ---: |
| DehazeFormer-T baseline (RESIDE) | 12.1761 | 0.4824 |
| DEA-Net (50 epoch) | 12.1485 | 0.4803 |
| SLRNet (50 epoch) | 12.1725 | 0.4871 |
| WLD-Net (50 epoch) | 14.1831 | 0.5495 |
| Baseline (RESIDE + NH-HAZE2 + HD-NH-HAZE) | 14.3478 | 0.5170 |
| M1 — SCAF (50 epoch) | 14.3278 | 0.5142 |
| M2 — M1 + haze guidance (50 epoch) | 14.2038 | 0.5153 |
| M3 — M2 + Haar detail (50 epoch) | 14.3824 | 0.5156 |
| M6 + MCBM (RESIDE + NH-HAZE2 + HD-NH-HAZE, 50 epoch) | **15.2468** | **0.5721** |

## Tham số, GFLOPs và FPS cố định

| Thí nghiệm | Tham số (M) | GFLOPs / 256² | FPS FP32 (batch 16, 512²) |
| --- | ---: | ---: | ---: |
| DehazeFormer-T baseline (RESIDE) | 0.6874 | 6.6455 | 16.0840 |
| DEA-Net (50 epoch) | 3.6534 | 32.2306 | 15.8937 |
| SLRNet (50 epoch) | 0.0965 | 6.2495 | 67.0210 |
| WLD-Net (50 epoch) | 0.3858 | 2.4391 | 69.5856 |
| Baseline (RESIDE + NH-HAZE2 + HD-NH-HAZE) | 0.6874 | 6.6455 | 15.9368 |
| M1 — SCAF (50 epoch) | 0.6909 | 6.7453 | 14.8090 |
| M2 — M1 + haze guidance (50 epoch) | 0.7117 | 6.8303 | 14.0681 |
| M3 — M2 + Haar detail (50 epoch) | 0.7121 | 6.8445 | 15.0170 |
| M6 + MCBM (RESIDE + NH-HAZE2 + HD-NH-HAZE, 50 epoch) | 0.7106 | 6.6455 | 15.9411 |

GFLOPs là ước lượng fvcore; operator không được hỗ trợ không được tính. M6 có hai nhánh phụ T/A chỉ chạy khi yêu cầu output phụ để tính loss lúc train. Số tham số bao gồm hai nhánh này; GFLOPs/FPS ở đây đo đường suy luận tạo ảnh sạch, không biểu thị chi phí train.

## Nhận xét

- RESIDE-6K-Outdoor: DEA-Net có PSNR và SSIM cao nhất trong nhóm bảng.
- O-HAZE: baseline train trên RESIDE + NH-HAZE2 + HD-NH-HAZE có PSNR và SSIM cao nhất.
- NH-HAZE: M6 + MCBM có PSNR và SSIM cao nhất.
- So với baseline train trên cùng ba nguồn, M6 + MCBM thay đổi RESIDE-6K-Outdoor −0.3011 dB PSNR / −0.0047 SSIM, O-HAZE −1.9558 dB / −0.0385, và NH-HAZE +0.8991 dB / +0.0551.

## Phạm vi báo cáo

Các lần đo lặp trong runs/fps_dataset_evaluation/ trùng với số liệu chính nên không được trình bày thành dòng riêng.

## Nguồn số liệu

Các summary.json và metadata checkpoint được tạo trong thư mục runs/ cục bộ và không được commit vào Git. Các đường dẫn dưới đây nhận diện artifact nguồn; bảng trong tài liệu lưu bản tóm tắt các metric.

- DehazeFormer-T baseline (RESIDE) — `runs/dehazeformer_t/evaluation/summary.json`
- SLRNet (50 epoch) — `runs/slrnet_recent_full50/evaluation_3tests/summary.json`
- WLD-Net (50 epoch) — `runs/wldnet_recent_full50/evaluation_3tests/summary.json`
- DEA-Net (50 epoch) — `runs/dea_net_fp32_50ep/evaluation/summary.json`
- Baseline đa nguồn — `runs/dehazeformer_t_reside_nhhaze2_hdnhhaze_bs8_tiles512_w0_clampfix/evaluation/summary.json`
- M1 — SCAF — `runs/dehazeformer_t_m1_full50/evaluation_full50/summary.json`
- M2 — M1 + haze guidance — `runs/dehazeformer_t_m2_full50/evaluation_full50/summary.json`
- M3 — M2 + Haar detail — `runs/dehazeformer_t_m3_full50/evaluation_full50/summary.json`
- M6 + MCBM — `runs/dehazeformer_t_m6_mcbm_full50/evaluation_full50/summary.json`
- M6 + MCBM checkpoint metadata — `runs/dehazeformer_t_m6_mcbm_full50/evaluation_full50/checkpoint_metadata.json`