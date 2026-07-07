# Notebooks for adaptive-fusion-ts2img

Bộ notebook này chỉ dùng để điều khiển chạy lệnh trong Colab. Logic huấn luyện, batch run, gom kết quả, xuất bảng, xếp hạng và kiểm định thống kê nằm trong `src/`, `config/` và `config/suites/`.

## Thứ tự chạy đề xuất

1. `00_setup_check_colab.ipynb` — kiểm tra môi trường, cài thư viện, kiểm tra CLI.
2. `01_stage1_smoke_test.ipynb` — chạy thử 5 dataset, 3 seed, all methods.
3. `02_stage2_main_benchmark.ipynb` — chạy benchmark chính nếu đã có YAML Stage 2.
4. `03_stage3_ablation_study.ipynb` — chạy ablation nếu đã có YAML Stage 3.
5. `04_stage4_efficiency_analysis.ipynb` — chạy efficiency nếu đã có YAML Stage 4.
6. `05_collect_tables_rank_statistics.ipynb` — gom kết quả, xuất bảng, rank và kiểm định thống kê.

## Đường dẫn mặc định

Các notebook mặc định dùng project ở:

```text
/content/adaptive-fusion-ts2img
```

Nếu project đặt trong Google Drive, đổi cell `%cd` thành:

```text
%cd /content/drive/MyDrive/adaptive-fusion-ts2img
```

Kết quả mặc định lưu tại:

```text
/content/drive/MyDrive/adaptive-fusion-ts2img/outputs
```

## Lưu ý

Nếu Stage 2, Stage 3 hoặc Stage 4 báo không tìm thấy file YAML, cần bổ sung các file trong `config/suites/` trước khi chạy các notebook tương ứng.