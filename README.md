# Adaptive Fusion TS2Img

Adaptive Fusion TS2Img là mã nguồn nghiên cứu cho bài toán **phân loại chuỗi thời gian** bằng cách chuyển chuỗi 1D thành nhiều biểu diễn ảnh 2D, sau đó kết hợp các biểu diễn này bằng cơ chế **adaptive gating / attention fusion**.

Pipeline chính:

```text
1D time series
    -> GAF / MTF / RP / STFT
    -> lightweight CNN branches
    -> adaptive gating fusion
    -> classifier
```

Repository được tổ chức theo hướng phục vụ thực nghiệm bài báo: notebook chỉ dùng để gọi lệnh, logic nghiên cứu nằm trong `src/`, tham số thí nghiệm nằm trong các file YAML ở `config/`.

---

## 1. Tuyên bố miễn trừ trách nhiệm

Mã nguồn này được cung cấp cho mục đích **nghiên cứu, học thuật và tái lập thực nghiệm**. Tác giả không bảo đảm rằng mã nguồn sẽ chạy thành công trong mọi môi trường hoặc tạo ra kết quả giống hệt trên mọi máy.

Kết quả thực nghiệm có thể khác nhau do nhiều nguyên nhân, bao gồm nhưng không giới hạn ở:

- khác biệt về phiên bản Python, PyTorch, CUDA, cuDNN, scikit-learn, pyts hoặc các thư viện phụ thuộc;
- khác biệt giữa CPU, GPU, RAM, hệ điều hành, Google Colab, máy local hoặc máy chủ tính toán;
- dữ liệu UCR/UEA không đúng định dạng, thiếu file, sai tên thư mục hoặc sai phân chia train/test;
- sai khác do seed, thuật toán GPU không hoàn toàn deterministic, batch size, early stopping hoặc checkpoint resume;
- thay đổi từ nền tảng Google Colab, Google Drive, GitHub hoặc hệ thống lưu trữ bên ngoài;
- cấu hình thí nghiệm bị chỉnh sửa so với file YAML gốc.

Người sử dụng có trách nhiệm tự kiểm tra môi trường, dữ liệu, đường dẫn, giấy phép sử dụng dữ liệu, phiên bản thư viện và kết quả đầu ra trước khi sử dụng trong bài báo, báo cáo, đồ án hoặc hệ thống thực tế.

Mã nguồn này **không phải phần mềm thương mại**, **không phải hệ thống sản xuất**, và **không đi kèm cam kết hỗ trợ, bảo hành hoặc đảm bảo kết quả công bố**. Nếu người dùng không chạy được hoặc không tái lập được kết quả, cần kiểm tra lại dữ liệu, môi trường, file cấu hình và log lỗi trước khi kết luận về phương pháp.

---

## 2. Cấu trúc thư mục

```text
adaptive-fusion-ts2img/
├── config/
│   ├── default.yaml
│   ├── experiments/      # cấu hình từng thí nghiệm đơn
│   ├── baselines/        # cấu hình baseline: 1D-CNN, single 2D, manual fusion
│   ├── ablations/        # cấu hình ablation bỏ từng nhánh biểu diễn
│   └── suites/           # cấu hình chạy nhiều dataset, seed, method theo từng stage
├── data/
│   └── UCR/              # nơi đặt dữ liệu UCR/UEA, không đẩy dữ liệu lên GitHub
├── docs/
│   └── EXPERIMENT_STAGES.md
├── notebooks/
│   ├── 01_colab_pipeline_commands_only.ipynb
│   ├── 02_local_pipeline_commands_only.ipynb
│   └── README.md
├── src/
│   ├── cli/              # các lệnh chạy, gom kết quả, kiểm định thống kê
│   ├── data/             # đọc dữ liệu UCR
│   ├── datasets/         # PyTorch Dataset
│   ├── models/           # AdaptiveFusionCNN, CNN1D
│   ├── transforms/       # GAF, MTF, RP, STFT
│   └── utils/            # config, seed, metric, experiment utilities
├── outputs/              # kết quả sinh ra, bị ignore bởi git
├── cache/                # cache ảnh 2D, bị ignore bởi git
├── scripts/
│   └── git_push_commands.txt
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 3. Notebook nên dùng

Nếu chạy trên Google Colab, dùng:

```text
notebooks/01_colab_pipeline_commands_only.ipynb
```

Nếu chạy trên máy cá nhân bằng Visual Studio Code hoặc Visual Studio, dùng:

```text
notebooks/02_local_pipeline_commands_only.ipynb
```

Notebook chỉ nên dùng để gọi lệnh. Không nên đặt logic xử lý dữ liệu, định nghĩa mô hình hoặc train loop trực tiếp trong notebook.

---

## 4. Cài đặt môi trường local

Tạo môi trường ảo:

```bash
python -m venv .venv
```

Kích hoạt trên Windows:

```bash
.venv\Scripts\activate
```

Cài thư viện:

```bash
pip install -r requirements.txt
```

Kiểm tra nhanh source code:

```bash
python -m compileall src
```

---

## 5. Cấu trúc dữ liệu UCR/UEA

Dữ liệu không được đẩy lên GitHub. Hãy đặt dữ liệu trong thư mục `data/UCR/` theo cấu trúc:

```text
data/UCR/Coffee/Coffee_TRAIN.tsv
data/UCR/Coffee/Coffee_TEST.tsv

data/UCR/ECG200/ECG200_TRAIN.tsv
data/UCR/ECG200/ECG200_TEST.tsv

data/UCR/GunPoint/GunPoint_TRAIN.tsv
data/UCR/GunPoint/GunPoint_TEST.tsv
```

Mỗi file `.tsv` cần có dạng:

```text
label    value_1    value_2    value_3    ...    value_T
```

Trong đó cột đầu là nhãn lớp, các cột còn lại là giá trị chuỗi thời gian.

---

## 6. Chạy một thí nghiệm đơn

Ví dụ chạy mô hình adaptive fusion trên dataset Coffee:

```bash
python -m src.train --config config/experiments/coffee_adaptive_fusion.yaml
```

Chạy các baseline:

```bash
python -m src.train --config config/baselines/cnn1d.yaml
python -m src.train --config config/baselines/single_gaf.yaml
python -m src.train --config config/baselines/single_mtf.yaml
python -m src.train --config config/baselines/single_rp.yaml
python -m src.train --config config/baselines/single_stft.yaml
python -m src.train --config config/baselines/manual_feature_concat.yaml
```

---

## 7. Chạy theo từng giai đoạn thực nghiệm

Nên chạy theo thứ tự từ nhỏ đến lớn.

### Stage 1 — Smoke test

Mục tiêu: kiểm tra pipeline, dữ liệu, cache, checkpoint và các baseline.

```bash
python -m src.cli.batch_run \
  --suite config/suites/stage1_smoke_5datasets_3seeds_all_methods.yaml
```

Chạy thử lệnh nhưng chưa train thật:

```bash
python -m src.cli.batch_run \
  --suite config/suites/stage1_smoke_5datasets_3seeds_all_methods.yaml \
  --dry-run
```

### Stage 2 — Pilot experiment

```bash
python -m src.cli.batch_run \
  --suite config/suites/stage2_pilot_12datasets_3seeds_all_methods.yaml
```

### Stage 3 — Paper-grade minimum

```bash
python -m src.cli.batch_run \
  --suite config/suites/stage3_paper_20datasets_3seeds_all_methods.yaml
```

### Stage 4 — Stronger Q1/Q2 scale

```bash
python -m src.cli.batch_run \
  --suite config/suites/stage4_strong_30datasets_5seeds_all_methods.yaml
```

hoặc chỉ mở rộng mô hình đề xuất trên nhiều dataset:

```bash
python -m src.cli.batch_run \
  --suite config/suites/stage4_strong_50datasets_3seeds_adaptive_only.yaml
```

---

## 8. Gom kết quả và phân tích

Gom tất cả kết quả thành một file CSV:

```bash
python -m src.cli.collect_results \
  --output-root outputs \
  --out-csv outputs/summary_all_runs.csv
```

Tính average rank:

```bash
python -m src.cli.rank_results \
  --results-csv outputs/summary_all_runs.csv \
  --metric test_macro_f1
```

Kiểm định thống kê:

```bash
python -m src.cli.statistical_tests \
  --results-csv outputs/summary_all_runs.csv \
  --metric test_macro_f1 \
  --proposed adaptive_fusion_full
```

Xuất bảng cho bài báo:

```bash
python -m src.cli.export_paper_tables \
  --results-csv outputs/summary_all_runs.csv \
  --out-dir outputs/paper_tables
```

---

## 9. Các file kết quả chính

Mỗi lần chạy sẽ sinh ra thư mục kết quả trong `outputs/`. Các file quan trọng gồm:

```text
summary.json                  # kết quả chính: Accuracy, Macro-F1, Precision, Recall
config_used.yaml              # cấu hình thực tế đã dùng cho lần chạy
environment.json              # thông tin môi trường chạy
best_model.pt                 # checkpoint tốt nhất
last_checkpoint.pt            # checkpoint để resume
history.csv                   # log theo epoch
classification_report.txt     # báo cáo phân loại
confusion_matrix.png          # ma trận nhầm lẫn
learning_curve_macro_f1.png   # đường học tập
complexity.json               # params, FLOPs nếu đo được
inference_time.json           # thời gian suy luận
alpha_mean.csv                # trọng số fusion trung bình
alpha_test_samples.csv        # trọng số fusion theo từng mẫu test
```

`config_used.yaml` và `environment.json` nên được lưu lại để phục vụ tái lập kết quả khi viết bài báo.

---

## 10. Resume khi bị ngắt

Nếu quá trình train bị ngắt, chạy lại đúng lệnh cũ:

```bash
python -m src.train --config config/experiments/coffee_adaptive_fusion.yaml
```

Chương trình sẽ kiểm tra `last_checkpoint.pt` trong thư mục output và tiếp tục nếu checkpoint tồn tại.

---

## 11. Quy tắc GitHub

Repository này chỉ nên lưu:

```text
source code
YAML config
notebooks command-only
documentation
requirements.txt
```

Không nên đẩy lên GitHub:

```text
data/UCR/*
outputs/*
cache/*
*.pt
*.pth
*.ckpt
```

Các mục này đã được khai báo trong `.gitignore`.

Lệnh đẩy code lần đầu:

```bash
git init
git add .
git commit -m "Initial adaptive fusion TS2Img project"
git branch -M main
git remote add origin https://github.com/<username>/adaptive-fusion-ts2img.git
git push -u origin main
```

Các lần cập nhật sau:

```bash
git status
git add .
git commit -m "Update experiment pipeline"
git push
```

---

## 12. Gợi ý xử lý lỗi thường gặp

Nếu báo thiếu dữ liệu, kiểm tra:

```text
data/UCR/<DatasetName>/<DatasetName>_TRAIN.tsv
data/UCR/<DatasetName>/<DatasetName>_TEST.tsv
```

Nếu Colab chạy chậm, nên đảm bảo code nằm ở `/content`, còn Google Drive chỉ dùng để lưu `outputs/` và `cache/`.

Nếu lỗi import module `src`, hãy chạy lệnh từ thư mục gốc của project:

```bash
cd adaptive-fusion-ts2img
python -m src.train --config config/experiments/coffee_adaptive_fusion.yaml
```

Nếu kết quả khác với lần chạy trước, kiểm tra:

```text
seed
config_used.yaml
phiên bản thư viện
GPU/CPU
checkpoint resume
dữ liệu train/test
```

---

## 13. Trích dẫn và sử dụng trong bài báo

Khi sử dụng mã nguồn này cho bài báo hoặc báo cáo, cần mô tả rõ:

```text
- nguồn dữ liệu;
- số dataset;
- số seed;
- cấu hình transform GAF/MTF/RP/STFT;
- kiến trúc mô hình;
- baseline so sánh;
- metric đánh giá;
- kiểm định thống kê;
- thông tin môi trường chạy.
```

Không nên công bố kết quả chỉ dựa trên một lần chạy đơn lẻ. Với bài báo chất lượng cao, nên dùng nhiều dataset, nhiều seed, baseline đầy đủ, kiểm định thống kê và phân tích ablation.
