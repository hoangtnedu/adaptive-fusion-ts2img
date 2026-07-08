# Adaptive Fusion TS2Img

[![Research](https://img.shields.io/badge/trạng_thái-research--prototype-blue)](#)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](#)
[![UCR/UEA](https://img.shields.io/badge/Dữ_liệu-UCR%2FUEA-lightgrey)](#)

**Adaptive Fusion TS2Img** là mã nguồn nghiên cứu cho bài toán **phân loại chuỗi thời gian** bằng cách chuyển chuỗi thời gian một chiều thành nhiều biểu diễn ảnh hai chiều, sau đó trích xuất đặc trưng bằng các nhánh CNN nhẹ và kết hợp đặc trưng bằng cơ chế **adaptive gating / attention fusion**.

Repository này được xây dựng để phục vụ thực nghiệm, tái lập kết quả và phát triển bài báo khoa học. Đây là **research prototype**, không phải phần mềm thương mại, không phải hệ thống production và không nên xem là một benchmark đã hoàn thiện tuyệt đối.

---

## 1. Định hướng nghiên cứu

Tên bài báo định hướng:

> **A Lightweight Adaptive Fusion Network of Two-Dimensional Representations for Time-Series Classification**

Ý tưởng chính: các phương pháp chuyển chuỗi thời gian sang ảnh 2D thường nhấn mạnh các đặc trưng tín hiệu khác nhau. Vì vậy, một biểu diễn ảnh cố định có thể tốt trên dataset này nhưng kém trên dataset khác.

| Biểu diễn | Thông tin nhấn mạnh | Trực giác sử dụng |
|---|---|---|
| **GAF / GASF / GADF** | Quan hệ góc, tương quan và hình dạng theo thời gian | Phù hợp khi hình dạng tổng thể và quan hệ giữa các thời điểm quan trọng |
| **MTF** | Xác suất chuyển trạng thái giữa các mức tín hiệu | Phù hợp khi chuỗi có mẫu chuyển trạng thái hoặc chế độ vận hành rõ |
| **RP** | Sự tái xuất hiện trạng thái trong không gian pha | Phù hợp với dữ liệu có chu kỳ, mẫu lặp hoặc động lực học phi tuyến |
| **STFT** | Phân bố năng lượng theo miền thời gian - tần số | Phù hợp với tín hiệu dao động, rung động, âm thanh hoặc y sinh |

Thay vì chọn thủ công một biểu diễn hoặc ghép cố định nhiều biểu diễn, mô hình trong project này hướng tới việc **học trọng số đóng góp của từng biểu diễn một cách thích nghi**.

---

## 2. Pipeline tổng quát

```text
Chuỗi thời gian 1D
    -> GAF / MTF / RP / STFT
    -> Các nhánh lightweight CNN
    -> Adaptive gating / attention fusion
    -> Bộ phân loại
    -> Accuracy, Macro-F1, Params, FLOPs, inference time, fusion weights
```

---

## 3. Mục tiêu của repository

Repository này được tổ chức theo hướng phục vụ bài báo nghiên cứu, gồm các mục tiêu sau:

1. Chạy kiểm thử pipeline trên một số dataset nhỏ để bảo đảm phần đọc dữ liệu, chuyển đổi ảnh, huấn luyện, checkpoint và lưu kết quả hoạt động đúng.
2. Chạy thí nghiệm pilot để quan sát xu hướng kết quả, lỗi phổ biến và tính ổn định của mô hình.
3. Chạy thí nghiệm mức bài báo với nhiều dataset, nhiều seed, baseline đầy đủ, kiểm định thống kê và ablation study.
4. Mở rộng lên quy mô mạnh hơn cho mục tiêu bài Q1/Q2, ưu tiên 30--50 dataset nếu tài nguyên tính toán cho phép.

Để sử dụng kết quả cho bài báo, không nên chỉ dựa vào một lần chạy hoặc vài dataset nhỏ. Tối thiểu cần báo cáo:

- Accuracy;
- Macro-F1;
- Precision và Recall nếu phù hợp;
- số tham số huấn luyện được;
- FLOPs;
- thời gian huấn luyện;
- thời gian suy luận trên mỗi mẫu;
- average rank trên nhiều dataset;
- kiểm định thống kê, ví dụ Friedman và Wilcoxon/Nemenyi;
- ablation study khi bỏ từng nhánh biểu diễn;
- phân tích trọng số fusion: `alpha_GAF`, `alpha_MTF`, `alpha_RP`, `alpha_STFT`.

---

## 4. Cấu trúc thư mục

```text
adaptive-fusion-ts2img/
├── config/
│   ├── default.yaml
│   ├── README.md
│   ├── experiments/          # cấu hình thí nghiệm đơn
│   ├── baselines/            # cấu hình baseline: 1D-CNN, single 2D, manual fusion
│   ├── ablations/            # cấu hình ablation bỏ từng nhánh biểu diễn
│   └── suites/               # cấu hình chạy theo stage, nhiều dataset, nhiều seed
├── data/
│   └── UCR/                  # nơi đặt dữ liệu UCR/UEA ở máy local hoặc Colab; không commit
├── docs/
│   └── EXPERIMENT_STAGES.md
├── notebooks/
│   ├── 01_colab_pipeline_commands_only.ipynb
│   ├── 02_local_pipeline_commands_only.ipynb
│   └── README.md
├── src/
│   ├── cli/                  # batch run, gom kết quả, xếp hạng, kiểm định thống kê
│   ├── data/                 # hàm đọc dữ liệu UCR/UEA
│   ├── datasets/             # PyTorch Dataset
│   ├── models/               # CNN1D, AdaptiveFusionCNN, các nhánh CNN nhẹ
│   ├── transforms/           # GAF, MTF, RP, STFT
│   └── utils/                # config, metric, seed, logging, tiện ích thực nghiệm
├── outputs/                  # kết quả sinh ra sau khi chạy; không commit
├── cache/                    # cache ảnh 2D; không commit
├── scripts/
│   └── git_push_commands.txt
├── requirements.txt
├── .gitignore
└── README.md
```

Nguyên tắc tổ chức:

- **Notebook chỉ dùng để gọi lệnh**.
- Logic xử lý dữ liệu, mô hình, train loop và phân tích kết quả nằm trong `src/`.
- Tham số thí nghiệm nằm trong các file YAML dưới `config/`.
- Dữ liệu, checkpoint, cache và output lớn không đưa lên GitHub.

---

## 5. Cài đặt môi trường local

### 5.1. Clone repository

```bash
git clone https://github.com/hoangtnedu/adaptive-fusion-ts2img.git
cd adaptive-fusion-ts2img
```

### 5.2. Tạo môi trường ảo

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 5.3. Cài thư viện

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Các nhóm thư viện chính:

- xử lý số liệu: `numpy`, `pandas`, `scipy`;
- học máy và metric: `scikit-learn`;
- xử lý ảnh: `scikit-image`;
- chuyển chuỗi thời gian sang ảnh: `pyts`;
- deep learning: `torch`, `torchvision`;
- đo độ phức tạp mô hình: `thop`;
- cấu hình và vẽ biểu đồ: `pyyaml`, `matplotlib`.

### 5.4. Kiểm tra nhanh source code

```bash
python -m compileall src
```

---

## 6. Chuẩn bị dữ liệu UCR/UEA

Dữ liệu UCR/UEA **không được lưu trực tiếp trong repository**. Người dùng cần tự tải dữ liệu và đặt vào thư mục `data/UCR/` theo đúng cấu trúc train/test chuẩn.

Ví dụ:

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

Trong đó:

- cột đầu tiên là nhãn lớp;
- các cột còn lại là giá trị của chuỗi thời gian;
- tên thư mục và tên file dataset cần khớp với tên trong file YAML.

Lưu ý: Trên Linux và Google Colab, tên thư mục và tên file có phân biệt chữ hoa/chữ thường.

---

## 7. Notebook nên dùng

Nếu chạy trên Google Colab, dùng:

```text
notebooks/01_colab_pipeline_commands_only.ipynb
```

Nếu chạy trên máy cá nhân bằng VS Code, Visual Studio hoặc terminal, dùng:

```text
notebooks/02_local_pipeline_commands_only.ipynb
```

Notebook trong project này nên giữ đúng vai trò **command-only notebook**. Không nên đưa logic mô hình hoặc train loop trực tiếp vào notebook.

---

## 8. Gợi ý chạy trên Google Colab

Quy trình khuyến nghị trên Colab:

```text
1. Clone hoặc pull code từ GitHub vào /content/adaptive-fusion-ts2img
2. Mount Google Drive
3. Lưu outputs, checkpoints, logs và cache vào Google Drive
4. Chạy lệnh từ thư mục source code ở /content/adaptive-fusion-ts2img
```

Ví dụ cấu trúc nên dùng:

```text
/content/adaptive-fusion-ts2img/              # source code clone từ GitHub
/content/drive/MyDrive/adaptive_fusion_runs/  # nơi lưu outputs, checkpoints, logs
/content/drive/MyDrive/ucr_data/              # nơi lưu dữ liệu UCR/UEA nếu muốn giữ trên Drive
```

Không nên sửa logic trực tiếp trong notebook Colab. Nếu cần sửa code, hãy sửa trong repository, commit lên GitHub, sau đó `git pull` lại trong Colab.

---

## 9. Chạy một thí nghiệm đơn

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
python -m src.train --config config/baselines/manual_feature_mean.yaml
```

Cần chạy lệnh từ thư mục gốc của repository:

```bash
cd adaptive-fusion-ts2img
```

Nếu gặp lỗi `ModuleNotFoundError: src`, thường là do đang chạy lệnh sai thư mục.

---

## 10. Chạy theo từng giai đoạn thực nghiệm

Nên chạy theo thứ tự từ nhỏ đến lớn. Không nên chạy thẳng Stage 3 hoặc Stage 4 khi Stage 1 chưa ổn định.

| Giai đoạn | Mục tiêu | File suite |
|---|---|---|
| **Stage 1** | Smoke test trên 5 dataset, 3 seed, các phương pháp chính | `config/suites/stage1_smoke_5datasets_3seeds_all_methods.yaml` |
| **Stage 2** | Pilot experiment trên 12 dataset, 3 seed | `config/suites/stage2_pilot_12datasets_3seeds_all_methods.yaml` |
| **Stage 3** | Mức tối thiểu cho bài báo: 20 dataset, 3 seed | `config/suites/stage3_paper_20datasets_3seeds_all_methods.yaml` |
| **Stage 3 ablation** | Ablation trên 20 dataset cho adaptive model | `config/suites/stage3_ablation_20datasets_3seeds_adaptive.yaml` |
| **Stage 3 adaptive-only** | Kiểm tra adaptive-only trên 30 dataset | `config/suites/stage3_paper_30datasets_3seeds_adaptive_only.yaml` |
| **Stage 4** | Quy mô mạnh hơn cho Q1/Q2: 30 dataset, 5 seed | `config/suites/stage4_strong_30datasets_5seeds_all_methods.yaml` |
| **Stage 4 adaptive-only** | Mở rộng adaptive-only trên 50 dataset, 3 seed | `config/suites/stage4_strong_50datasets_3seeds_adaptive_only.yaml` |

### 10.1. Dry run trước khi train thật

```bash
python -m src.cli.batch_run \
  --suite config/suites/stage1_smoke_5datasets_3seeds_all_methods.yaml \
  --dry-run
```

Lệnh này chỉ in ra các lệnh dự kiến chạy, giúp kiểm tra cấu hình trước khi train thật.

### 10.2. Stage 1 — Smoke test

```bash
python -m src.cli.batch_run \
  --suite config/suites/stage1_smoke_5datasets_3seeds_all_methods.yaml
```

### 10.3. Stage 2 — Pilot experiment

```bash
python -m src.cli.batch_run \
  --suite config/suites/stage2_pilot_12datasets_3seeds_all_methods.yaml
```

### 10.4. Stage 3 — Paper-grade minimum

```bash
python -m src.cli.batch_run \
  --suite config/suites/stage3_paper_20datasets_3seeds_all_methods.yaml
```

### 10.5. Stage 3 — Ablation study

```bash
python -m src.cli.batch_run \
  --suite config/suites/stage3_ablation_20datasets_3seeds_adaptive.yaml
```

### 10.6. Stage 4 — Stronger Q1/Q2 scale

```bash
python -m src.cli.batch_run \
  --suite config/suites/stage4_strong_30datasets_5seeds_all_methods.yaml
```

Hoặc chỉ chạy adaptive-only trên nhiều dataset:

```bash
python -m src.cli.batch_run \
  --suite config/suites/stage4_strong_50datasets_3seeds_adaptive_only.yaml
```

---

## 11. Gom kết quả và phân tích

Gom tất cả kết quả thành một file CSV:

```bash
python -m src.cli.collect_results \
  --output-root outputs \
  --out-csv outputs/summary_all_runs.csv
```

Tính average rank theo Macro-F1:

```bash
python -m src.cli.rank_results \
  --results-csv outputs/summary_all_runs.csv \
  --metric test_macro_f1
```

Chạy kiểm định thống kê:

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

## 12. Các file kết quả chính

Mỗi lần chạy sẽ sinh ra một thư mục kết quả trong `outputs/`. Các file quan trọng có thể gồm:

```text
summary.json                  # Accuracy, Macro-F1, precision, recall và thông tin chính
config_used.yaml              # cấu hình thực tế đã dùng cho lần chạy
environment.json              # thông tin Python, thư viện, phần cứng và môi trường
best_model.pt                 # checkpoint tốt nhất
last_checkpoint.pt            # checkpoint mới nhất để resume
history.csv                   # log huấn luyện theo epoch
classification_report.txt     # báo cáo phân loại theo từng lớp
confusion_matrix.png          # ma trận nhầm lẫn
learning_curve_macro_f1.png   # đường học tập Macro-F1
complexity.json               # số tham số và FLOPs nếu đo được
inference_time.json           # thời gian suy luận
alpha_mean.csv                # trọng số fusion trung bình
alpha_test_samples.csv        # trọng số fusion theo từng mẫu test
```

Khi viết bài báo, cần lưu lại ít nhất:

```text
summary.json
config_used.yaml
environment.json
```

Ba file này giúp kiểm tra lại cấu hình, môi trường và kết quả đã báo cáo.

---

## 13. Resume khi bị ngắt

Nếu quá trình train bị ngắt, chạy lại đúng lệnh cũ:

```bash
python -m src.train --config config/experiments/coffee_adaptive_fusion.yaml
```

Chương trình sẽ kiểm tra `last_checkpoint.pt` trong thư mục output và tiếp tục nếu checkpoint hợp lệ tồn tại.

---

## 14. Quy tắc GitHub và lưu trữ

Chỉ nên commit lên GitHub các file nhẹ và có khả năng tái lập:

```text
source code
YAML configuration files
command-only notebooks
documentation
requirements.txt
small summary tables hoặc lightweight result summaries
```

Không nên commit:

```text
data/UCR/*
outputs/*
cache/*
*.pt
*.pth
*.ckpt
large generated image caches
large raw result folders
```

Các mục dữ liệu, cache, output và checkpoint lớn nên lưu ở máy local hoặc Google Drive.

Quy trình cập nhật code khuyến nghị:

```bash
git status
git add README.md config src notebooks docs requirements.txt
git commit -m "Update Vietnamese README for adaptive fusion workflow"
git push
```

Nếu đang chạy trên Colab, sau khi cập nhật GitHub, dùng:

```bash
git pull
```

---

## 15. Troubleshooting

### 15.1. Không tìm thấy dataset

Kiểm tra đúng cấu trúc:

```text
data/UCR/<DatasetName>/<DatasetName>_TRAIN.tsv
data/UCR/<DatasetName>/<DatasetName>_TEST.tsv
```

Tên dataset phải khớp với file YAML.

### 15.2. Lỗi import `src`

Cần chạy lệnh từ thư mục gốc repository:

```bash
cd adaptive-fusion-ts2img
python -m src.train --config config/experiments/coffee_adaptive_fusion.yaml
```

### 15.3. Colab chạy chậm

Nên để source code ở `/content` và lưu kết quả sang Google Drive:

```text
/content/adaptive-fusion-ts2img/              # code clone từ GitHub
/content/drive/MyDrive/adaptive_fusion_runs/  # outputs, checkpoints, logs
```

Không nên chạy toàn bộ source code trực tiếp trong thư mục Google Drive nếu không cần thiết, vì thao tác đọc/ghi nhiều file nhỏ có thể chậm.

### 15.4. Kết quả khác nhau giữa các lần chạy

Kiểm tra:

```text
seed
config_used.yaml
phiên bản Python và thư viện
CPU/GPU
CUDA/cuDNN
batch size
early stopping
checkpoint resume
train/test split
tham số chuyển đổi ảnh
```

Sai khác nhỏ giữa các môi trường là bình thường, đặc biệt khi dùng GPU và các kernel không hoàn toàn deterministic.

### 15.5. Macro-F1 quá thấp

Cần kiểm tra:

```text
mất cân bằng lớp
label encoding
train/validation split
normalization
learning curves
confusion matrix
early stopping epoch
sự phù hợp của biểu diễn ảnh với dataset
```

Không nên chỉ nhìn Accuracy, đặc biệt với dataset mất cân bằng. Macro-F1 cần được xem là chỉ số chính song song với Accuracy.

---

## 16. Checklist tái lập cho bài báo

Trước khi đưa kết quả vào bản thảo, cần ghi lại:

- tên dataset;
- miền dữ liệu của dataset;
- nguồn dữ liệu và train/test split;
- số lớp, số mẫu train, số mẫu test, độ dài chuỗi;
- cấu hình GAF;
- cấu hình MTF;
- cấu hình RP;
- cấu hình STFT;
- kích thước ảnh đầu vào;
- kiến trúc nhánh CNN;
- loại fusion;
- số seed;
- optimizer, learning rate, scheduler, batch size, số epoch, early stopping;
- chiến lược validation;
- phần cứng và môi trường chạy;
- metric;
- kiểm định thống kê;
- Git commit hash;
- `config_used.yaml` và `environment.json` của từng lần chạy được báo cáo.

Đối với bài Q1/Q2, nên có thêm:

- nhiều dataset, tối thiểu khoảng 20; tốt hơn 30--50 nếu tài nguyên cho phép;
- baseline mạnh như 1D-CNN, InceptionTime, MiniROCKET, single 2D representation và manual fusion;
- kiểm định thống kê đa dataset;
- ablation study;
- báo cáo chi phí tính toán;
- phân tích trọng số fusion;
- thảo luận trung thực về trường hợp 1D-CNN hoặc biểu diễn đơn lẻ tốt hơn mô hình đề xuất.

---

## 17. Tuyên bố miễn trừ trách nhiệm

Mã nguồn này được cung cấp cho mục đích **nghiên cứu, giảng dạy và tái lập thực nghiệm**. Tác giả không bảo đảm rằng mã nguồn sẽ chạy thành công trong mọi môi trường hoặc tạo ra kết quả giống hệt trên mọi máy.

Kết quả thực nghiệm có thể khác nhau do nhiều nguyên nhân, bao gồm nhưng không giới hạn ở:

- khác biệt về phiên bản Python, PyTorch, CUDA, cuDNN, scikit-learn, pyts hoặc các thư viện phụ thuộc;
- khác biệt giữa CPU, GPU, RAM, hệ điều hành, Google Colab, máy local hoặc máy chủ tính toán;
- dữ liệu UCR/UEA không đúng định dạng, thiếu file, sai tên thư mục hoặc sai phân chia train/test;
- sai khác do seed, thuật toán GPU không hoàn toàn deterministic, batch size, early stopping hoặc checkpoint resume;
- thay đổi từ nền tảng Google Colab, Google Drive, GitHub hoặc hệ thống lưu trữ bên ngoài;
- cấu hình thí nghiệm bị chỉnh sửa so với file YAML gốc.

Người sử dụng có trách nhiệm tự kiểm tra môi trường, dữ liệu, đường dẫn, giấy phép sử dụng dữ liệu, phiên bản thư viện và kết quả đầu ra trước khi sử dụng trong bài báo, báo cáo, đồ án hoặc hệ thống thực tế.

Mã nguồn này **không phải phần mềm thương mại**, **không phải hệ thống production**, và **không đi kèm cam kết hỗ trợ, bảo hành hoặc đảm bảo kết quả công bố**. Nếu người dùng không chạy được hoặc không tái lập được kết quả, cần kiểm tra lại dữ liệu, môi trường, file cấu hình và log lỗi trước khi kết luận về phương pháp.

---

## 18. Ghi chú về giấy phép

Tại thời điểm viết README này, repository có thể chưa có file `LICENSE` rõ ràng. Nếu muốn công khai cho người khác tái sử dụng, phân phối hoặc phát triển tiếp, nên bổ sung giấy phép nguồn mở phù hợp.
