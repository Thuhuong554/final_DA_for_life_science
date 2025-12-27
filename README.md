# Computational Drug Discovery – Multi-Stage Pipeline (Track C)

**[English](README_EN.md)** | **[Tiếng Việt](README.md)**

Hệ thống pipeline đa giai đoạn để phát hiện và sàng lọc các phân tử tiềm năng trong quá trình khám phá thuốc. Hệ thống sử dụng các mô hình deep learning và machine learning để dự đoán hoạt tính sinh học (bioactivity) và độc tính (toxicity) của các phân tử dựa trên chuỗi SMILES.

## Tính năng chính

- **Dự đoán hoạt tính sinh học**: Sử dụng 3 mô hình khác nhau (ChemBERTa, CNN-LSTM, Random Forest)
- **Dự đoán độc tính**: Mô hình ChemBERTa fine-tuned trên dataset Tox21
- **Xác thực SMILES**: Tự động kiểm tra tính hợp lệ của chuỗi SMILES trước khi dự đoán
- **Explainable AI (XAI)**: Giải thích kết quả dự đoán từ cả 3 mô hình bioactivity
- **Giao diện web Streamlit**: Dễ sử dụng, hỗ trợ upload CSV và test từng phân tử
- **Sàng lọc đa giai đoạn**: Tự động lọc các phân tử đáp ứng tiêu chí (P_active > 0.5 và P_toxic < 0.5)

## Cài đặt

### Yêu cầu hệ thống

- Python 3.8 trở lên
- CUDA (khuyến nghị cho GPU) hoặc CPU
- 8GB RAM trở lên (khuyến nghị 16GB)

### Cài đặt dependencies

1. Clone repository hoặc tải xuống project:

```bash
cd final_DA_for_life_science
```

2. Tạo môi trường ảo (khuyến nghị):

```bash
python -m venv venv
```

3. Kích hoạt môi trường ảo:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

4. Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## Tải xuống Models

Các mô hình đã được train sẵn cần được tải xuống từ Google Drive và đặt vào đúng thư mục trong project.

### Hướng dẫn tải xuống

1. Truy cập link Google Drive: https://drive.google.com/drive/folders/16-v8Z3ewsRItOOQJdlO9a4f0U5pCFU_t?usp=sharing

2. Tải xuống 3 thư mục sau:
   - **artifacts**: Chứa mô hình Tox21 (admet_chemberta_tox21)
   - **checkpoints**: Chứa các mô hình bioactivity đã train
   - **experiments**: Chứa các baseline models (Random Forest, CNN-LSTM tokenizer)

3. Giải nén và đặt các thư mục vào thư mục gốc của project với cấu trúc như sau:

```
final_DA_for_life_science/
├── artifacts/
│   └── admet_chemberta_tox21/
│       ├── config.json
│       ├── merges.txt
│       ├── meta.json
│       ├── model.safetensors
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       └── vocab.json
├── checkpoints/
│   └── bioactivity/
│       ├── best_advanced_model.pth
│       ├── best_reference_chemberta_xai.pth
│       └── best_simple_cnn.pth
└── experiments/
    └── bioactivity_baselines/
        ├── best_baseline_rf.joblib
        └── cnn_tokenizer_meta.joblib
```

**Lưu ý**: Đảm bảo các file model được đặt đúng vị trí như trên, nếu không ứng dụng sẽ không thể load được models.

## Sử dụng

### Chạy ứng dụng Streamlit

Sau khi đã cài đặt dependencies và tải xuống models, chạy lệnh sau:

```bash
streamlit run app.py
```

Ứng dụng sẽ tự động mở trong trình duyệt tại địa chỉ `http://localhost:8501`

### Các chức năng chính

#### 1. Batch CSV Upload (Khuyến nghị)

- Upload file CSV chứa các phân tử cần sàng lọc
- Chọn cột chứa SMILES từ dropdown
- Thiết lập số lượng phân tử tối đa để test (0 = tất cả)
- Nhấn "Run Screening on CSV" để bắt đầu
- Xem kết quả sàng lọc và tải xuống file kết quả
- Xem XAI explanations cho các phân tử đạt tiêu chí

#### 2. Single SMILES (Quick test)

- Nhập một chuỗi SMILES để test nhanh
- Nhấn "Run Single Screening"
- Xem kết quả dự đoán và XAI explanations

### Định dạng CSV đầu vào

File CSV cần có ít nhất một cột chứa chuỗi SMILES. File có thể sử dụng dấu `;` hoặc `,` làm delimiter.

Ví dụ:
```csv
smiles;name;id
CCO;Ethanol;1
CC(=O)O;Acetic acid;2
```

### Kết quả đầu ra

Hệ thống sẽ trả về các cột sau:

- `is_valid`: SMILES có hợp lệ không
- `validation_error`: Lỗi validation (nếu có)
- `p_active`: Xác suất hoạt tính sinh học
- `active`: Có hoạt tính hay không (P_active > 0.5)
- `p_toxic`: Xác suất độc tính
- `non_toxic`: Không độc hay không (P_toxic < 0.5)
- `keep`: Có giữ lại phân tử này không (Active & Non-Toxic)
- `reason`: Lý do quyết định

## Cấu trúc Project

```
final_DA_for_life_science/
├── app.py                          # Ứng dụng Streamlit chính
├── requirements.txt                # Danh sách dependencies
├── README.md                       # File này (Tiếng Việt)
├── README_EN.md                    # English version
├── TRAINING.md                     # Hướng dẫn training (Tiếng Việt)
├── TRAINING_EN.md                  # Training guide (English)
│
├── artifacts/                      # Mô hình Tox21 (cần tải từ Google Drive)
│   └── admet_chemberta_tox21/
│
├── checkpoints/                    # Mô hình Bioactivity (cần tải từ Google Drive)
│   └── bioactivity/
│
├── experiments/                    # Baseline models (cần tải từ Google Drive)
│   └── bioactivity_baselines/
│
├── data/                          # Dữ liệu training và test
│   ├── bioactivity_data.csv
│   └── Toxic21_processed/
│
├── models/                        # Code định nghĩa và load models
│   ├── bioactivity/
│   │   ├── model_def.py          # Định nghĩa kiến trúc mô hình
│   │   ├── loader.py             # Load models
│   │   ├── infer.py              # Inference
│   │   └── xai.py                # Explainable AI
│   └── tox21/
│       ├── hf_loader.py          # Load HuggingFace model
│       ├── hf_infer.py           # Inference Tox21
│       └── loader.py
│
├── pipeline/                      # Pipeline xử lý
│   ├── schemas.py                # Data schemas
│   └── screening.py              # End-to-end screening logic
│
├── utils/                         # Utilities
│   └── smiles_validator.py       # Validate SMILES strings
│
└── *.ipynb                        # Jupyter notebooks cho training và EDA
```

## Models

### Bioactivity Models

1. **ChemBERTa Reference**: Mô hình dựa trên ChemBERTa với classification head
   - File: `checkpoints/bioactivity/best_reference_chemberta_xai.pth`
   - XAI: Token-level importance

2. **CNN-LSTM Hybrid**: Mô hình kết hợp CNN và LSTM
   - File: `checkpoints/bioactivity/best_advanced_model.pth`
   - XAI: Character-level saliency

3. **Random Forest Baseline**: Mô hình Random Forest
   - File: `experiments/bioactivity_baselines/best_baseline_rf.joblib`
   - XAI: Atom-level importance

### Toxicity Model

- **ChemBERTa Tox21**: Mô hình ChemBERTa fine-tuned trên Tox21 dataset
  - Directory: `artifacts/admet_chemberta_tox21/`
  - Dự đoán 12 endpoints độc tính khác nhau

## Quy tắc quyết định

Hệ thống sử dụng quy tắc đơn giản để quyết định giữ lại một phân tử:

- **KEEP = TRUE** nếu: `P_active > 0.5` VÀ `P_toxic < 0.5`
- **KEEP = FALSE** trong các trường hợp khác

Các ngưỡng (thresholds) có thể được điều chỉnh trong code:
- `TAU_BIO = 0.5` (ngưỡng cho bioactivity)
- `TAU_TOX = 0.5` (ngưỡng cho toxicity)

## Troubleshooting

### Lỗi: Model không tìm thấy

- Kiểm tra lại các file model đã được tải xuống và đặt đúng vị trí
- Đảm bảo cấu trúc thư mục đúng như mô tả ở phần "Tải xuống Models"

### Lỗi: CUDA out of memory

- Giảm số lượng phân tử trong một lần chạy (sử dụng `max_rows` trong CSV upload)
- Hoặc chạy trên CPU (sẽ chậm hơn)

### Lỗi: Invalid SMILES

- Kiểm tra định dạng SMILES string
- Sử dụng công cụ validation trong ứng dụng để kiểm tra

## Phát triển

### Training models

Để train lại các models hoặc tìm hiểu chi tiết về quy trình training, vui lòng xem file **[TRAINING.md](TRAINING.md)** (Tiếng Việt) hoặc **[TRAINING_EN.md](TRAINING_EN.md)** (English).

File TRAINING bao gồm hướng dẫn chi tiết về:
- **bioactivity_train.ipynb**: Training 3 mô hình bioactivity (ChemBERTa, CNN-LSTM, Random Forest)
- **EDA_Tox21.ipynb**: Phân tích và xử lý dataset Tox21
- **xai_4_bioactivity.ipynb**: Phân tích và visualization Explainable AI
- **integration_test.ipynb**: Test tích hợp hệ thống

### Testing

Sử dụng `integration_test.ipynb` để test các chức năng của hệ thống. Xem chi tiết trong [TRAINING.md](TRAINING.md) hoặc [TRAINING_EN.md](TRAINING_EN.md).

## License

Xem file LICENSE (nếu có) hoặc liên hệ tác giả để biết thêm chi tiết.

## Liên hệ

Nếu có câu hỏi hoặc gặp vấn đề, vui lòng tạo issue trên repository hoặc liên hệ trực tiếp.

---

**Lưu ý**: Đảm bảo đã tải xuống đầy đủ các models từ Google Drive trước khi chạy ứng dụng!

