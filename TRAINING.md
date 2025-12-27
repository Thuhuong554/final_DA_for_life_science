# Hướng dẫn Training Models

**[English](TRAINING_EN.md)** | **[Tiếng Việt](TRAINING.md)**

Tài liệu này mô tả chi tiết về các file notebook được sử dụng để train và phân tích các mô hình trong dự án Computational Drug Discovery.

## Tổng quan

Dự án bao gồm 4 notebook chính:
1. **bioactivity_train.ipynb**: Training các mô hình dự đoán hoạt tính sinh học
2. **EDA_Tox21.ipynb**: Phân tích dữ liệu và xử lý dataset Tox21
3. **xai_4_bioactivity.ipynb**: Phân tích và visualization Explainable AI
4. **integration_test.ipynb**: Test tích hợp các chức năng của hệ thống

---

## 1. bioactivity_train.ipynb

### Mục đích
Notebook này chứa toàn bộ quy trình training cho 3 mô hình dự đoán hoạt tính sinh học:
- **ChemBERTa Reference**: Mô hình dựa trên pre-trained ChemBERTa
- **CNN-LSTM Hybrid**: Mô hình kết hợp CNN và LSTM
- **Random Forest Baseline**: Mô hình baseline sử dụng molecular fingerprints

### Cấu trúc Notebook

#### 1.1. Setup và Import Libraries
- Import các thư viện cần thiết: PyTorch, Transformers, RDKit, scikit-learn
- Thiết lập random seed để đảm bảo reproducibility
- Cấu hình device (CPU/GPU)

#### 1.2. Data Loading và Preprocessing
- Load dữ liệu từ `data/bioactivity_data.csv`
- Xử lý và làm sạch dữ liệu:
  - Canonicalize SMILES strings
  - Loại bỏ các phân tử không hợp lệ
  - Xử lý missing values
- Chia dữ liệu thành train/validation/test sets (stratified split)

#### 1.3. Baseline Model: Random Forest
- **Feature Engineering**: 
  - Tạo Morgan fingerprints (radius=2, nBits=1024) từ SMILES
  - Sử dụng RDKit để tính toán molecular descriptors
- **Training**:
  - Grid search để tìm hyperparameters tối ưu
  - Cross-validation để đánh giá
- **Evaluation**:
  - Metrics: ROC-AUC, AUPRC, F1-score, Precision, Recall
  - Confusion matrix
  - ROC curve và Precision-Recall curve
- **Lưu model**: Thay đổi đường dẫn thành `experiments/bioactivity_baselines/best_baseline_rf.joblib` 

#### 1.4. Simple CNN Model
- **Tokenization**: Character-level tokenization cho SMILES
- **Architecture**: 
  - Embedding layer
  - Convolutional layers
  - Fully connected layers
- **Training**: Training loop với early stopping
- **Lưu model**: Thay đổi đường dẫn thành `checkpoints/bioactivity/best_simple_cnn.pth`

#### 1.5. ChemBERTa Reference Model
- **Pre-trained Model**: `seyonec/ChemBERTa-zinc-base-v1`
- **Architecture**:
  - ChemBERTa encoder (BERT-based)
  - Linear classification head trên [CLS] token
- **Tokenization**: Sử dụng AutoTokenizer từ HuggingFace
- **Training**:
  - Fine-tuning pre-trained model
  - Xử lý class imbalance (nếu có)
  - Learning rate scheduling
  - Early stopping dựa trên validation loss
- **Evaluation**: Tương tự như baseline model
- **Lưu model**: Thay đổi đường dẫn thành `checkpoints/bioactivity/best_reference_chemberta_xai.pth`

#### 1.6. Hybrid CNN-LSTM Model
- **Architecture**:
  - Embedding layer
  - CNN block: Extract local features
  - LSTM block: Learn sequence dependencies (bidirectional)
  - Classification head với dropout
- **Tokenization**: Character-level với custom vocabulary
- **Hyperparameter Tuning**:
  - Random search trên parameter grid:
    - `hidden_dim`: [32, 64, 128]
    - `dropout`: [0.3, 0.5, 0.7]
    - `batch_size`: [32, 64, 128]
    - `lr`: [1e-4, 5e-4, 1e-3]
- **Training**:
  - Training với best parameters từ random search
  - Early stopping với patience
  - Lưu model tốt nhất dựa trên validation loss
- **Lưu model**: Thay đổi đường dẫn thành `checkpoints/bioactivity/best_advanced_model.pth`
- **Lưu tokenizer metadata**: Thay đổi đường dẫn thành `experiments/bioactivity_baselines/cnn_tokenizer_meta.joblib`

### Output Files
Sau khi chạy notebook, các file sau sẽ được tạo:
- `checkpoints/bioactivity/best_reference_chemberta_xai.pth`
- `checkpoints/bioactivity/best_advanced_model.pth`
- `checkpoints/bioactivity/best_simple_cnn.pth`
- `experiments/bioactivity_baselines/best_baseline_rf.joblib`
- `experiments/bioactivity_baselines/cnn_tokenizer_meta.joblib`

### Lưu ý khi Training
- **GPU khuyến nghị**: Training các mô hình deep learning (ChemBERTa, CNN-LSTM) cần GPU để train nhanh
- **Thời gian training**: 
  - Random Forest: ~5-10 phút
  - ChemBERTa: ~30-60 phút (tùy GPU)
  - CNN-LSTM: ~1-2 giờ (bao gồm hyperparameter tuning)
- **Memory**: Đảm bảo có đủ RAM (khuyến nghị 16GB+) và VRAM cho GPU (khuyến nghị 8GB+)

---

## 2. EDA_Tox21.ipynb

### Mục đích
Notebook này thực hiện Exploratory Data Analysis (EDA) và xử lý dataset Tox21 cho mô hình dự đoán độc tính.

### Cấu trúc Notebook

#### 2.1. Data Loading
- Sử dụng DeepChem để load Tox21 dataset
- Dataset chứa 12 toxicity endpoints:
  - NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase
  - NR-ER, NR-ER-LBD, NR-PPAR-gamma
  - SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53
- Tổng số samples: ~7,823 phân tử

#### 2.2. Data Conversion
- Chuyển đổi từ DeepChem Dataset sang Pandas DataFrame
- Xử lý missing labels (sử dụng weight masks)
- Thêm các cột cho từng endpoint và weight mask

#### 2.3. Data Validation
- Kiểm tra tính hợp lệ của SMILES strings
- Loại bỏ các phân tử có SMILES không hợp lệ
- Thống kê số lượng valid/invalid SMILES

#### 2.4. Data Splitting
- Chia dữ liệu thành train/validation/test sets
- Lưu các file CSV:
  - `data/Toxic21_processed/tox21_train.csv`
  - `data/Toxic21_processed/tox21_val.csv`
  - `data/Toxic21_processed/tox21_test.csv`

#### 2.5. Exploratory Analysis
- Phân tích phân phối của các endpoints
- Visualize class imbalance
- Phân tích scaffold diversity
- Thống kê mô tả về dataset

#### 2.6. Feature Engineering: Morgan Fingerprints
- Tạo Morgan fingerprints từ SMILES strings
- Parameters: `radius=2`, `nBits=2048`
- Áp dụng cho train/val/test sets
- Chuyển đổi thành numpy arrays để training

#### 2.7. Baseline Model: Random Forest
- **Feature**: Morgan fingerprints (2048 bits)
- **Training**: 
  - Train Random Forest classifier trên fingerprints
  - Đánh giá trên validation và test sets
- **Evaluation**: 
  - Metrics: AUPRC, F1-score
  - So sánh với các mô hình khác

#### 2.8. MLP Model với Fingerprints
- **Architecture**: Multi-Layer Perceptron
  - Input: 2048 (Morgan fingerprints)
  - Hidden layers: 1024 → 256 → 1
  - BatchNorm và Dropout (0.2-0.3)
  - Output: Binary classification (toxic/non-toxic)
- **Training**:
  - Loss: `BCEWithLogitsLoss` với class weights để xử lý imbalance
  - Optimizer: AdamW (lr=3e-4, weight_decay=1e-3)
  - Early stopping dựa trên validation AUPRC
  - Patience: 10-12 epochs
- **Class Imbalance Handling**:
  - Tính `pos_weight = n_negative / n_positive`
  - Áp dụng trong loss function
- **Evaluation**: 
  - Validation AUPRC
  - Test AUPRC và F1-score
  - Threshold tuning trên validation set

#### 2.9. ChemBERTa Fine-tuning (Model chính)
- **Pre-trained Model**: `seyonec/ChemBERTa-zinc-base-v1`
- **Architecture**:
  - AutoModelForSequenceClassification với 2 labels
  - Fine-tuning toàn bộ model
- **Tokenization**:
  - Sử dụng AutoTokenizer từ HuggingFace
  - Max length: 128 tokens
  - Padding và truncation
- **Dataset**:
  - Custom `SmilesDataset` class
  - DataLoader với batch_size=16 (train), 32 (val/test)
- **Training**:
  - Loss: `CrossEntropyLoss` với class weights
  - Optimizer: AdamW (lr=2e-5, weight_decay=0.01)
  - Learning rate scheduler: Linear schedule with warmup
  - Gradient clipping: max_norm=1.0
  - Early stopping: Patience=2, dựa trên validation AUPRC
  - Max epochs: 5
- **Evaluation**:
  - Validation AUPRC và F1-score
  - Test AUPRC và F1-score
  - Threshold tuning trên validation set để tối ưu F1
- **Lưu model**: 
  - Sử dụng `save_pretrained()` để lưu model và tokenizer
  - Lưu vào: `artifacts/admet_chemberta_tox21/`
  - Lưu metadata (threshold, metrics) vào `meta.json`

### Output Files
Sau khi chạy notebook, các file sau sẽ được tạo:

**Data files:**
- `data/Toxic21_processed/tox21_train.csv`
- `data/Toxic21_processed/tox21_val.csv`
- `data/Toxic21_processed/tox21_test.csv`

**Model files (nếu train):**
- `artifacts/admet_chemberta_tox21/` (thư mục chứa model đã fine-tune)
  - `config.json`
  - `model.safetensors`
  - `tokenizer.json`, `tokenizer_config.json`
  - `vocab.json`, `merges.txt`
  - `special_tokens_map.json`
  - `meta.json` (chứa threshold và metrics)

### Lưu ý khi Training
- Dataset Tox21 có nhiều missing labels (không phải tất cả phân tử đều có label cho tất cả endpoints)
- Cần xử lý class imbalance khi training model (sử dụng class weights)
- **ChemBERTa là model tốt nhất** được chọn sau khi so sánh với Random Forest và MLP
- Threshold tối ưu được tìm trên validation set (thường ~0.48-0.50)
- Model được lưu dưới dạng HuggingFace format để dễ sử dụng sau này
- **Thời gian training**:
  - Random Forest: ~2-5 phút
  - MLP: ~10-20 phút (tùy số epochs)
  - ChemBERTa fine-tuning: ~15-30 phút (5 epochs, tùy GPU)
- **GPU khuyến nghị**: Cần GPU cho ChemBERTa fine-tuning để train nhanh

---

## 3. xai_4_bioactivity.ipynb

### Mục đích
Notebook này thực hiện phân tích và visualization Explainable AI (XAI) cho các mô hình bioactivity.

### Cấu trúc Notebook

#### 3.1. Model Loading
- Load các mô hình đã được train:
  - ChemBERTa Reference model
  - CNN-LSTM model
  - Random Forest model
- Load tokenizers và metadata cần thiết

#### 3.2. XAI Methods

##### 3.2.1. ChemBERTa XAI
- **Method**: Token-level importance
- **Technique**: Sử dụng gradient-based attribution (Captum)
- **Output**: 
  - Importance scores cho từng token trong SMILES
  - Visualization token importance

##### 3.2.2. CNN-LSTM XAI
- **Method**: Character-level saliency
- **Technique**: Gradient-based saliency maps
- **Output**:
  - Saliency scores cho từng character
  - Visualization saliency chart

##### 3.2.3. Random Forest XAI
- **Method**: Atom-level importance
- **Technique**: SHAP values hoặc feature importance
- **Output**:
  - Importance scores cho từng atom
  - Visualization molecule với highlighted atoms (sử dụng RDKit)

#### 3.3. Visualization
- Tạo các biểu đồ và hình ảnh để visualize XAI results
- So sánh kết quả XAI giữa các mô hình
- Phân tích các patterns quan trọng

### Lưu ý
- XAI chỉ có thể chạy sau khi đã train và lưu các models
- Cần có đủ memory để load các models cùng lúc

---

## 4. integration_test.ipynb

### Mục đích
Notebook này test tích hợp các chức năng của hệ thống, đảm bảo pipeline hoạt động đúng từ đầu đến cuối.

### Cấu trúc Notebook

#### 4.1. Model Loading Test
- Test load các models từ đúng đường dẫn
- Kiểm tra models có thể load được không
- Verify device configuration

#### 4.2. Inference Test
- Test prediction cho một số SMILES mẫu
- Kiểm tra output format đúng với schemas
- Verify thresholds hoạt động đúng

#### 4.3. Pipeline Test
- Test end-to-end screening pipeline
- Kiểm tra validation SMILES
- Verify decision logic (KEEP/REJECT)

#### 4.4. XAI Test
- Test XAI functions cho từng model
- Kiểm tra visualization functions

### Sử dụng
Chạy notebook này sau khi:
- Đã train và lưu tất cả models
- Đã setup đúng cấu trúc thư mục
- Muốn verify hệ thống hoạt động đúng trước khi chạy Streamlit app

---

## Hướng dẫn chạy Training

### Yêu cầu
1. Đã cài đặt tất cả dependencies từ `requirements.txt`
   - Đặc biệt cần: DeepChem (cho Tox21 dataset), Transformers, PyTorch
2. Có dữ liệu training trong thư mục `data/`
   - `data/bioactivity_data.csv` (cho bioactivity training)
   - Tox21 dataset sẽ được tự động download qua DeepChem
3. GPU (khuyến nghị) cho training các mô hình deep learning
   - ChemBERTa fine-tuning cần GPU để train nhanh
   - Random Forest và MLP có thể chạy trên CPU

### Thứ tự chạy Notebooks

1. **EDA_Tox21.ipynb**
   - Chạy từ đầu đến cuối để:
     - Load và xử lý dataset Tox21
     - Tạo train/val/test splits (scaffold split)
     - Train các baseline models (Random Forest, MLP)
     - Fine-tune ChemBERTa model (model chính cho toxicity)
     - Lưu model vào `artifacts/admet_chemberta_tox21/`
   - **Lưu ý**: Nếu chỉ cần processed data, có thể dừng sau phần data splitting

2. **bioactivity_train.ipynb**
   - Chạy từ đầu đến cuối để train tất cả models:
     - Random Forest baseline
     - Simple CNN
     - ChemBERTa Reference
     - CNN-LSTM Hybrid
   - Có thể chạy từng section riêng nếu chỉ muốn train một model cụ thể

3. **xai_4_bioactivity.ipynb** (optional)
   - Chạy để phân tích XAI sau khi đã có models
   - Cần có tất cả bioactivity models đã được train

4. **integration_test.ipynb**
   - Chạy để verify hệ thống hoạt động đúng
   - Test cả bioactivity và toxicity models

### Tips

- **Chạy từng cell**: Nên chạy từng cell một để dễ debug
- **Lưu checkpoint**: Models sẽ được tự động lưu trong quá trình training
- **Monitor training**: Theo dõi loss và metrics trong quá trình training
- **Early stopping**: Các mô hình đã được cấu hình early stopping để tránh overfitting

---

## Troubleshooting

### Lỗi: Out of Memory
- Giảm batch size
- Giảm max_length của sequences
- Sử dụng gradient accumulation

### Lỗi: Model không load được
- Kiểm tra đường dẫn file model
- Đảm bảo đã train và lưu model trước đó
- Kiểm tra version của PyTorch và Transformers

### Lỗi: SMILES validation
- Kiểm tra format của SMILES trong dataset
- Sử dụng RDKit để validate trước khi training

### Lỗi: CUDA không available
- Kiểm tra cài đặt CUDA và PyTorch với CUDA support
- Hoặc chạy trên CPU (sẽ chậm hơn nhiều)

---

## Tham khảo

- **ChemBERTa**: https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1
- **DeepChem Tox21**: https://deepchem.readthedocs.io/
- **RDKit**: https://www.rdkit.org/
- **Captum (XAI)**: https://captum.ai/

