# Model Training Guide

**[English](TRAINING_EN.md)** | **[Tiếng Việt](TRAINING.md)**

This document describes in detail the notebook files used to train and analyze models in the Computational Drug Discovery project.

## Overview

The project includes 4 main notebooks:
1. **bioactivity_train.ipynb**: Training bioactivity prediction models
2. **EDA_Tox21.ipynb**: Data analysis and Tox21 dataset processing
3. **xai_4_bioactivity.ipynb**: Explainable AI analysis and visualization
4. **integration_test.ipynb**: System integration testing

---

## 1. bioactivity_train.ipynb

### Purpose
This notebook contains the complete training pipeline for 3 bioactivity prediction models:
- **ChemBERTa Reference**: Model based on pre-trained ChemBERTa
- **CNN-LSTM Hybrid**: Model combining CNN and LSTM
- **Random Forest Baseline**: Baseline model using molecular fingerprints

### Notebook Structure

#### 1.1. Setup and Import Libraries
- Import necessary libraries: PyTorch, Transformers, RDKit, scikit-learn
- Set random seed for reproducibility
- Configure device (CPU/GPU)

#### 1.2. Data Loading and Preprocessing
- Load data from `data/bioactivity_data.csv`
- Process and clean data:
  - Canonicalize SMILES strings
  - Remove invalid molecules
  - Handle missing values
- Split data into train/validation/test sets (stratified split)

#### 1.3. Baseline Model: Random Forest
- **Feature Engineering**: 
  - Generate Morgan fingerprints (radius=2, nBits=1024) from SMILES
  - Use RDKit to calculate molecular descriptors
- **Training**:
  - Grid search to find optimal hyperparameters
  - Cross-validation for evaluation
- **Evaluation**:
  - Metrics: ROC-AUC, AUPRC, F1-score, Precision, Recall
  - Confusion matrix
  - ROC curve and Precision-Recall curve
- **Save model**: Change path to `experiments/bioactivity_baselines/best_baseline_rf.joblib` 

#### 1.4. Simple CNN Model
- **Tokenization**: Character-level tokenization for SMILES
- **Architecture**: 
  - Embedding layer
  - Convolutional layers
  - Fully connected layers
- **Training**: Training loop with early stopping
- **Save model**: Change path to `checkpoints/bioactivity/best_simple_cnn.pth`

#### 1.5. ChemBERTa Reference Model
- **Pre-trained Model**: `seyonec/ChemBERTa-zinc-base-v1`
- **Architecture**:
  - ChemBERTa encoder (BERT-based)
  - Linear classification head on [CLS] token
- **Tokenization**: Use AutoTokenizer from HuggingFace
- **Training**:
  - Fine-tune pre-trained model
  - Handle class imbalance (if any)
  - Learning rate scheduling
  - Early stopping based on validation loss
- **Evaluation**: Similar to baseline model
- **Save model**: Change path to `checkpoints/bioactivity/best_reference_chemberta_xai.pth`

#### 1.6. Hybrid CNN-LSTM Model
- **Architecture**:
  - Embedding layer
  - CNN block: Extract local features
  - LSTM block: Learn sequence dependencies (bidirectional)
  - Classification head with dropout
- **Tokenization**: Character-level with custom vocabulary
- **Hyperparameter Tuning**:
  - Random search on parameter grid:
    - `hidden_dim`: [32, 64, 128]
    - `dropout`: [0.3, 0.5, 0.7]
    - `batch_size`: [32, 64, 128]
    - `lr`: [1e-4, 5e-4, 1e-3]
- **Training**:
  - Train with best parameters from random search
  - Early stopping with patience
  - Save best model based on validation loss
- **Save model**: Change path to `checkpoints/bioactivity/best_advanced_model.pth`
- **Save tokenizer metadata**: Change path to `experiments/bioactivity_baselines/cnn_tokenizer_meta.joblib`

### Output Files
After running the notebook, the following files will be created:
- `checkpoints/bioactivity/best_reference_chemberta_xai.pth`
- `checkpoints/bioactivity/best_advanced_model.pth`
- `checkpoints/bioactivity/best_simple_cnn.pth`
- `experiments/bioactivity_baselines/best_baseline_rf.joblib`
- `experiments/bioactivity_baselines/cnn_tokenizer_meta.joblib`

### Training Notes
- **GPU recommended**: Training deep learning models (ChemBERTa, CNN-LSTM) requires GPU for fast training
- **Training time**: 
  - Random Forest: ~5-10 minutes
  - ChemBERTa: ~30-60 minutes (depending on GPU)
  - CNN-LSTM: ~1-2 hours (including hyperparameter tuning)
- **Memory**: Ensure sufficient RAM (16GB+ recommended) and GPU VRAM (8GB+ recommended)

---

## 2. EDA_Tox21.ipynb

### Purpose
This notebook performs Exploratory Data Analysis (EDA) and processes the Tox21 dataset for toxicity prediction models.

### Notebook Structure

#### 2.1. Data Loading
- Use DeepChem to load Tox21 dataset
- Dataset contains 12 toxicity endpoints:
  - NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase
  - NR-ER, NR-ER-LBD, NR-PPAR-gamma
  - SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53
- Total samples: ~7,823 molecules

#### 2.2. Data Conversion
- Convert from DeepChem Dataset to Pandas DataFrame
- Handle missing labels (using weight masks)
- Add columns for each endpoint and weight mask

#### 2.3. Data Validation
- Check validity of SMILES strings
- Remove molecules with invalid SMILES
- Statistics on valid/invalid SMILES count

#### 2.4. Data Splitting
- Split data into train/validation/test sets
- Save CSV files:
  - `data/Toxic21_processed/tox21_train.csv`
  - `data/Toxic21_processed/tox21_val.csv`
  - `data/Toxic21_processed/tox21_test.csv`

#### 2.5. Exploratory Analysis
- Analyze distribution of endpoints
- Visualize class imbalance
- Analyze scaffold diversity
- Descriptive statistics about dataset

#### 2.6. Feature Engineering: Morgan Fingerprints
- Generate Morgan fingerprints from SMILES strings
- Parameters: `radius=2`, `nBits=2048`
- Apply to train/val/test sets
- Convert to numpy arrays for training

#### 2.7. Baseline Model: Random Forest
- **Feature**: Morgan fingerprints (2048 bits)
- **Training**: 
  - Train Random Forest classifier on fingerprints
  - Evaluate on validation and test sets
- **Evaluation**: 
  - Metrics: AUPRC, F1-score
  - Compare with other models

#### 2.8. MLP Model with Fingerprints
- **Architecture**: Multi-Layer Perceptron
  - Input: 2048 (Morgan fingerprints)
  - Hidden layers: 1024 → 256 → 1
  - BatchNorm and Dropout (0.2-0.3)
  - Output: Binary classification (toxic/non-toxic)
- **Training**:
  - Loss: `BCEWithLogitsLoss` with class weights to handle imbalance
  - Optimizer: AdamW (lr=3e-4, weight_decay=1e-3)
  - Early stopping based on validation AUPRC
  - Patience: 10-12 epochs
- **Class Imbalance Handling**:
  - Calculate `pos_weight = n_negative / n_positive`
  - Apply in loss function
- **Evaluation**: 
  - Validation AUPRC
  - Test AUPRC and F1-score
  - Threshold tuning on validation set

#### 2.9. ChemBERTa Fine-tuning (Main Model)
- **Pre-trained Model**: `seyonec/ChemBERTa-zinc-base-v1`
- **Architecture**:
  - AutoModelForSequenceClassification with 2 labels
  - Full model fine-tuning
- **Tokenization**:
  - Use AutoTokenizer from HuggingFace
  - Max length: 128 tokens
  - Padding and truncation
- **Dataset**:
  - Custom `SmilesDataset` class
  - DataLoader with batch_size=16 (train), 32 (val/test)
- **Training**:
  - Loss: `CrossEntropyLoss` with class weights
  - Optimizer: AdamW (lr=2e-5, weight_decay=0.01)
  - Learning rate scheduler: Linear schedule with warmup
  - Gradient clipping: max_norm=1.0
  - Early stopping: Patience=2, based on validation AUPRC
  - Max epochs: 5
- **Evaluation**:
  - Validation AUPRC and F1-score
  - Test AUPRC and F1-score
  - Threshold tuning on validation set to optimize F1
- **Save model**: 
  - Use `save_pretrained()` to save model and tokenizer
  - Save to: `artifacts/admet_chemberta_tox21/`
  - Save metadata (threshold, metrics) to `meta.json`

### Output Files
After running the notebook, the following files will be created:

**Data files:**
- `data/Toxic21_processed/tox21_train.csv`
- `data/Toxic21_processed/tox21_val.csv`
- `data/Toxic21_processed/tox21_test.csv`

**Model files (if training):**
- `artifacts/admet_chemberta_tox21/` (directory containing fine-tuned model)
  - `config.json`
  - `model.safetensors`
  - `tokenizer.json`, `tokenizer_config.json`
  - `vocab.json`, `merges.txt`
  - `special_tokens_map.json`
  - `meta.json` (contains threshold and metrics)

### Training Notes
- Tox21 dataset has many missing labels (not all molecules have labels for all endpoints)
- Need to handle class imbalance when training models (use class weights)
- **ChemBERTa is the best model** selected after comparing with Random Forest and MLP
- Optimal threshold found on validation set (usually ~0.48-0.50)
- Model saved in HuggingFace format for easy use later
- **Training time**:
  - Random Forest: ~2-5 minutes
  - MLP: ~10-20 minutes (depending on number of epochs)
  - ChemBERTa fine-tuning: ~15-30 minutes (5 epochs, depending on GPU)
- **GPU recommended**: GPU needed for ChemBERTa fine-tuning for fast training

---

## 3. xai_4_bioactivity.ipynb

### Purpose
This notebook performs Explainable AI (XAI) analysis and visualization for bioactivity models.

### Notebook Structure

#### 3.1. Model Loading
- Load trained models:
  - ChemBERTa Reference model
  - CNN-LSTM model
  - Random Forest model
- Load necessary tokenizers and metadata

#### 3.2. XAI Methods

##### 3.2.1. ChemBERTa XAI
- **Method**: Token-level importance
- **Technique**: Use gradient-based attribution (Captum)
- **Output**: 
  - Importance scores for each token in SMILES
  - Token importance visualization

##### 3.2.2. CNN-LSTM XAI
- **Method**: Character-level saliency
- **Technique**: Gradient-based saliency maps
- **Output**:
  - Saliency scores for each character
  - Saliency chart visualization

##### 3.2.3. Random Forest XAI
- **Method**: Atom-level importance
- **Technique**: SHAP values or feature importance
- **Output**:
  - Importance scores for each atom
  - Molecule visualization with highlighted atoms (using RDKit)

#### 3.3. Visualization
- Create charts and images to visualize XAI results
- Compare XAI results between models
- Analyze important patterns

### Notes
- XAI can only run after models have been trained and saved
- Need sufficient memory to load all models simultaneously

---

## 4. integration_test.ipynb

### Purpose
This notebook tests system integration functions, ensuring the pipeline works correctly end-to-end.

### Notebook Structure

#### 4.1. Model Loading Test
- Test loading models from correct paths
- Check if models can be loaded
- Verify device configuration

#### 4.2. Inference Test
- Test prediction on sample SMILES
- Check output format matches schemas
- Verify thresholds work correctly

#### 4.3. Pipeline Test
- Test end-to-end screening pipeline
- Check SMILES validation
- Verify decision logic (KEEP/REJECT)

#### 4.4. XAI Test
- Test XAI functions for each model
- Check visualization functions

### Usage
Run this notebook after:
- All models have been trained and saved
- Directory structure is set up correctly
- Want to verify system works correctly before running Streamlit app

---

## Training Instructions

### Requirements
1. All dependencies from `requirements.txt` installed
   - Especially need: DeepChem (for Tox21 dataset), Transformers, PyTorch
2. Training data in `data/` directory
   - `data/bioactivity_data.csv` (for bioactivity training)
   - Tox21 dataset will be automatically downloaded via DeepChem
3. GPU (recommended) for training deep learning models
   - ChemBERTa fine-tuning needs GPU for fast training
   - Random Forest and MLP can run on CPU

### Notebook Execution Order

1. **EDA_Tox21.ipynb**
   - Run from start to finish to:
     - Load and process Tox21 dataset
     - Create train/val/test splits (scaffold split)
     - Train baseline models (Random Forest, MLP)
     - Fine-tune ChemBERTa model (main model for toxicity)
     - Save model to `artifacts/admet_chemberta_tox21/`
   - **Note**: If only need processed data, can stop after data splitting section

2. **bioactivity_train.ipynb**
   - Run from start to finish to train all models:
     - Random Forest baseline
     - Simple CNN
     - ChemBERTa Reference
     - CNN-LSTM Hybrid
   - Can run individual sections if only want to train a specific model

3. **xai_4_bioactivity.ipynb** (optional)
   - Run to analyze XAI after models are available
   - Need all bioactivity models to be trained

4. **integration_test.ipynb**
   - Run to verify system works correctly
   - Test both bioactivity and toxicity models

### Tips

- **Run cell by cell**: Should run cells one by one for easier debugging
- **Save checkpoints**: Models will be automatically saved during training
- **Monitor training**: Track loss and metrics during training
- **Early stopping**: Models are configured with early stopping to avoid overfitting

---

## Troubleshooting

### Error: Out of Memory
- Reduce batch size
- Reduce max_length of sequences
- Use gradient accumulation

### Error: Model cannot be loaded
- Check model file paths
- Ensure models have been trained and saved
- Check PyTorch and Transformers versions

### Error: SMILES validation
- Check SMILES format in dataset
- Use RDKit to validate before training

### Error: CUDA not available
- Check CUDA installation and PyTorch with CUDA support
- Or run on CPU (will be much slower)

---

## References

- **ChemBERTa**: https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1
- **DeepChem Tox21**: https://deepchem.readthedocs.io/
- **RDKit**: https://www.rdkit.org/
- **Captum (XAI)**: https://captum.ai/

