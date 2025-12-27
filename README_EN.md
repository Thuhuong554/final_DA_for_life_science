# Computational Drug Discovery – Multi-Stage Pipeline (Track C)

**[English](README_EN.md)** | **[Tiếng Việt](README.md)**

A multi-stage pipeline system for discovering and screening potential molecules in drug discovery. The system uses deep learning and machine learning models to predict bioactivity and toxicity of molecules based on SMILES strings.

## Key Features

- **Bioactivity Prediction**: Uses 3 different models (ChemBERTa, CNN-LSTM, Random Forest)
- **Toxicity Prediction**: ChemBERTa model fine-tuned on Tox21 dataset
- **SMILES Validation**: Automatically validates SMILES strings before prediction
- **Explainable AI (XAI)**: Explains prediction results from all 3 bioactivity models
- **Streamlit Web Interface**: Easy to use, supports CSV upload and single molecule testing
- **Multi-stage Screening**: Automatically filters molecules meeting criteria (P_active > 0.5 and P_toxic < 0.5)

## Installation

### System Requirements

- Python 3.8 or higher
- CUDA (recommended for GPU) or CPU
- 8GB RAM or more (16GB recommended)

### Installing Dependencies

1. Clone the repository or download the project:

```bash
cd final_DA_for_life_science
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
```

3. Activate the virtual environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

4. Install required libraries:

```bash
pip install -r requirements.txt
```

## Download Models

Pre-trained models need to be downloaded from Google Drive and placed in the correct directories in the project.

### Download Instructions

1. Access the Google Drive link: https://drive.google.com/drive/folders/16-v8Z3ewsRItOOQJdlO9a4f0U5pCFU_t?usp=sharing

2. Download the following 3 folders:
   - **artifacts**: Contains Tox21 model (admet_chemberta_tox21)
   - **checkpoints**: Contains trained bioactivity models
   - **experiments**: Contains baseline models (Random Forest, CNN-LSTM tokenizer)

3. Extract and place the folders in the project root directory with the following structure:

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

**Note**: Ensure model files are placed in the correct locations as shown above, otherwise the application will not be able to load the models.

## Usage

### Running the Streamlit Application

After installing dependencies and downloading models, run the following command:

```bash
streamlit run app.py
```

The application will automatically open in your browser at `http://localhost:8501`

### Main Features

#### 1. Batch CSV Upload (Recommended)

- Upload a CSV file containing molecules to screen
- Select the SMILES column from the dropdown
- Set maximum number of molecules to test (0 = all)
- Click "Run Screening on CSV" to start
- View screening results and download result files
- View XAI explanations for molecules that meet criteria

#### 2. Single SMILES (Quick test)

- Enter a SMILES string for quick testing
- Click "Run Single Screening"
- View prediction results and XAI explanations

### Input CSV Format

The CSV file needs at least one column containing SMILES strings. The file can use `;` or `,` as delimiter.

Example:
```csv
smiles;name;id
CCO;Ethanol;1
CC(=O)O;Acetic acid;2
```

### Output Results

The system will return the following columns:

- `is_valid`: Whether SMILES is valid
- `validation_error`: Validation error (if any)
- `p_active`: Bioactivity probability
- `active`: Whether active (P_active > 0.5)
- `p_toxic`: Toxicity probability
- `non_toxic`: Whether non-toxic (P_toxic < 0.5)
- `keep`: Whether to keep this molecule (Active & Non-Toxic)
- `reason`: Decision reason

## Project Structure

```
final_DA_for_life_science/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Dependencies list
├── README.md                       # Vietnamese version
├── README_EN.md                    # This file (English)
├── TRAINING.md                     # Training guide (Vietnamese)
├── TRAINING_EN.md                  # Training guide (English)
│
├── artifacts/                      # Tox21 model (download from Google Drive)
│   └── admet_chemberta_tox21/
│
├── checkpoints/                    # Bioactivity models (download from Google Drive)
│   └── bioactivity/
│
├── experiments/                    # Baseline models (download from Google Drive)
│   └── bioactivity_baselines/
│
├── data/                          # Training and test data
│   ├── bioactivity_data.csv
│   └── Toxic21_processed/
│
├── models/                        # Model definition and loading code
│   ├── bioactivity/
│   │   ├── model_def.py          # Model architecture definition
│   │   ├── loader.py             # Load models
│   │   ├── infer.py              # Inference
│   │   └── xai.py                # Explainable AI
│   └── tox21/
│       ├── hf_loader.py          # Load HuggingFace model
│       ├── hf_infer.py           # Tox21 inference
│       └── loader.py
│
├── pipeline/                      # Processing pipeline
│   ├── schemas.py                # Data schemas
│   └── screening.py              # End-to-end screening logic
│
├── utils/                         # Utilities
│   └── smiles_validator.py       # Validate SMILES strings
│
└── *.ipynb                        # Jupyter notebooks for training and EDA
```

## Models

### Bioactivity Models

1. **ChemBERTa Reference**: Model based on ChemBERTa with classification head
   - File: `checkpoints/bioactivity/best_reference_chemberta_xai.pth`
   - XAI: Token-level importance

2. **CNN-LSTM Hybrid**: Model combining CNN and LSTM
   - File: `checkpoints/bioactivity/best_advanced_model.pth`
   - XAI: Character-level saliency

3. **Random Forest Baseline**: Random Forest model
   - File: `experiments/bioactivity_baselines/best_baseline_rf.joblib`
   - XAI: Atom-level importance

### Toxicity Model

- **ChemBERTa Tox21**: ChemBERTa model fine-tuned on Tox21 dataset
  - Directory: `artifacts/admet_chemberta_tox21/`
  - Predicts 12 different toxicity endpoints

## Decision Rules

The system uses a simple rule to decide whether to keep a molecule:

- **KEEP = TRUE** if: `P_active > 0.5` AND `P_toxic < 0.5`
- **KEEP = FALSE** otherwise

Thresholds can be adjusted in the code:
- `TAU_BIO = 0.5` (threshold for bioactivity)
- `TAU_TOX = 0.5` (threshold for toxicity)

## Troubleshooting

### Error: Model not found

- Check that model files have been downloaded and placed in correct locations
- Ensure directory structure matches the description in "Download Models" section

### Error: CUDA out of memory

- Reduce the number of molecules processed at once (use `max_rows` in CSV upload)
- Or run on CPU (will be slower)

### Error: Invalid SMILES

- Check SMILES string format
- Use the validation tool in the application to check

## Development

### Training Models

To retrain models or learn more about the training process, please see **[TRAINING_EN.md](TRAINING_EN.md)** (English) or **[TRAINING.md](TRAINING.md)** (Vietnamese).

The TRAINING files include detailed guides on:
- **bioactivity_train.ipynb**: Training 3 bioactivity models (ChemBERTa, CNN-LSTM, Random Forest)
- **EDA_Tox21.ipynb**: Analysis and processing of Tox21 dataset
- **xai_4_bioactivity.ipynb**: Explainable AI analysis and visualization
- **integration_test.ipynb**: System integration testing

### Testing

Use `integration_test.ipynb` to test system functions. See details in [TRAINING_EN.md](TRAINING_EN.md) or [TRAINING.md](TRAINING.md).

## License

See LICENSE file (if available) or contact the author for more details.

## Contact

If you have questions or encounter issues, please create an issue on the repository or contact directly.

---

**Note**: Make sure to download all models from Google Drive before running the application!

