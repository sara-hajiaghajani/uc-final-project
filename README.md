# Weather Forecasting with Patch-based Time Series Models

This repository contains the implementation and experiments for our paper "Evaluating Patch-wise Decoding for Urban Time Series Forecasting". We compare four time series forecasting architectures on real-world KNMI weather data, focusing on evaluating the patch-wise decoding strategy.

## Overview

**Dataset:** KNMI hourly weather data (2011-2020)
- Schiphol (Station 240) - 87,672 hourly records
- Maastricht (Station 380) - 87,672 hourly records
- Total: 175,344 hourly records

**Models:**
1. Baseline (Sequential) - Standard transformer without patching
2. PatchTST (Sequence-wise) - Patch-based with flatten-and-project decoder
3. PatchTST (Patch-wise) - Patch-based with independent patch decoder
4. aLLM4TS (GPT-2) - Fine-tuned language model for time series

**Features:** Temperature, Wind Speed, Humidity, Pressure

**Key Results:**
- PatchTST Patch-wise achieves 6.7% lower MSE than baseline
- 1.8% improvement over sequence-wise decoding with 17% fewer parameters
- Consistent performance across both stations (generalization validated)

---

## Quick Start

### 1. Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

**Required packages:**
- PyTorch >= 2.0
- transformers (for GPT-2)
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- tqdm

---

### 2. Prepare Data

**Step 2.1:** Place raw KNMI data files in `data/` directory:
```
data/
  Schiphol_240_2011-2020.txt
  Maastricht_380_2011-2020.txt
```

**Step 2.2:** Run data preparation:
```bash
python data_preparation.py
```

**Output:** Creates `data/knmi_hourly.csv` with combined station data.

**Expected console output:**
```
Processing schiphol...
  Loaded 87,672 records (2011-01-01 to 2020-12-31)
Processing maastricht...
  Loaded 87,672 records (2011-01-01 to 2020-12-31)

Combined 175,344 records from 2 stations
Date range: 2011-01-01 to 2020-12-31
Saved to: data/knmi_hourly.csv

Records per station:
  schiphol: 87,672
  maastricht: 87,672
```

---

### 3. Run Experiments

Important: Run experiments in order (1, 2, 3, then Final). Each builds on the previous.

#### Experiment 1: Patch Length Ablation
```bash
python experiment_1_patch_ablation.py
```
- Tests: patch lengths [8, 16, 32] hours
- Model: PatchTST Patchwise only
- Fixed: SEQ_LEN=336, PRED_LEN=96
- Total runs: 3 configs x 1 model x 3 seeds = 9 training runs
- Result: Optimal patch length = 32 hours

#### Experiment 2: Input Length Ablation
```bash
python experiment_2_input_ablation.py
```
- Tests: input lengths [168, 336, 672] hours (7, 14, 28 days)
- Model: PatchTST Patchwise only
- Fixed: PRED_LEN=96, PATCH_LEN=32 (from Exp 1)
- Total runs: 3 configs x 1 model x 3 seeds = 9 training runs
- Result: Optimal input length = 672 hours

#### Experiment 3: Forecast Horizon Ablation
```bash
python experiment_3_horizon_ablation.py
```
- Tests: horizons [24, 96, 336] hours (1, 4, 14 days)
- Model: PatchTST Patchwise only
- Fixed: SEQ_LEN=672, PATCH_LEN=32 (from Exp 1-2)
- Total runs: 3 configs x 1 model x 3 seeds = 9 training runs
- Result: Optimal horizon = 24 hours

#### Final Evaluation
```bash
python final_evaluation.py
```
- Configuration: SEQ_LEN=672, PRED_LEN=24, PATCH_LEN=32 (optimal from Exp 1-3)
- Models: All 4 models
- Total runs: 4 models x 5 seeds = 20 training runs
- Output: Final model comparison with robust statistics

#### Generate Analysis
```bash
python generate_analysis.py
```
- Runs after final_evaluation.py
- Generates statistical significance tests and convergence plots
- Time: Less than 1 minute
- Output: Statistical tests (txt) and training curves (pdf)

**What it does:**
1. Calculates p-values comparing all model pairs
2. Plots validation MSE convergence across 5 runs with standard deviation bands
3. Shows which improvements are statistically significant

**Statistical Tests Generated:**
- PatchTST Patch-wise vs Baseline: +6.69% improvement (p < 0.001, highly significant)
- PatchTST Patch-wise vs Sequence-wise: +1.77% improvement (p < 0.001, highly significant)
- PatchTST Patch-wise vs GPT-2: +0.01% improvement (p = 0.989, not significant)
- PatchTST Sequence-wise vs Baseline: +5.01% improvement (p < 0.001, highly significant)

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
│
├── Core Files
│   ├── data_preparation.py          # Converts raw KNMI to CSV
│   ├── data_loader.py                # PyTorch data loaders
│   └── train_and_evaluate.py         # Training loop and metrics
│
├── Model Files
│   ├── model_baseline_sequential.py  # Baseline without patches
│   ├── model_patchtst_sequential.py  # PatchTST sequence-wise
│   ├── model_patchtst_patchwise.py   # PatchTST patch-wise
│   └── model_allm4ts_gpt2.py         # GPT-2 patch-wise
│
├── Experiment Files
│   ├── experiment_1_patch_ablation.py  # Tests patch lengths [8, 16, 32]
│   ├── experiment_2_input_ablation.py  # Tests input lengths [168, 336, 672]
│   ├── experiment_3_horizon_ablation.py  # Tests forecast horizons [24, 96, 336]
│   ├── final_evaluation.py           # Compares all 4 models with optimal 
│   └── generate_analysis.py          # Statistical tests and convergence plots
│
└── Data and Results (created when running)
    ├── data/
    │   ├── Schiphol_240_2011-2020.txt (raw)
    │   ├── Maastricht_380_2011-2020.txt (raw)
    │   └── knmi_hourly.csv (processed)
    ├── checkpoints/
    └── results/
```

---

## Results Structure

Each experiment creates organized results:

```
results/
  experiment1_patch/
    patch_8/results.json
    patch_16/results.json
    patch_32/results.json
    summary/
      summary.txt
      patch_ablation_combined.pdf
      
  experiment2_input/
    input_168/results.json
    input_336/results.json
    input_672/results.json
    summary/
      summary.txt
      input_ablation_combined.pdf
    
  experiment3_horizon/
    horizon_24/results.json
    horizon_96/results.json
    horizon_336/results.json
    summary/
      summary.txt
      horizon_ablation_combined.pdf
    
  final_evaluation/
    results.json
    summary.txt
    statistical_tests.txt
    baseline_comparison.pdf
    schiphol_baseline_comparison.pdf
    maastricht_baseline_comparison.pdf
    training_curves/
      model_comparison_convergence.pdf
```

**Understanding Results:**

- results.json: Contains config, individual_runs, and averaged_results
- summary.txt: Overall results table, per-station breakdown, best configuration
- statistical_tests.txt: P-values and significance levels for model comparisons
- model_comparison_convergence.pdf: Validation MSE curves across training epochs
- Plots: Experiments 1-3 show combined stations, Final evaluation shows per-station comparisons

---

## Configuration

Key parameters in each experiment file's `Config` class:

```python
class Config:
    # Data
    FEATURES = ["temperature", "wind_speed", "humidity", "pressure"]
    SEQ_LEN = 672      # Input sequence length (hours)
    PRED_LEN = 24      # Forecast horizon (hours)
    PATCH_LEN = 32     # Patch size
    STRIDE = 16        # PATCH_LEN // 2
    
    # Training
    NUM_EPOCHS = 50    # Maximum epochs (final eval)
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-5
    PATIENCE = 10      # Early stopping patience
    
    # Data Split
    TRAIN_SPLIT = 0.7  # 70% training (122,740 samples)
    VAL_SPLIT = 0.15   # 15% validation (26,301 samples)
    TEST_SPLIT = 0.15  # 15% testing (26,302 samples)
    
    # Hardware
    DEVICE = "mps"     # Options: "cuda", "mps", or "cpu"
```

---

## Key Results

### Final Model Comparison

| Model | MSE (Mean±Std) | MAE (Mean±Std) | Parameters |
|-------|----------------|----------------|------------|
| Baseline (No Patches) | 25.60±0.18 | 2.98±0.016 | 8.9M |
| PatchTST Sequence-wise | 24.32±0.04 | 2.85±0.002 | 730K |
| PatchTST Patch-wise | 23.89±0.03 | 2.82±0.002 | 608K |
| aLLM4TS (GPT-2) | 23.89±0.14 | 2.88±0.007 | 123.9M |

**Key Findings:**
1. 6.7% improvement over baseline (25.60 to 23.89 MSE)
2. 1.8% improvement over sequence-wise (24.32 to 23.89 MSE)
3. 17% fewer parameters than sequence-wise (608K vs 730K)
4. 5x lower variance than GPT-2 (±0.03 vs ±0.14)
5. 204x fewer parameters than GPT-2 with same performance

### Statistical Significance

All improvements are statistically significant (p < 0.001):
- Patch-wise vs Baseline: p = 0.000042
- Patch-wise vs Sequence-wise: p = 0.000007
- Sequence-wise vs Baseline: p = 0.000231
- Patch-wise vs GPT-2: p = 0.989 (not significant, equivalent performance)

### Ablation Study Results

| Experiment | Best Setting | MSE | Improvement |
|------------|-------------|-----|-------------|
| Patch Length | 32 hours | 45.79±0.12 | Baseline |
| Input Length | 672 hours | 45.34±0.06 | 1.0% vs 336h |
| Forecast Horizon | 24 hours | 24.42±0.01 | Best accuracy |

### Data Statistics

| Feature | Min | Max | Mean | Std |
|---------|-----|-----|------|-----|
| Temperature (C) | -17.8 | 39.3 | 11.1 | 6.8 |
| Wind Speed (m/s) | 0.0 | 23.0 | 4.4 | 2.5 |
| Humidity (%) | 16 | 100 | 78.5 | 15.4 |
| Pressure (hPa) | 968.7 | 1049.0 | 1015.7 | 9.8 |

---

## Expected Runtime

On Apple M1/M2 with MPS or NVIDIA GPU:
- Data preparation: ~10 seconds
- Single training run (20 epochs): ~30-60 minutes depending on configuration
- Single training run (50 epochs): ~75-150 minutes depending on configuration
- Experiment 1-3 (each): ~3-6 hours (9 runs)
- Final evaluation: ~10-20 hours (20 runs)
- Generate analysis: ~1 minute
- Total for all experiments: ~40-80 hours

**Training time per epoch:**

| Configuration | Time per Epoch |
|--------------|----------------|
| Exp 1: Patch=8 | 9 min          |
| Exp 1: Patch=16 | 5 min          |
| Exp 1: Patch=32 | 3 min          |
| Exp 2: Input=168h | 2 min          |
| Exp 2: Input=336h | 3 min          |
| Exp 2: Input=672h | 6 min          |
| Exp 3: Horizon=24h | 4 min          |
| Exp 3: Horizon=96h | 6 min          |
| Exp 3: Horizon=336h | 5 min          |
| Final: Baseline | 30 min         |
| Final: PatchTST Sequence | 1.5 min        |
| Final: PatchTST Patch-wise | 1.5 min        |
| Final: aLLM4TS GPT-2 | 7 min          |

For testing: Use NUM_EPOCHS=10, PATIENCE=5 to verify everything works, then increase to NUM_EPOCHS=20/50 for final runs.

---

## Troubleshooting

### Data Issues

- "Data not found": Run `python data_preparation.py` first
- Missing values warning: Normal, handled automatically via time-based interpolation

### Training Issues

- "CUDA out of memory": Reduce BATCH_SIZE to 16 or 8
- GPT-2 download fails: Check internet connection, model downloads from HuggingFace on first run (~500MB)
- "Some weights of GPT2Model were not initialized": This warning is expected and suppressed
- FutureWarning about DataFrame.interpolate: Pandas version compatibility, safe to ignore

### Performance Issues

- Training very slow: Check DEVICE setting (use "cuda" for NVIDIA GPU, "mps" for Apple Silicon)

---

## Metrics

All experiments report:
- MSE (Mean Squared Error): Primary metric (lower is better)
- MAE (Mean Absolute Error): Secondary metric (lower is better)
- Per-station metrics: Separate performance for Schiphol and Maastricht

Metrics are calculated on denormalized data (original scale) for interpretability.

---

## Experimental Design

### Phase 1: Ablation Studies (Experiments 1-3)
- Focus on PatchTST Patchwise only
- Find optimal hyperparameters (PATCH_LEN, SEQ_LEN, PRED_LEN)
- 3 runs per configuration for robust statistics
- Combined plots showing both stations

### Phase 2: Final Comparison (Final Evaluation)
- Compare all 4 models using optimal settings
- 5 runs per model for statistical validity
- Demonstrates effectiveness of patch-wise decoding

### Phase 3: Analysis Generation
- Statistical significance tests with paired t-tests
- Training convergence visualization
- Confirms improvements are not due to random chance

### Statistical Robustness
- Multiple random seeds: Ensures results are not due to lucky initialization
- Per-station evaluation: Tests generalization across locations
- Early stopping: Prevents overfitting automatically
- Time-based splits: Realistic evaluation (no data leakage)
- Paired t-tests: Rigorous statistical comparison

---

## Technical Details

### Data Processing
1. Raw KNMI files to clean CSV with unit conversion
2. Combined multi-station data with station labels
3. Time-based interpolation for missing values (<0.1%)
4. StandardScaler normalization (zero mean, unit variance)
5. 70/15/15 time-based train/val/test split

### Model Architectures

- Baseline: Timestep projection, Transformer (3 layers, 128d, 16 heads), Flatten, Linear
- PatchTST Sequence-wise: Patching, Channel-independent Transformer, Flatten all patches, Linear
- PatchTST Patch-wise: Patching, Channel-independent Transformer, Select last N patches, Shared patch decoder
- aLLM4TS: Channel-dependent patching, Frozen GPT-2 (12 layers, 768d), Patch-wise decoder

### Key Features
- RevIN normalization: Stabilizes training for PatchTST models
- Channel-independent processing: Each feature processed separately
- Learnable positional encodings: For patch positions
- Early stopping: Monitors validation loss with patience
- Learning rate scheduling: ReduceLROnPlateau (factor 0.5, patience 5)

---

## Acknowledgments

- KNMI for providing open weather data
- PatchTST authors for the base architecture
- aLLM4TS authors for the patch-wise decoding concept
