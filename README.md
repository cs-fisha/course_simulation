# Multi-stress Failure Simulation App

This repository contains the source code for a Streamlit-based visualization app for remaining useful life (RUL) prediction and failure-mechanism simulation under coupled environmental stresses.

中文说明见 [README_zh.md](README_zh.md).

The implementation uses the NASA C-MAPSS FD004 turbofan degradation dataset as the experimental data source and combines:

- LightGBM-based RUL prediction
- Weibull reliability modeling
- hidden Markov model (HMM) degradation-state simulation
- SHAP-based model interpretation
- Monte Carlo lifetime sampling

## Repository Scope

This course-project repository contains the simulation program, the experiment paper, the presentation slides, and the supporting figures used for submission.

## Requirements

- Python 3.10+
- Windows, macOS, or Linux
- No GPU required

Install dependencies:

```bash
pip install -r requirements.txt
```

On high-core-count servers, limiting numerical library threads is recommended:

```bash
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export LGBM_NUM_THREADS=4
```

## Data Preparation

Download or prepare the NASA C-MAPSS FD004 files and place them under `data/`:

```text
data/
├── train_FD004.txt
├── test_FD004.txt
└── RUL_FD004.txt
```

For the current course submission snapshot, the FD004 files are already placed in `data/`.

## Run

Start the visualization app:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LGBM_NUM_THREADS=4 streamlit run app.py
```

For remote access, bind Streamlit to all interfaces:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LGBM_NUM_THREADS=4 \
streamlit run app.py \
  --server.headless true \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.enableCORS false \
  --server.enableXsrfProtection false
```

The first visit loads data and trains the models. Later interactions reuse the cached pipeline.

## Build Paper and Slides

Compile the course paper:

```bash
cd paper
xelatex -interaction=nonstopmode -halt-on-error paper.tex
```

Compile the presentation slides:

```bash
cd slide
xelatex -interaction=nonstopmode -halt-on-error presentation.tex
```

## Features

1. **Weibull Reliability Simulation**  
   Compare reliability curves under different stress types and stress levels.

2. **Degradation State Evolution**  
   Visualize HMM state trajectories and forward state-probability prediction for selected units.

3. **RUL Prediction Comparison**  
   Compare single-stress models with the coupled multi-stress model using line and scatter plots.

4. **Monte Carlo Lifetime Simulation**  
   Adjust stress multipliers and observe how the sampled lifetime distribution changes.

5. **Model Metrics**  
   Inspect RMSE, MAE, R², ±20-cycle accuracy, and PHM08 score.

6. **SHAP Model Interpretation**  
   Inspect global feature importance and dependency plots for the coupled model.

## Project Structure

```text
course_simulation/
├── app.py              # Streamlit app entry point
├── requirements.txt    # Python dependencies
├── README_zh.md        # Chinese project guide
├── SUBMISSION_CHECKLIST.md
├── data/
│   ├── README.md       # Dataset placement instructions
│   ├── train_FD004.txt
│   ├── test_FD004.txt
│   └── RUL_FD004.txt
├── paper/              # Course paper source and PDF
├── slide/              # Presentation source, PDF, and notes
├── scripts/            # Figure-generation helpers
├── src/
│   ├── __init__.py
│   └── engine.py       # Simulation and modeling pipeline
└── results/            # Runtime output directory
```

## License

This project is released under the MIT License.
