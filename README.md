# Multi-stress Failure Simulation App

This repository contains the source code for a Streamlit-based visualization app for remaining useful life (RUL) prediction and failure-mechanism simulation under coupled environmental stresses.

The implementation uses the NASA C-MAPSS FD004 turbofan degradation dataset as the experimental data source and combines:

- LightGBM-based RUL prediction
- Weibull reliability modeling
- hidden Markov model (HMM) degradation-state simulation
- SHAP-based model interpretation
- Monte Carlo lifetime sampling

## Repository Scope

This public repository contains only the simulation program source code and usage instructions. Course reports, presentation slides, generated paper figures, runtime logs, and local deployment details are intentionally excluded.

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

The dataset files are not included in this public repository.

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

## Project Structure

```text
course_simulation/
├── app.py              # Streamlit app entry point
├── requirements.txt    # Python dependencies
├── data/
│   └── README.md       # Dataset placement instructions
├── src/
│   ├── __init__.py
│   └── engine.py       # Simulation and modeling pipeline
└── results/            # Runtime output directory
```

## License

This project is released under the MIT License.
