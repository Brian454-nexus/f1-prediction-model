# F1 APEX — Advanced Predictive Engine for F1 Race Outcomes
**Xtension Labs × Antigravity | 2026 Season Precision**

F1 APEX is a production-grade, highly modular F1 prediction engine custom-built for the **2026 regulatory era**. It orchestrates telemetry ingestion, 9 distinct machine-learning and heuristic modules, and Bayesian post-race calibration to output highly accurate race predictions, driver rationales, and social-ready graphics.

---

## 🏎️ Core Architecture

The system is split into **Data Ingestion**, **ML Evaluation**, and **Orchestration / Serving**.

### 1. Data Layer
- **`data_pipeline.py`**: Ingests official FOM telemetry via `FastF1`. Caches laps to Parquet for fast, offline, and resumable downstream processing. Fully supports the 2026 Sprint format (`SQ` and `S` sessions).
- **`feature_engineering.py`**: Computes 2026-critical metrics: Minimum Corner Speed (MCS), Energy Harvest Index (EHI), Peak Deployment Efficiency (PDE), and Qualifying Tyre Degradation (QTD).
- **`weather_router.py`**: Queries OpenMeteo APIs (T-3h) to route predictions to dry or wet-weather ML branches.
- **`fia_scraper.py`**: Defensively scrapes the FIA document feed to automatically apply grid penalties before race prediction.

### 2. ML Ensemble (9 Modules)
The Meta-Ensemble aggregates features from 8 specialized sub-models via a 50,000-run Monte Carlo simulation:
1. **CPM (Constructor Performance Model)**: Bayesian hierarchical model scoring true pace relative to the field.
2. **DPI (Driver Performance Index)**: Elo-based module assessing the "value-added" by a driver. Updates automatically after sprints ($K=20$) and races ($K=40$). Features symmetric 2026 rookie CI priors and mechanical DNF protection.
3. **ESS (Energy Strategy System)**: Tri-module system modelling 2026 hybrid battery deployment. Includes Circuit Profiling (`ehi`), an MDP Solver for optimal deployment, and a Monte Carlo simulator for clipping vs. yoyo risk.
4. **RRM (Reliability Risk Model)**: Cox survival analysis evaluating mechanical failure risk based on circuit thermal stress, altitude (e.g., Mexico), and team historical norms.
5. **CDC (Circuit DNA Classifier)**: K-Means model clustering circuits into archetypes (e.g., Power/Stop-Go, High-Downforce/Twisty) to modify Constructor pace.
6. **QRT (Qualifying-to-Race Translator)**: Ordinal regression mapping grid position and energy advantage to expected race finish ranges.
7. **Wet-Weather Model**: Activates if $P(rain) > 30\%$. Reweights the ensemble to heavily favour DPI (Driver Skill) and lowers Constructor impact, mirroring chaotic wet-race mechanics.
8. **MetaEnsemble**: The final combining layer. Merges sub-model predictions via MC sampling to output Top-3, Top-10, Win, and DNF probabilities with $P10$/$P90$ confidence intervals.
9. **Calibration (AutoCalibrator)**: Post-race Bayesian update step using Spearman Rank correlation to continuously tune sub-model weights for the next round.

### 3. Explanation & API
- **`explainer.py`**: Uses a SHAP-inspired ruleset to generate Natural Language rationale for *why* a driver was assigned a specific prediction (e.g., "Elite DPI score offset poor energy strategy").
- **FastAPI Application (`api/main.py`)**: Exposes REST endpoints (`/predict`, `/health`, `/calibration`).
- **Prediction Card Generator**: Generates aesthetic, social-media ready PNG output cards via Python Pillow.

### 4. Orchestration & State
- **Prefect DAG (`orchestration/race_weekend_dag.py`)**: Automates the weekend flow: `FP1 → (SQ → Sprint) → FP2/FP3 → Quali → FIA Scrape → Weather Check → Predict → GCS Sync`.
- **GCS Sync (`utils/gcs_sync.py`)**: Synchronises local JSON model state to Google Cloud Storage to guarantee statelessness and resumability on Cloud Run.

---

## 🛠️ Installation & Setup

1. **Clone & Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Environment Variables**
   Ensure your `.env` file is populated with GCP credentials and bucket names:
   ```env
   GCP_PROJECT=xtension-f1-apex
   GCS_BUCKET=f1-apex-state-bucket-2026
   FF1_CACHE_DIR=cache/fastf1
   ```
3. **GCP Service Account Auth**
   The `.gitignore` is configured to ignore `f1-apex-*.json`. Ensure your service account key is securely placed in the root directory and pointed to by `GOOGLE_APPLICATION_CREDENTIALS` (if running locally outside a container).

---

## 🚀 Running the Pipeline

### Option A: Fully Automated Race Weekend Pipeline

**1. Friday: Live Free Practice Ingestion**
To automatically collect FP1 and FP2 telemetry the moment it resolves on the CDN, leave the background daemon running on your machine:
```bash
python orchestration/daemon_runner.py
```
*(The daemon sleeps silently, wakes up exactly when sessions finish, parses the telemetry, extracts Long-Run Pace, and updates the Bayesian state on your drive).*

**2. Saturday: Final Prediction Generation**
Once Qualifying has concluded, execute the main Directed Acyclic Graph (DAG) to generate the Sunday grid:
```bash
python orchestration/race_weekend_dag.py
```
*(This automatically ingests Qualifying, downloads FIA penalty docs, checks weather, runs the 50,000 Monte Carlo sims, outputs the `rich` terminal prediction table, and syncs state to GCS).*

**Optional: Manual Single-Session Ingestion**
If your laptop was asleep and you want to trigger a single session manually right away:
```bash
# Example: Manually ingest FP1
python -c "from orchestration.race_weekend_dag import task_ingest_session; task_ingest_session.fn(3, 'FP1')"

# Example: Manually ingest FP2 (and run long-pace analysis)
python -c "from orchestration.daemon_runner import trigger_ingestion; trigger_ingestion(3, 'FP2')"

# Example: Manually ingest FP3 (and run high-weight pace analysis)
python -c "from orchestration.daemon_runner import trigger_ingestion; trigger_ingestion(3, 'FP3')"
```

### Option B: API Server (Interactive & Frontend Support)
To launch the FastAPI backend server:
```bash
uvicorn api.main:app --reload
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Generate Prediction Card (returns PNG): `http://localhost:8000/export/prediction-card`

### Option C: Manual CLI Testing
To test the core sub-models individually:

```bash
# 1. Test data ingestion for a specific session
python data/data_pipeline.py --year 2026 --round 3 --session Q

# 2. View current Driver Performance Index (DPI) Standings (22-Driver Grid)
python models/dpi.py

# 3. View automated Race Strategy decisions (ESS system)
python models/ess/ess_mdp.py

# 4. Generate SHAP-like explanations for driver performances
python models/explainer.py

# 5. Run the full unit/integration test suite
pytest tests -v
```

### Option D: AI Session Insights (Claude 4.6 Sonnet)
To generate a professional Race Engineer's narrative summary using telemetry and live news:

1. **Add your API Key**: Update `ANTHROPIC_API_KEY` in your `.env` file.
2. **Generate Report**:
```bash
# Get a report for a specific team in a session
python -m orchestration.reporter --session FP2 --scope "Mercedes"

# Get a report for the Top 5 teams
python -m orchestration.reporter --session FP3 --scope "Top 5"

# Get a report specifically for the qualifying session (Top 5 teams)
python -m orchestration.reporter --session Q --scope "Top 5"

# Get a full weekend high-level summary (FP1 through Qualifying)
python -m orchestration.reporter --summary --scope "All Teams"
```
*(These reports autonomously fetch live technical news using `duckduckgo_search` to provide the "Human Story" context based on your chosen scope).*

---

*Engineered by Xtension Labs × Antigravity.*
