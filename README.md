# FlightPath to Loyalty: Predictive Modeling and Driver Analysis of Airline Passenger Satisfaction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)

---

## 📖 Project Overview
Modern airline profitability and passenger retention directly hinge on end-to-end travel experience. While airline operations routinely measure punctuality and operational statistics, understanding the multidimensional drivers that tilt passenger sentiment from dissatisfaction to loyalty requires rigorous empirical analytics and machine learning.

This project delivers an enterprise-grade Data Analytics and Supervised Machine Learning system built upon **129,880 passenger survey records**. We investigate touchpoint pain points across the entire customer journey, benchmark non-linear ensemble models against regularized linear baselines, and provide interpretable driver rankings alongside actionable prescriptive strategies for airline operations.

---

## 🎯 Problem Statement
1. **Diagnostic Analytics**: Which passenger demographics, travel classes, journey touchpoints (e.g., online boarding, in-flight WiFi, seat comfort, legroom), and operational flight delays most strongly correlate with passenger dissatisfaction?
2. **Predictive Machine Learning Engine**: Build, validate, and deploy a robust binary classification pipeline that accurately classifies whether a passenger is **Satisfied** ($y=1$) or **Neutral/Dissatisfied** ($y=0$), enabling proactive service intervention before silent churn occurs.

---

## 📊 Dataset Description
The analysis utilizes the official airline passenger satisfaction survey dataset consisting of **129,880 observations** across **24 features** (no synthetic or fabricated data).

- **Source / Reference**: [Kaggle Airline Passenger Satisfaction Dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
- **Observations (Rows)**: `129,880`
- **Variables (Columns)**: `24`
- **Target Variable**: `Satisfaction` (Binary: `Satisfied` [43.45%, 56,428] vs. `Neutral or Dissatisfied` [56.55%, 73,452])

### Attribute Categories:
- **Identifier**: `ID` (Surrogate integer key, excluded from predictors)
- **Demographic & Flight Profile**: `Gender`, `Age`, `Customer Type`, `Type of Travel`, `Class`, `Flight Distance`
- **Operational Metrics**: `Departure Delay` (minutes), `Arrival Delay` (minutes)
- **14 Service Rating Touchpoints (0–5 Likert Scale, where 0 = "Not Applicable")**:
  - `Departure and Arrival Time Convenience`, `Ease of Online Booking`, `Check-in Service`, `Online Boarding`, `Gate Location`, `On-board Service`, `Seat Comfort`, `Leg Room Service`, `Cleanliness`, `Food and Drink`, `In-flight Service`, `In-flight Wifi Service`, `In-flight Entertainment`, `Baggage Handling`

---

## 🏗️ 4-Tier Methodology

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      4-TIER ANALYTICS ARCHITECTURE                      │
├───────────────────┬───────────────────┬────────────────┬────────────────┤
│      TIER 1       │      TIER 2       │     TIER 3     │     TIER 4     │
│  Data Quality &   │ Exploratory Data  │ Machine Learn. │  Prescriptive  │
│     Auditing      │     Analytics     │ & Diagnostics  │ Strategy & ROI │
├───────────────────┼───────────────────┼────────────────┼────────────────┤
│ • Missing audit   │ • Cabin & travel  │ • 80/20 train/ │ • High-risk    │
│ • "0 = N/A" logic │   type breakdowns │   test split   │   segmentation │
│ • Delay anomaly   │ • Service curves  │ • RF vs LR     │ • Driver rank  │
│   verification    │ • Delay impact    │ • ROC-AUC &    │ • Automated    │
│ • No data leakage │ • Demographic KDE │   PR curves    │   triage rules │
└───────────────────┴───────────────────┴────────────────┴────────────────┘
```

1. **Tier 1: Data Quality & Preprocessing**:
   - Imputed `Arrival Delay` (393 nulls, 0.30%) using median values fitted strictly on training data folds.
   - Preserved valid extreme delay durations without blind deletion.
   - Accounted for `0` ratings as non-applicable responses per the data dictionary.
   - Encoded nominal categoricals via `OneHotEncoder(drop='first')` and scaled numericals via `StandardScaler` inside a unified `ColumnTransformer`.
2. **Tier 2: Exploratory Data Analytics (EDA)**:
   - Generated 6 high-resolution empirical visualizations illustrating sentiment patterns by cabin class, travel purpose, delay tiers, age density, and touchpoint rating inflection points.
3. **Tier 3: Supervised Machine Learning & Evaluation**:
   - Benchmarked **Random Forest Classifier** against **Logistic Regression**.
   - Evaluated via Confusion Matrix, Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
4. **Tier 4: Prescriptive Business Strategy & Simulation**:
   - Model feature importance ranking via MDI / Gini reduction.
   - Developed a decision rule for passenger churn mitigation and a real-time Streamlit risk simulator.

---

## 🛠️ Technologies Used
- **Core Language**: Python 3.10+
- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`, `joblib`
- **Visualization**: `matplotlib`, `seaborn`
- **Web Application**: `streamlit`

---

## 📂 Project Directory Structure

```
Airline_Passenger_Satisfaction_Project/
│
├── airline_passenger_satisfaction.csv      # Main raw dataset (129,880 rows)
├── data_dictionary.csv                     # Official column definitions reference
├── requirements.txt                        # Production dependency manifest
├── README.md                               # Comprehensive documentation
├── app.py                                  # Interactive Streamlit analytics application
│
├── Piyush_FlightPath_to_Loyalty.ipynb      # End-to-end executed Jupyter Notebook
│
├── models/                                 # Serialized model artifacts
│   ├── satisfaction_rf_model.joblib        # Champion Random Forest pipeline
│   └── satisfaction_lr_model.joblib        # Baseline Logistic Regression pipeline
│
└── figures/                                # High-resolution visual artifacts
    ├── 1_overall_satisfaction.png
    ├── 2_satisfaction_by_class.png
    ├── 3_travel_type_customer_type.png
    ├── 4_service_ratings_curve.png
    ├── 5_departure_delays_impact.png
    ├── 6_age_satisfaction_distribution.png
    ├── 7_confusion_matrices.png
    ├── 8_roc_pr_curves.png
    └── 9_feature_importances.png
```

---

## 🚀 Installation & Setup Instructions

### 1. Clone or Navigate to the Workspace:
```bash
cd Airline_Passenger_Satisfaction_Project
```

### 2. Set Up a Virtual Environment:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies:
```bash
pip install -r requirements.txt
```

---

## 💻 How to Run

### Run the Jupyter Notebook:
```bash
jupyter notebook Piyush_FlightPath_to_Loyalty.ipynb
```

### Launch the Streamlit Interactive Dashboard:
```bash
streamlit run app.py
```
*The dashboard will automatically open in your default browser at `http://localhost:8501`.*

---

## 🤖 Machine Learning Approach & Benchmark

The dataset was partitioned using an **80/20 stratified split** (`random_state=42`), preserving class ratios across folds:
- **Training Set**: 103,904 observations
- **Testing Set**: 25,976 observations

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 87.71% | 87.16% | 84.11% | 85.61% | 0.9295 |
| **Random Forest (Champion)** | **96.09%** | **96.61%** | **94.30%** | **95.44%** | **0.9934** |
| **Performance Delta** | **+8.38%** | **+9.45%** | **+10.19%** | **+9.83%** | **+0.0639** |

---

## 🎯 Key Empirical Findings & Exact Dataset Metrics

### 1. Purpose of Travel × Customer Type Interaction:
- **Business Travel & Returning Customers**: **70.62% Satisfied** (29.38% Dissatisfied) — *Highest satisfaction segment*.
- **Business Travel & First-time Customers**: **24.04% Satisfied** (75.96% Dissatisfied).
- **Personal Travel & First-time Customers**: **15.92% Satisfied** (84.08% Dissatisfied).
- **Personal Travel & Returning Customers**: **10.10% Satisfied** (89.90% Dissatisfied) — *Lowest satisfaction segment*.

### 2. Operational Departure Delay Impact Tiers:
- **0 min (On-Time)**: **54.06% Dissatisfied** / **45.94% Satisfied**
- **1–15 min**: **56.45% Dissatisfied** / **43.55% Satisfied**
- **16–60 min**: **62.50% Dissatisfied** / **37.50% Satisfied**
- **61–180 min**: **64.15% Dissatisfied** / **35.85% Satisfied**
- **> 180 min**: **64.11% Dissatisfied** / **35.89% Satisfied**

### 3. Service Touchpoint Rating Inflection Points:
- **Online Boarding**:
  - Rating 1: **13.78% Satisfied** (86.22% Dissatisfied)
  - Rating 2: **11.45% Satisfied** (88.55% Dissatisfied)
  - Rating 3: **13.76% Satisfied** (86.24% Dissatisfied)
  - Rating 4: **62.30% Satisfied** (37.70% Dissatisfied)
  - Rating 5: **87.06% Satisfied** (12.94% Dissatisfied)
- **In-flight Wifi Service**:
  - Rating 1: **32.82% Satisfied** (67.18% Dissatisfied)
  - Rating 2: **24.72% Satisfied** (75.28% Dissatisfied)
  - Rating 3: **25.18% Satisfied** (74.82% Dissatisfied)
  - Rating 4: **60.08% Satisfied** (39.92% Dissatisfied)
  - Rating 5: **99.02% Satisfied** (0.98% Dissatisfied)

### 4. Top Feature Importances (Random Forest MDI / Gini):
1. **Online Boarding (18.11%)**
2. **In-flight Wifi Service (14.60%)**
3. **Type of Travel: Personal (12.91%)**
4. **Class: Economy (7.61%)**
5. **In-flight Entertainment (6.49%)**
6. **Seat Comfort (4.89%)**
7. **Customer Type: Returning (4.38%)**
8. **Ease of Online Booking (4.02%)**
9. **Leg Room Service (3.80%)**

---

## 💡 Prescriptive Business Recommendations

1. **Digital Boarding Gateway Modernization (Priority 1)**:
   - Allocate capital expenditure toward the mobile app boarding pass workflow, automated seat reallocation, and biometric fast-track gate boarding (highest ROI intervention).
2. **Tiered In-Flight Connectivity Architecture (Priority 2)**:
   - Introduce free messaging (WhatsApp/iMessage) fleet-wide and subsidized high-speed browsing packages for loyalty program members.
3. **Economy Cabin Ergonomics & Cleanliness (Priority 3)**:
   - Target quick-turn cabin sanitization and ergonomic seat enhancements in Economy configurations.
4. **Automated Real-Time Service Recovery Rule (Priority 4)**:
   - **Operational Decision Rule**: If a passenger is predicted with **Dissatisfaction Risk $> 60\%$** OR encounters a **Departure Delay $> 30$ minutes** with low digital ratings ($\le 2$), the airline CRM automatically triggers an immediate service recovery gesture (e.g., 1,000 bonus loyalty miles, complimentary lounge pass, or meal voucher).

---

## ⚖️ Methodological & Ethical Notes
- **Correlation vs. Causation**: Findings reflect statistical associations within observational survey data. Operational initiatives should be validated through A/B testing before fleet-wide rollouts.
- **Model Governance**: To maintain predictive fidelity, retraining pipelines should be executed quarterly to detect feature drift in passenger preferences.

---
**Delivered by:** Piyush | Principal Data Analyst & Machine Learning Engineer
