import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve

st.set_page_config(
    page_title="FlightPath to Loyalty | Airline Satisfaction Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .recommendation-box {
        background-color: #EFF6FF;
        border-left: 5px solid #2563EB;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .risk-high {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 12px;
        border-radius: 6px;
    }
    .risk-low {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        padding: 12px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("airline_passenger_satisfaction.csv")
    return df

@st.cache_resource
def load_models():
    rf = joblib.load("models/satisfaction_rf_model.joblib")
    lr = joblib.load("models/satisfaction_lr_model.joblib")
    return rf, lr

df = load_data()
rf_model, lr_model = load_models()

# Sidebar Navigation
st.sidebar.title("✈️ Navigation")
page = st.sidebar.radio(
    "Select Module:",
    [
        "📊 Executive KPI Dashboard",
        "🔍 Exploratory Data Analysis",
        "🤖 Machine Learning Diagnostics",
        "🎯 Feature Driver Analysis",
        "🎛️ Real-Time Risk Simulator",
        "💡 Prescriptive Business Roadmap"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("**FlightPath to Loyalty Analytics**\nDataset: 129,880 Passenger Records\nVersion: 1.0.0 (Production)")

# -------------------------------------------------------------
# 1. EXECUTIVE KPI DASHBOARD
# -------------------------------------------------------------
if page == "📊 Executive KPI Dashboard":
    st.markdown('<div class="main-header">✈️ FlightPath to Loyalty: Executive Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predictive Modeling & Touchpoint Driver Analysis for Airline Passenger Satisfaction</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_passengers = len(df)
    satisfied_count = (df['Satisfaction'] == 'Satisfied').sum()
    satisfaction_rate = (satisfied_count / total_passengers) * 100
    avg_dep_delay = df['Departure Delay'].mean()
    
    with col1:
        st.metric("Total Passengers Analyzed", f"{total_passengers:,}")
    with col2:
        st.metric("Overall Satisfaction Rate", f"{satisfaction_rate:.2f}%", delta=f"{satisfaction_rate-50:.1f}% vs baseline")
    with col3:
        st.metric("Neutral/Dissatisfied", f"{100 - satisfaction_rate:.2f}%", delta="-At Risk", delta_color="inverse")
    with col4:
        st.metric("Avg Departure Delay", f"{avg_dep_delay:.1f} mins")
    with col5:
        st.metric("Model Prediction Accuracy", "96.09%", delta="+8.38% vs LR")
        
    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("📌 Strategic Insights Summary")
        st.markdown("""
        - **Core Retention Threat**: **56.55% (73,452 passengers)** reported being neutral or dissatisfied, representing substantial customer churn vulnerability.
        - **Primary Satisfaction Drivers**: Random Forest driver decomposition confirms that **Online Boarding (18.11%)** and **In-flight WiFi Service (14.60%)** are the single strongest differentiators between loyal and at-risk travelers.
        - **Class & Travel Type Dynamics**: Business travelers who are returning customers exhibit a **70.62% satisfaction rate**, whereas personal travelers who are returning customers fall to **10.10% satisfaction**.
        - **Operational Delay Vulnerability**: Punctuality is essential, but baseline dissatisfaction remains **54.06% on on-time flights** when digital/in-flight touchpoints (WiFi, online boarding < 3) are poorly rated.
        """)
        
    with col_right:
        st.subheader("🎯 Satisfaction Split Overview")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        counts = df['Satisfaction'].value_counts()
        colors = ['#EF4444', '#10B981']
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=colors, startangle=140, explode=(0.04, 0))
        ax.set_title("Passenger Sentiment Distribution", fontsize=11, weight='bold')
        st.pyplot(fig)
        plt.close()

# -------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# -------------------------------------------------------------
elif page == "🔍 Exploratory Data Analysis":
    st.markdown('<div class="main-header">🔍 Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Empirical associations and touchpoint patterns across passenger cohorts</div>', unsafe_allow_html=True)
    
    eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
        "Travel Class & Purpose", "Service Touchpoints", "Operational Delays", "Demographics & Distance"
    ])
    
    with eda_tab1:
        st.subheader("Travel Class & Purpose Dynamics")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            c_sat = pd.crosstab(df['Class'], df['Satisfaction'], normalize='index') * 100
            c_sat[['Satisfied', 'Neutral or Dissatisfied']].plot(
                kind='bar', stacked=True, color=['#10B981', '#EF4444'], ax=ax
            )
            ax.set_title("Satisfaction Rate by Cabin Class", fontsize=12, weight='bold')
            ax.set_ylabel("Percentage (%)")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            st.pyplot(fig)
            plt.close()
            st.caption("**Interpretation**: Business Class passengers enjoy significantly higher satisfaction (69.44%) compared to Economy (18.77%) and Economy Plus (24.64%).")
            
        with col2:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            travel_cust = df.groupby(['Type of Travel', 'Customer Type'])['Satisfaction'].apply(lambda s: (s == 'Satisfied').mean() * 100).unstack()
            travel_cust.plot(kind='bar', ax=ax, color=['#3B82F6', '#F59E0B'], width=0.6)
            ax.set_title("Satisfaction Rate (%) by Purpose & Customer Type", fontsize=12, weight='bold')
            ax.set_ylabel("Satisfied Rate (%)")
            ax.set_ylim(0, 100)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            st.pyplot(fig)
            plt.close()
            st.caption("**Interpretation**: Returning business travelers report 70.62% satisfaction vs 24.04% for first-time business travelers, and ~10-16% for personal travel.")

    with eda_tab2:
        st.subheader("Service Touchpoint Rating Curves")
        services = [
            "Online Boarding", "In-flight Wifi Service", "In-flight Entertainment",
            "Seat Comfort", "Leg Room Service", "Cleanliness", "On-board Service", "Baggage Handling"
        ]
        selected_services = st.multiselect("Select Services to Compare:", services, default=services[:4])
        
        if selected_services:
            df_temp = df.copy()
            df_temp['Sat_Binary'] = (df_temp['Satisfaction'] == 'Satisfied').astype(int)
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for s in selected_services:
                valid = df_temp[df_temp[s] > 0]
                rate = valid.groupby(s)['Sat_Binary'].mean() * 100
                ax.plot(rate.index, rate.values, marker='o', linewidth=2.2, label=s)
            ax.set_title("Satisfaction Rate (%) Across Survey Rating Levels (1-5)", fontsize=12, weight='bold')
            ax.set_xlabel("Rating (1 = Low, 5 = High)")
            ax.set_ylabel("Satisfied Rate (%)")
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
            plt.close()
            st.caption("**Key Finding**: Ratings of 4 or 5 in Online Boarding (62.3% and 87.1%) and In-flight WiFi (60.1% and 99.0%) drive huge satisfaction gains, whereas scores <= 3 exhibit heavy dissatisfaction.")

    with eda_tab3:
        st.subheader("Flight Delay Impact Analysis")
        col1, col2 = st.columns(2)
        with col1:
            delay_bins = [-1, 0, 15, 60, 180, 2000]
            delay_labels = ["On-Time (0 min)", "1-15 min", "16-60 min", "61-180 min", "> 180 min"]
            df_del = df.copy()
            df_del['Delay_Group'] = pd.cut(df_del['Departure Delay'], bins=delay_bins, labels=delay_labels)
            del_summary = df_del.groupby('Delay_Group', observed=False)['Satisfaction'].value_counts(normalize=True).unstack() * 100
            
            fig, ax = plt.subplots(figsize=(7, 4.5))
            del_summary[['Neutral or Dissatisfied', 'Satisfied']].plot(
                kind='bar', color=['#EF4444', '#10B981'], ax=ax
            )
            ax.set_title("Satisfaction by Departure Delay Tier", fontsize=12, weight='bold')
            ax.set_ylabel("Percentage (%)")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
            st.pyplot(fig)
            plt.close()
            
        with col2:
            st.markdown("""
            #### Operational Delay Takeaways:
            - **Non-Linear Escalation**: Dissatisfaction increases from **54.06%** on on-time flights to **62.50%** for 16-60 min delays and **64.15%** for 1-3 hour delays.
            - **Tolerance Threshold**: Minor delays (<15 mins) maintain 43.55% satisfaction if boarding and digital amenities remain fluent.
            - **Recovery Window**: Automated recovery gestures should be initiated once delays surpass 30 minutes.
            """)

    with eda_tab4:
        st.subheader("Passenger Demographics & Flight Distance")
        fig, ax = plt.subplots(figsize=(9, 4.2))
        sns.kdeplot(data=df[df['Satisfaction'] == 'Satisfied'], x='Age', label='Satisfied', fill=True, color='#10B981', alpha=0.4, ax=ax)
        sns.kdeplot(data=df[df['Satisfaction'] == 'Neutral or Dissatisfied'], x='Age', label='Neutral/Dissatisfied', fill=True, color='#EF4444', alpha=0.4, ax=ax)
        ax.set_title("Age Distribution Density by Satisfaction Status", fontsize=12, weight='bold')
        ax.set_xlabel("Age (Years)")
        ax.legend()
        st.pyplot(fig)
        plt.close()
        st.caption("**Insight**: Passengers aged 40-60 show higher baseline satisfaction, whereas young adult travelers (18-32) report significantly lower satisfaction.")

# -------------------------------------------------------------
# 3. MACHINE LEARNING DIAGNOSTICS
# -------------------------------------------------------------
elif page == "🤖 Machine Learning Diagnostics":
    st.markdown('<div class="main-header">🤖 Machine Learning Model Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation of Supervised Classification Pipelines (80/20 Stratified Split)</div>', unsafe_allow_html=True)
    
    metrics_data = {
        "Metric": ["Accuracy", "Precision", "Recall (Sensitivity)", "F1-Score", "ROC-AUC Score"],
        "Logistic Regression (Baseline)": ["87.71%", "87.16%", "84.11%", "85.61%", "0.9295"],
        "Random Forest (Champion)": ["96.09%", "96.61%", "94.30%", "95.44%", "0.9934"],
        "Performance Delta": ["+8.38%", "+9.45%", "+10.19%", "+9.83%", "+0.0639"]
    }
    st.table(pd.DataFrame(metrics_data))
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("figures/7_confusion_matrices.png", caption="Normalized Confusion Matrix Comparison", use_container_width=True)
    with col2:
        st.image("figures/8_roc_pr_curves.png", caption="ROC & Precision-Recall Curves", use_container_width=True)
        
    st.markdown("---")
    st.subheader("⚖️ Business Context & Cost of Errors")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="risk-high">
            <strong>❌ False Positive (Predicted: Satisfied | Actual: Dissatisfied)</strong><br>
            <strong>Business Cost:</strong> High silent churn risk. The airline misses the opportunity to intervene, offer service recovery miles/vouchers, leading to lost lifetime customer value.
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="risk-low">
            <strong>⚠️ False Negative (Predicted: Dissatisfied | Actual: Satisfied)</strong><br>
            <strong>Business Cost:</strong> Modest operational expense. The airline sends unneeded retention incentives or check-in assistance to a customer who was already satisfied.
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# 4. FEATURE DRIVER ANALYSIS
# -------------------------------------------------------------
elif page == "🎯 Feature Driver Analysis":
    st.markdown('<div class="main-header">🎯 Feature Driver & Touchpoint Hierarchy</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Decomposition of the most influential predictors of passenger loyalty</div>', unsafe_allow_html=True)
    
    st.image("figures/9_feature_importances.png", caption="Random Forest Feature Importances (MDI / Gini)", use_container_width=True)
    
    st.markdown("### Top Drivers Breakdown & Evidence")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("""
        <div class="metric-card">
            <h4>1. Online Boarding (18.11%)</h4>
            <p>The #1 driver of satisfaction. Seamless mobile boarding passes and check-in apps set the baseline psychological expectation for the flight.</p>
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown("""
        <div class="metric-card">
            <h4>2. In-flight WiFi (14.60%)</h4>
            <p>Reliable connectivity is essential for modern travelers. Low WiFi ratings (<=3) correlate with 67-75% dissatisfaction.</p>
        </div>
        """, unsafe_allow_html=True)
    with d3:
        st.markdown("""
        <div class="metric-card">
            <h4>3. Travel Purpose & Class (20.52%)</h4>
            <p>Personal travel in Economy class carries the largest baseline dissatisfaction risk (81.23%). Physical seat comfort and legroom are crucial differentiators.</p>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# 5. REAL-TIME RISK SIMULATOR
# -------------------------------------------------------------
elif page == "🎛️ Real-Time Risk Simulator":
    st.markdown('<div class="main-header">🎛️ Live Passenger Satisfaction Risk Scoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Simulate passenger journey parameters and predict satisfaction probability in real-time</div>', unsafe_allow_html=True)
    
    with st.form("simulation_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            customer_type = st.selectbox("Customer Type", ["Returning", "First-time"])
            age = st.slider("Age", 7, 85, 38)
        with c2:
            travel_type = st.selectbox("Type of Travel", ["Business", "Personal"])
            travel_class = st.selectbox("Class", ["Business", "Economy", "Economy Plus"])
            distance = st.number_input("Flight Distance (miles)", 30, 5000, 1200)
        with c3:
            dep_delay = st.number_input("Departure Delay (mins)", 0, 1500, 0)
            arr_delay = st.number_input("Arrival Delay (mins)", 0, 1500, 0)
            online_boarding = st.slider("Online Boarding (0=N/A, 1-5)", 0, 5, 4)
            wifi = st.slider("In-flight Wifi (0=N/A, 1-5)", 0, 5, 4)
        with c4:
            seat_comfort = st.slider("Seat Comfort (0=N/A, 1-5)", 0, 5, 4)
            entertainment = st.slider("In-flight Entertainment (0=N/A, 1-5)", 0, 5, 4)
            legroom = st.slider("Leg Room Service (0=N/A, 1-5)", 0, 5, 4)
            cleanliness = st.slider("Cleanliness (0=N/A, 1-5)", 0, 5, 4)
            
        submitted = st.form_submit_button("🚀 Run Predictive Scoring Engine")
        
    if submitted:
        input_data = pd.DataFrame([{
            'Gender': gender,
            'Age': age,
            'Customer Type': customer_type,
            'Type of Travel': travel_type,
            'Class': travel_class,
            'Flight Distance': distance,
            'Departure Delay': dep_delay,
            'Arrival Delay': arr_delay,
            'Departure and Arrival Time Convenience': 3,
            'Ease of Online Booking': 3,
            'Check-in Service': 4,
            'Online Boarding': online_boarding,
            'Gate Location': 3,
            'On-board Service': 4,
            'Seat Comfort': seat_comfort,
            'Leg Room Service': legroom,
            'Cleanliness': cleanliness,
            'Food and Drink': 3,
            'In-flight Service': 4,
            'In-flight Wifi Service': wifi,
            'In-flight Entertainment': entertainment,
            'Baggage Handling': 4
        }])
        
        prob_satisfied = rf_model.predict_proba(input_data)[0, 1]
        prediction = "Satisfied" if prob_satisfied >= 0.5 else "Neutral or Dissatisfied"
        
        st.markdown("### 📊 Prediction Results")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if prediction == "Satisfied":
                st.success(f"**Predicted Status**: {prediction}")
            else:
                st.error(f"**Predicted Status**: {prediction}")
            st.metric("Satisfaction Probability", f"{prob_satisfied * 100:.1f}%")
            
        with res_col2:
            st.progress(prob_satisfied)
            if prob_satisfied < 0.5:
                st.warning("⚠️ **High Churn Risk Triggered**: Proactive recovery actions recommended.")
            else:
                st.info("✅ **Low Churn Risk**: Passenger likely to remain loyal.")

# -------------------------------------------------------------
# 6. PRESCRIPTIVE BUSINESS ROADMAP
# -------------------------------------------------------------
elif page == "💡 Prescriptive Business Roadmap":
    st.markdown('<div class="main-header">💡 Prescriptive Business Strategy & Action Plan</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Data-driven recommendations for airline operations and customer experience leadership</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="recommendation-box">
        <h3>1. Digital Gateway Overhaul (Online Boarding & Mobile App)</h3>
        <p><strong>Rationale:</strong> Online boarding is the single most important factor (18.11% feature importance). Upgrading digital boarding pass issuance, automated seat selection, and biometric boarding delivers the highest ROI on satisfaction.</p>
    </div>
    
    <div class="recommendation-box">
        <h3>2. Next-Gen In-Flight Connectivity (WiFi Tiering)</h3>
        <p><strong>Rationale:</strong> In-flight WiFi service is the second largest driver (14.60% importance). Transition from high-friction paid models to complimentary high-speed messaging/browsing for returning and business travelers.</p>
    </div>
    
    <div class="recommendation-box">
        <h3>3. Economy Class Comfort Upgrades</h3>
        <p><strong>Rationale:</strong> Economy passengers have an alarming 81.23% dissatisfaction rate. Ergonomic seat cushions, optimized legroom presets, and refreshed cabin cleanliness will reduce churn among leisure travelers.</p>
    </div>
    
    <div class="recommendation-box">
        <h3>4. Proactive Delay Recovery Protocol</h3>
        <p><strong>Rationale:</strong> Model simulations reveal that satisfaction drops significantly when delays exceed 15-30 minutes. Automate instant meal vouchers or frequent flyer bonus miles pushed directly to mobile apps when flight delays occur.</p>
    </div>
    """, unsafe_allow_html=True)
