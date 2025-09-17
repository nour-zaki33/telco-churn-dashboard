import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


                          # Page Configuration
st.set_page_config(
    page_title="Telco Customer Churn Analysis", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1f77b4; padding: 10px 0;}
    .section-header {font-size: 1.8rem; color: #1f77b4; border-bottom: 2px solid #1f77b4; padding: 10px 0;}
    .highlight-box {background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;}
    .metric-card {background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid #1f77b4; margin: 5px 0;}
    .info-text {font-size: 1.1rem; line-height: 1.6;}
</style>
""", unsafe_allow_html=True)

                              # Data Loading and Preprocessing 
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Finance\Desktop\telcom\WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df_encoded = df.copy()
    df_encoded = df_encoded.drop("customerID", axis=1)

    label_encoder = LabelEncoder()
    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = label_encoder.fit_transform(df_encoded[col])

    X = df_encoded.drop("Churn", axis=1)
    y = df_encoded["Churn"]
    return X, y, df

X, y, raw_df = load_data()


           # Split data into training and testing sets 

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)                  
                                         # Header Section


st.markdown('<h1 class="main-header">📊 Telco Customer Churn Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="highlight-box">
    <p class="info-text">This dashboard helps you understand customer churn patterns, predict which customers might leave, 
    and find the best strategies to keep them. Use the options on the left to explore different aspects of your data.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration

with st.sidebar:
    st.header("🔧 Settings")
    
    st.subheader("Prediction Model")
    option = st.radio(
        "Choose a prediction model:",
        ("Logistic Regression", "Random Forest"),
        help="Logistic Regression is simpler and faster, while Random Forest can find more complex patterns"
    )
                     # model selection

    if option == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        model_info = "Good for understanding simple relationships"
    else:
        model = RandomForestClassifier(random_state=42)
        model_info = "Better for complex patterns, but harder to interpret"
    
    st.info(f"**{option}**: {model_info}")
    
    st.divider()
    
    st.subheader("📊 Data Overview")
    st.metric("Total Customers", f"{len(raw_df):,}")
    churn_rate = (raw_df['Churn'] == 'Yes').mean()
    st.metric("Churn Rate", f"{churn_rate:.1%}", 
              help="Percentage of customers who have left the service")
    
    st.divider()
    


                                     # Main Content - Tabs

tab1, tab2, tab3 = st.tabs(["👥 Customer Overview", "📈 Prediction Model", "💰 Discount Simulator"])

with tab1:
    st.markdown('<h2 class="section-header">Customer Overview</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight-box">
        <p class="info-text">Understanding your customer base is the first step to reducing churn. 
        Here's an overview of your customers and their behavior patterns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1: 
        st.subheader("Customer Distribution")
        churn_count = raw_df["Churn"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#2E86AB', '#A23B72']
        wedges, texts, autotexts = ax.pie(churn_count.values, 
                                         labels=['Stay', 'Leave'], 
                                         autopct='%1.1f%%',
                                         colors=colors,
                                         startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
        st.caption("This shows what percentage of your customers are staying vs leaving.")

    with col2:
        st.subheader("Churn by Contract Type")
        fig, ax = plt.subplots(figsize=(10, 6))
        contract_churn = pd.crosstab(raw_df['Contract'], raw_df['Churn'], normalize='index') * 100
        contract_churn.plot(kind='bar', ax=ax, color=['#2E86AB', '#A23B72'])
        plt.xticks(rotation=45)
        plt.ylabel('Percentage (%)')
        plt.xlabel('Contract Type')
        plt.legend(['Stay', 'Leave'])
        st.pyplot(fig)
        st.caption("Customers with monthly contracts are much more likely to leave than those with longer contracts.")

    st.subheader("Key Insights")
    st.success("""
    -  **Monthly contract customers** are 3x more likely to leave than yearly contract customers
    -  **Higher monthly charges** are associated with higher churn rates
    -  **New customers** (0-10 months) are more likely to leave than long-term customers
    -  **Fiber optic internet** customers have higher churn rates than other service types
    """)

with tab2:
    st.markdown('<h2 class="section-header">Churn Prediction Model</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight-box">
        <p class="info-text">Our prediction model helps identify customers who might leave your service. 
        This allows you to take action before they decide to cancel.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Train and evaluate model
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Model Accuracy", f"{acc:.1%}", 
                 help="How often the model correctly predicts if a customer will stay or leave")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Precision for 'Leave'", f"{report['1']['precision']:.1%}", 
                 help="When the model predicts a customer will leave, how often is it correct")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Recall for 'Leave'", f"{report['1']['recall']:.1%}", 
                 help="What percentage of customers who actually left were correctly identified")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("Prediction Results")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), 
                    annot=True, fmt="d", 
                    cmap="Blues", 
                    xticklabels=['Stay', 'Leave'],
                    yticklabels=['Stay', 'Leave'],
                    ax=ax)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)
        st.caption("This shows how many predictions were correct and incorrect.")
    
    if option == "Random Forest":
        st.subheader("What Influences Churn Most?")
        importances = model.feature_importances_
        features = x_train.columns
        feat_imp = sorted(zip(importances, features), reverse=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=[val[0] for val in feat_imp[:8]], 
                   y=[val[1] for val in feat_imp[:8]], 
                   ax=ax, palette='viridis')
        plt.xlabel('Importance Score')
        plt.title('Top Factors Influencing Churn')
        st.pyplot(fig)
        
        st.info("""
        **Key factors that predict churn:**
        1. **Contract type** - Monthly contracts have highest churn
        2. **Tenure** - Newer customers are more likely to leave
        3. **Monthly charges** - Higher charges correlate with higher churn
        4. **Internet service** - Fiber optic customers churn more
        """)

with tab3:
    st.markdown('<h2 class="section-header">Discount Strategy Simulator</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight-box">
        <p class="info-text">Use this simulator to test different discount strategies for converting 
        monthly customers to annual contracts. Find the optimal discount that maximizes your revenue.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filter Month-to-month customers
    monthly_customers = raw_df[raw_df["Contract"] == "Month-to-month"].copy()
    monthly_customers["CLV_Current"] = monthly_customers["MonthlyCharges"] * monthly_customers["tenure"]
    
    st.subheader("Simulation Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tenure_yearly = st.slider(
            "Expected tenure for yearly contracts (months)", 
            12, 36, 24,
            help="How long you expect yearly contract customers to stay with you"
        )

    with col2:
        base_conversion = st.slider(
            "Base conversion rate without discount (%)", 
            5, 50, 20,
            help="What percentage would convert without any discount"
        ) / 100

    with col3:
        discount_sensitivity = st.slider(
            "Discount sensitivity", 
            0.5, 5.0, 1.5, 0.5,
            help="How much conversion increases for each 1% discount"
        )
    
                          # Run simulation


    discount_range = np.arange(0, 0.51, 0.01)
    simulation_results = []

    for d in discount_range:
        conversion_rate = min(1.0, base_conversion + d * discount_sensitivity)
        monthly_customers["CLV_New"] = monthly_customers["MonthlyCharges"] * tenure_yearly * (1 - d)
        monthly_customers["Uplift"] = monthly_customers["CLV_New"] - monthly_customers["CLV_Current"]
        
        n_convert = int(len(monthly_customers) * conversion_rate)
        converted_customers = monthly_customers.nlargest(n_convert, "Uplift")
        
        total_gain = converted_customers["Uplift"].sum()
        
        simulation_results.append({
            "Discount": d*100,
            "Conversion Rate": conversion_rate*100,
            "Converted Customers": n_convert,
            "Net Gain": total_gain
        })

    sim_df = pd.DataFrame(simulation_results)
    best_row = sim_df.loc[sim_df["Net Gain"].idxmax()]
    
    st.subheader("Optimal Discount Strategy")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Recommended Discount", f"{best_row['Discount']:.0f}%")

    with col2:
        st.metric("Expected Conversion", f"{best_row['Conversion Rate']:.1f}%")

    with col3:
        st.metric("Customers Converted", f"{best_row['Converted Customers']:,}")

    with col4:
        st.metric("Projected Net Gain", f"${best_row['Net Gain']:,.0f}")
    
    st.info(f"""
    **Strategy Recommendation:** Offer a **{best_row['Discount']:.0f}% discount** to convert monthly customers to yearly contracts.
    - This should convert approximately **{best_row['Converted Customers']:,} customers**
    - Generating a net gain of **${best_row['Net Gain']:,.0f}** in customer lifetime value
    """)
    
    # Visualizations
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Net Gain vs Discount")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sim_df["Discount"], sim_df["Net Gain"]/1000, marker='o', linewidth=2.5)
        ax.axvline(x=best_row['Discount'], color='red', linestyle='--', alpha=0.7, label='Optimal Discount')
        ax.set_xlabel("Discount (%)")
        ax.set_ylabel("Net Gain ($ thousands)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

    with chart_col2:
        st.subheader("Conversion Rate vs Discount")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sim_df["Discount"], sim_df["Conversion Rate"], marker='o', color='orange', linewidth=2.5)
        ax.set_xlabel("Discount (%)")
        ax.set_ylabel("Conversion Rate (%)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

                     # Footer 
st.divider()
st.caption("Telco Customer Churn Analysis Dashboard | Created with Streamlit")