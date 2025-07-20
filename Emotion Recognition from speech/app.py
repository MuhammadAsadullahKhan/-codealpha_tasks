# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("📊 Credit Scoring Model")
st.markdown("Predict an individual's creditworthiness using financial data.")

# File Upload
uploaded_file = st.file_uploader("📁 Upload your credit dataset (.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.write(df.head())

    # Check for required columns
    st.subheader("✅ Adding Dummy Target Column (if missing)")
    if 'defaulted' not in df.columns:
        st.warning("⚠️ 'defaulted' column not found. Adding dummy target for testing...")
        df['defaulted'] = np.random.randint(0, 2, df.shape[0])
    else:
        st.success("✅ 'defaulted' column found.")

    # Preprocessing
    st.subheader("🧹 Data Preprocessing")
    df = df.dropna()  # Remove missing values
    df = pd.get_dummies(df, drop_first=True)  # One-hot encoding
    st.write("✅ Data after encoding:")
    st.write(df.head())

    # Features and target
    X = df.drop("defaulted", axis=1)
    y = df["defaulted"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model selection
    model_option = st.selectbox("🤖 Choose Model", ("Logistic Regression", "Decision Tree", "Random Forest"))

    if st.button("Train Model"):
        if model_option == "Logistic Regression":
            model = LogisticRegression()
        elif model_option == "Decision Tree":
            model = DecisionTreeClassifier()
        else:
            model = RandomForestClassifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        st.subheader("📊 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("🔥 ROC-AUC Score")
        roc_auc = roc_auc_score(y_test, y_proba)
        st.success(f"ROC-AUC Score: {roc_auc:.2f}")

        # Plot ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], linestyle='--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)
