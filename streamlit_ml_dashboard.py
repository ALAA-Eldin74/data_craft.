import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Smart ML Platform",
    page_icon="🤖",
    layout="wide"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
.metric-card {
    background-color: #0e1117;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
st.sidebar.title("🤖 Smart ML Platform")
st.sidebar.markdown("Upload → Visualize → Train → Evaluate")

uploaded_file = st.sidebar.file_uploader("📁 Upload Dataset", type=["csv", "xlsx"])

# ---------------- Main ----------------
st.title("📊 Interactive Data Science Dashboard")
st.caption("No code • Smart visualization • Automatic model selection")

if uploaded_file is None:
    st.info("⬅️ Upload a dataset from the sidebar to get started")
    st.stop()

# ---------------- Load Data ----------------
if uploaded_file.name.endswith("csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

st.subheader("🔍 Data Preview")
st.dataframe(df.head(), use_container_width=True)

# ---------------- Column Types ----------------
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# ---------------- Visualization ----------------
st.subheader("📈 Interactive Visualization")
col1, col2, col3 = st.columns(3)
with col1:
    x_col = st.selectbox("X Axis", df.columns)
with col2:
    y_col = st.selectbox("Y Axis", num_cols)
with col3:
    chart_type = st.selectbox("Chart Type", ["Auto", "Scatter", "Bar", "Line", "Box", "Histogram"])

if chart_type == "Histogram":
    fig = px.histogram(df, x=x_col)
elif chart_type == "Bar":
    fig = px.bar(df, x=x_col, y=y_col)
elif chart_type == "Line":
    fig = px.line(df, x=x_col, y=y_col)
elif chart_type == "Box":
    fig = px.box(df, x=x_col, y=y_col)
else:
    fig = px.scatter(df, x=x_col, y=y_col)

st.plotly_chart(fig, use_container_width=True)

# ---------------- Machine Learning ----------------
st.subheader("🧠 Machine Learning")
with st.expander("⚙️ Model Settings", expanded=True):
    target = st.selectbox("🎯 Select Target Column", df.columns)
    features = st.multiselect("📌 Select Feature Columns", [col for col in df.columns if col != target], default=[col for col in df.columns if col != target])
    model_choice = st.selectbox("🤖 Model", ["Logistic Regression", "Random Forest"])

X = df[features]
y = df[target]

# Handle binary target encoding automatically
if y.nunique() == 2 and y.dtype != np.number:
    le = LabelEncoder()
    y = le.fit_transform(y)
    st.info(f"Binary target detected. Values converted to 0/1.")

# Fill missing values
X = X.copy()
for col in X.columns:
    if X[col].dtype == 'number' or np.issubdtype(X[col].dtype, np.number):
        X[col].fillna(X[col].mean(), inplace=True)
    else:
        X[col].fillna(X[col].mode()[0], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y if len(np.unique(y))>1 else None
)

# ColumnTransformer to handle numeric + categorical
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

if model_choice == "Logistic Regression":
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
else:
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
    ])

train_btn = st.button("🚀 Train Model")

if train_btn:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    st.subheader("🏆 Model Performance")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Accuracy", f"{acc:.2%}")
    with m2:
        st.metric("Train Size", X_train.shape[0])
    with m3:
        st.metric("Test Size", X_test.shape[0])

    st.markdown("### 📊 Confusion Matrix")
    fig_cm = px.imshow(cm, text_auto=True)
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("### 📄 Classification Report")
    st.text(classification_report(y_test, preds))

    st.success("✅ Training completed successfully")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Smart ML Platform")
