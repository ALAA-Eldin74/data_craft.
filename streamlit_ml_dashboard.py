import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, recall_score, precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ================= FAIL-SAFE SMOTE =================
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except Exception:
    SMOTE_AVAILABLE = False

# ---------------- Page Config ----------------
st.set_page_config(page_title="Pro ML Platform", page_icon="🤖", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.title("🤖 Pro ML Platform")
st.sidebar.markdown("Upload → Visualize → Model → Evaluate")

uploaded_file = st.sidebar.file_uploader("📁 Upload Dataset", type=["csv", "xlsx"])

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

# ================= Visualization =================
st.subheader("📈 Interactive Visualization")
num_cols = df.select_dtypes(include=np.number).columns.tolist()

c1, c2, c3, c4 = st.columns(4)
with c1:
    x_col = st.selectbox("X Axis", df.columns)
with c2:
    y_col = st.selectbox("Y Axis", num_cols)
with c3:
    chart_type = st.selectbox(
        "Chart Type",
        ["Scatter", "Bar", "Line", "Box", "Histogram", "Pie", "Heatmap"]
    )
with c4:
    color_col = st.selectbox("Color", [None] + df.columns.tolist())

if chart_type == "Histogram":
    fig = px.histogram(df, x=x_col, color=color_col)
elif chart_type == "Bar":
    fig = px.bar(df, x=x_col, y=y_col, color=color_col)
elif chart_type == "Line":
    fig = px.line(df, x=x_col, y=y_col, color=color_col)
elif chart_type == "Box":
    fig = px.box(df, x=x_col, y=y_col, color=color_col)
elif chart_type == "Pie":
    fig = px.pie(df, names=x_col)
elif chart_type == "Heatmap":
    fig = px.imshow(df[num_cols].corr(), text_auto=True)
else:
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col)

st.plotly_chart(fig, use_container_width=True)

# ================= Machine Learning =================
st.subheader("🧠 Machine Learning")

target = st.selectbox("🎯 Target Column", df.columns)
features = st.multiselect(
    "📌 Feature Columns",
    [c for c in df.columns if c != target],
    default=[c for c in df.columns if c != target]
)

X = df[features].copy()
y = df[target]

# Encode binary target
if y.nunique() == 2 and y.dtype != np.number:
    y = LabelEncoder().fit_transform(y)

# Fill missing values
for col in X.columns:
    if np.issubdtype(X[col].dtype, np.number):
        X[col].fillna(X[col].mean(), inplace=True)
    else:
        X[col].fillna(X[col].mode()[0], inplace=True)

# Detect imbalance
class_ratio = pd.Series(y).value_counts(normalize=True)
imbalance = class_ratio.min() < 0.3

if imbalance:
    st.warning("⚠️ Imbalanced dataset detected")

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Preprocessing
num_features = X.select_dtypes(include=np.number).columns.tolist()
cat_features = X.select_dtypes(exclude=np.number).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# Optional SMOTE
if imbalance and SMOTE_AVAILABLE:
    use_smote = st.checkbox("🧪 Use SMOTE (if available)", value=True)
elif imbalance:
    st.info("SMOTE not available → using class_weight only")
    use_smote = False
else:
    use_smote = False

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced" if imbalance else None),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced" if imbalance else None),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVM": SVC(probability=True, class_weight="balanced" if imbalance else None),
    "KNN": KNeighborsClassifier()
}

metric = "f1" if imbalance else "accuracy"

if st.button("🚀 Train & Recommend Model"):
    best_score = 0
    best_model = None
    best_name = ""

    for name, clf in models.items():
        if use_smote and imbalance and SMOTE_AVAILABLE:
            pipe = ImbPipeline([
                ("prep", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("model", clf)
            ])
        else:
            pipe = Pipeline([
                ("prep", preprocessor),
                ("model", clf)
            ])

        scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring=metric)
        mean_score = scores.mean()
        st.write(f"{name} ({metric.upper()}): {mean_score:.2%}")

        if mean_score > best_score:
            best_score = mean_score
            best_model = pipe
            best_name = name

    st.success(f"🏆 Best Model: {best_name} | CV {metric.upper()}: {best_score:.2%}")

    best_model.fit(X_train, y_train)

    # Smart threshold
    if imbalance and hasattr(best_model.named_steps["model"], "predict_proba"):
        probs = best_model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.3).astype(int)
    else:
        preds = best_model.predict(X_test)

    st.subheader("📊 Test Performance")
    st.metric("Accuracy", f"{accuracy_score(y_test, preds):.2%}")
    st.metric("F1 Score", f"{f1_score(y_test, preds):.2%}")
    st.metric("Recall (Class 1)", f"{recall_score(y_test, preds):.2%}")
    st.metric("Precision", f"{precision_score(y_test, preds):.2%}")

    st.markdown("### 🔢 Confusion Matrix")
    st.plotly_chart(px.imshow(confusion_matrix(y_test, preds), text_auto=True))

    st.markdown("### 📄 Classification Report")
    st.text(classification_report(y_test, preds))

st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Professional ML Platform")
