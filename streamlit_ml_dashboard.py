import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ---------------- Page Config ----------------
st.set_page_config(page_title="Pro ML Platform", page_icon="🤖", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.title("🤖 Pro ML Platform")
st.sidebar.markdown("Upload Dataset → Explore → Visualize → Train & Recommend")

uploaded_file = st.sidebar.file_uploader("📁 Upload Dataset", type=["csv", "xlsx"])

if uploaded_file is None:
    st.info("⬅️ Upload a dataset from the sidebar to get started")
    st.stop()

# ---------------- Load Data ----------------
if uploaded_file.name.endswith("csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

# ---------------- Tabs ----------------
tab1, tab2 = st.tabs(["📊 Data Overview", "🧠 Visualization & ML"])

# ================= TAB 1: Data Overview =================
with tab1:
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("🧾 Column Information")
    desc_df = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })
    st.dataframe(desc_df, use_container_width=True)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        st.subheader("📐 Numerical Statistics")
        stats_df = df[num_cols].describe().T
        st.dataframe(stats_df.round(2), use_container_width=True)

# ================= TAB 2: Visualization & ML =================
with tab2:
    st.subheader("📈 Interactive Visualization")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        x_col = st.selectbox("X Axis", df.columns)
    with c2:
        y_col = st.selectbox("Y Axis", num_cols if num_cols else df.columns)
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

    # ---------------- Machine Learning ----------------
    st.markdown("---")
    st.subheader("🧠 Machine Learning & Model Recommendation")

    target = st.selectbox("🎯 Target Column", df.columns)
    features = st.multiselect(
        "📌 Feature Columns",
        [c for c in df.columns if c != target],
        default=[c for c in df.columns if c != target]
    )

    X = df[features].copy()
    y = df[target]

    # Encode target if needed
    if y.nunique() == 2 and y.dtype == object:
        y = LabelEncoder().fit_transform(y)

    # Handle missing values
    for col in X.columns:
        if np.issubdtype(X[col].dtype, np.number):
            X[col].fillna(X[col].mean(), inplace=True)
        else:
            X[col].fillna(X[col].mode()[0], inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    num_features = X.select_dtypes(include=np.number).columns.tolist()
    cat_features = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200),
        "Gradient Boosting": GradientBoostingClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier()
    }

    metric = st.selectbox(
        "📏 Metric to Recommend Best Model",
        ["accuracy", "f1", "recall", "precision"]
    )

    if st.button("🚀 Train & Recommend Best Model"):
        best_score = 0
        best_model = None
        best_name = ""

        st.markdown("### 🔍 Cross Validation Results")
        for name, clf in models.items():
            pipe = Pipeline([("prep", preprocessor), ("model", clf)])
            scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring=metric)
            mean_score = scores.mean()
            st.write(f"**{name}** → {mean_score:.2%}")

            if mean_score > best_score:
                best_score = mean_score
                best_model = pipe
                best_name = name

        st.success(f"🧠 Recommended Model based on {metric.upper()}: **{best_name}**")

        best_model.fit(X_train, y_train)
        preds = best_model.predict(X_test)

        st.metric("✅ Test Accuracy", f"{accuracy_score(y_test, preds):.2%}")

        st.markdown("### 📊 Confusion Matrix")
        st.plotly_chart(px.imshow(confusion_matrix(y_test, preds), text_auto=True))

        st.markdown("### 📄 Classification Report")
        st.text(classification_report(y_test, preds))

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Professional ML Platform")# 
