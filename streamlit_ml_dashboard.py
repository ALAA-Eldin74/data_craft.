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
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------- Page Config ----------------
st.set_page_config(page_title="Pro ML Platform", page_icon="🤖", layout="wide")

# ---------------- Session State ----------------
for step in ["preview", "desc", "viz", "ml"]:
    if f"show_{step}" not in st.session_state:
        st.session_state[f"show_{step}"] = False

# ---------------- Sidebar ----------------
st.sidebar.title("🤖 Pro ML Platform")
st.sidebar.markdown("Upload → Describe → Visualize → Model → Evaluate")

uploaded_file = st.sidebar.file_uploader("📁 Upload Dataset", type=["csv", "xlsx"])

if uploaded_file is None:
    st.info("⬅️ Upload a dataset from the sidebar to get started")
    st.stop()

# ---------------- Load Data ----------------
if uploaded_file.name.endswith("csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

# ================= Step 1: Preview =================
if st.button("🔍 Step 1: Show Data Preview"):
    st.session_state.show_preview = True

if st.session_state.show_preview:
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    if st.button("📊 Step 2: Go to Dataset Description"):
        st.session_state.show_desc = True

# ================= Step 2: Description =================
if st.session_state.show_desc:
    st.subheader("📊 Dataset Overview & Statistics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    st.markdown("### 🧾 Column Information")
    desc_df = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str),
        "Missing": df.isnull().sum().values,
        "Unique": df.nunique().values
    })
    st.dataframe(desc_df, use_container_width=True)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        st.markdown("### 📐 Numerical Statistics")
        stats_df = pd.DataFrame({
            "Column": num_cols,
            "Mean": [df[c].mean() for c in num_cols],
            "Median": [df[c].median() for c in num_cols],
            "Std": [df[c].std() for c in num_cols],
            "Min": [df[c].min() for c in num_cols],
            "Max": [df[c].max() for c in num_cols],
        })
        st.dataframe(stats_df.round(2), use_container_width=True)

    if st.button("📈 Step 3: Go to Visualization"):
        st.session_state.show_viz = True

# ================= Step 3: Visualization =================
if st.session_state.show_viz:
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

    if st.button("🧠 Step 4: Go to Machine Learning"):
        st.session_state.show_ml = True

# ================= Step 4: Machine Learning =================
if st.session_state.show_ml:
    st.subheader("🧠 Machine Learning")

    target = st.selectbox("🎯 Target Column", df.columns)
    features = st.multiselect(
        "📌 Feature Columns",
        [c for c in df.columns if c != target],
        default=[c for c in df.columns if c != target]
    )

    X = df[features].copy()
    y = df[target]

    if y.nunique() == 2 and y.dtype != np.number:
        y = LabelEncoder().fit_transform(y)

    for col in X.columns:
        if np.issubdtype(X[col].dtype, np.number):
            X[col].fillna(X[col].mean(), inplace=True)
        else:
            X[col].fillna(X[col].mode()[0], inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

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

    if st.button("🚀 Step 5: Train & Recommend Model"):
        best_acc = 0
        best_model = None
        best_name = ""

        for name, clf in models.items():
            pipe = Pipeline([
                ("prep", preprocessor),
                ("model", clf)
            ])
            scores = cross_val_score(pipe, X_train, y_train, cv=3)
            mean_score = scores.mean()
            st.write(f"{name}: {mean_score:.2%}")

            if mean_score > best_acc:
                best_acc = mean_score
                best_model = pipe
                best_name = name

        st.success(f"Best Model: {best_name}")

        best_model.fit(X_train, y_train)
        preds = best_model.predict(X_test)

        st.metric("Test Accuracy", f"{accuracy_score(y_test, preds):.2%}")

        st.markdown("### 📊 Confusion Matrix")
        st.plotly_chart(px.imshow(confusion_matrix(y_test, preds), text_auto=True))

        st.markdown("### 📄 Classification Report")
        st.text(classification_report(y_test, preds))

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Professional ML Platform")
