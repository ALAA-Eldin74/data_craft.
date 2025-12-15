import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------- Page Config ----------------
st.set_page_config(page_title="Pro ML Platform", page_icon="🤖", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.title("🤖 Pro ML Platform")
st.sidebar.markdown("Upload → Clean → Features → Model → Evaluate")

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

# ---------------- Column Types ----------------
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
text_cols = cat_cols.copy()

# ---------------- Visualization ----------------
st.subheader("📈 Interactive Visualization")
col1, col2, col3, col4 = st.columns(4)
with col1:
    x_col = st.selectbox("X Axis", df.columns)
with col2:
    y_col = st.selectbox("Y Axis", num_cols)
with col3:
    chart_type = st.selectbox("Chart Type", ["Scatter", "Bar", "Line", "Box", "Histogram", "Pie", "Heatmap"])
with col4:
    color_col = st.selectbox("Color (optional)", [None]+df.columns.tolist())

if chart_type == "Histogram":
    fig = px.histogram(df, x=x_col, color=color_col)
elif chart_type == "Bar":
    fig = px.bar(df, x=x_col, y=y_col, color=color_col)
elif chart_type == "Line":
    fig = px.line(df, x=x_col, y=y_col, color=color_col)
elif chart_type == "Box":
    fig = px.box(df, x=x_col, y=y_col, color=color_col)
elif chart_type == "Pie":
    fig = px.pie(df, names=x_col, values=y_col if y_col else None, color=color_col)
elif chart_type == "Heatmap":
    corr = df[num_cols].corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis')
else:
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col)

st.plotly_chart(fig, use_container_width=True)

# ---------------- Machine Learning ----------------
st.subheader("🧠 Machine Learning")
with st.expander("⚙️ Model Settings", expanded=True):
    target = st.selectbox("🎯 Select Target Column", df.columns)
    features = st.multiselect("📌 Select Feature Columns", [col for col in df.columns if col != target], default=[col for col in df.columns if col != target])

X = df[features]
y = df[target]

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
        X[col].fillna(X[col].astype(str).mode()[0], inplace=True)

# Train/Test split with fallback
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
except ValueError:
    st.warning("Stratify failed. Splitting without stratify.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Text columns safe handling
text_cols_existing = [col for col in text_cols if col in X_train.columns]
if len(text_cols_existing) > 0:
    for col in text_cols_existing:
        X_train[col] = X_train[col].astype(str).fillna('')
        X_test[col] = X_test[col].astype(str).fillna('')
    X_train['__text_combined__'] = X_train[text_cols_existing].agg(' '.join, axis=1)
    X_test['__text_combined__'] = X_test[text_cols_existing].agg(' '.join, axis=1)
    text_feature = '__text_combined__'
else:
    text_feature = None

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = [col for col in X.select_dtypes(exclude=np.number).columns.tolist() if col not in text_cols_existing]

transformers = [('num', StandardScaler(), numeric_features)]
if categorical_features:
    transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))
if text_feature:
    transformers.append(('text', TfidfVectorizer(), text_feature))

preprocessor = ColumnTransformer(transformers=transformers)

# Multi-model dictionary for Auto Recommendation
models_dict = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier()
}

train_btn = st.button("🚀 Train & Recommend Model")

if train_btn:
    best_acc = 0
    best_model_name = None
    best_model_pipeline = None
    for name, clf in models_dict.items():
        model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
        try:
            cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=3, scoring='accuracy')
            mean_score = np.mean(cv_scores)
            st.write(f"{name} CV Accuracy: {mean_score:.2%}")
            if mean_score > best_acc:
                best_acc = mean_score
                best_model_name = name
                best_model_pipeline = model_pipeline
        except Exception as e:
            st.write(f"{name} failed: {e}")

    st.success(f"Best Model: {best_model_name} with CV Accuracy: {best_acc:.2%}")

    # Fit best model
    best_model_pipeline.fit(X_train, y_train)
    preds = best_model_pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    st.subheader("🏆 Test Performance")
    st.metric("Test Accuracy", f"{acc:.2%}")

    st.markdown("### 📊 Confusion Matrix")
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Viridis')
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("### 📄 Classification Report")
    st.text(classification_report(y_test, preds))

    st.subheader("📝 Sample Predictions")
    sample_df = X_test.copy()
    sample_df['Actual'] = y_test
    sample_df['Predicted'] = preds
    st.dataframe(sample_df.head(10), use_container_width=True)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Professional ML Platform")
