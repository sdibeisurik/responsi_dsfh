import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

st.title("Pelatihan Model Logistic Regression")

# Upload dataset
uploaded_file = st.file_uploader(
    "Upload Dataset CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset")
    st.dataframe(df.head())

    # Memilih kolom target
    target = st.selectbox(
        "Pilih Kolom Target",
        df.columns
    )

    if st.button("Latih Model"):

        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        model = LogisticRegression(max_iter=1000)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        st.success(f"Akurasi Model: {acc:.4f}")

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )

        st.dataframe(pd.DataFrame(report).transpose())

        # Simpan model
        joblib.dump(
            model,
            "Logistic_Regression.pkl"
        )

        st.success(
            "Modelnya berhasil disimpan sebagai Logistic_Regression.pkl"
        )