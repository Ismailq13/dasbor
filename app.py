import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Regresi Linier", layout="centered")
st.title("Regresi Linier dari CSV")

uploaded_file = st.file_uploader("Unggah file CSV (satu kolom berisi x,y)", type=["csv"])

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    st.subheader("Data Mentah")
    st.write(raw_df.head())

    try:
        # Pisahkan kolom
        data_split = raw_df.iloc[:, 0].str.split(",", expand=True)
        data_split.columns = ['x', 'y']
        data_split['x'] = pd.to_numeric(data_split['x'], errors='coerce')
        data_split['y'] = pd.to_numeric(data_split['y'], errors='coerce')
        df = data_split.dropna()

        st.subheader("Data Split")
        st.write(df)

        # Model
        X = df[['x']]
        y = df['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluasi
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Evaluasi Model")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**R² Score:** {r2:.4f}")

        # Penjelasan R²
        st.subheader("Interpretasi R²")
        if r2 >= 0.95:
            st.success("Model sangat baik — hampir semua variasi pada y bisa dijelaskan oleh x (R² mendekati 1).")
        elif r2 >= 0.75:
            st.info("Model cukup baik — sebagian besar variasi pada y dapat dijelaskan oleh x.")
        elif r2 >= 0.5:
            st.warning("Model sedang — mungkin ada variabel lain yang memengaruhi y.")
        else:
            st.error("Model lemah — hubungan antara x dan y tidak cukup kuat untuk prediksi yang akurat.")

        # Grafik Regresi
        st.subheader("Grafik Regresi Linier")
        fig, ax = plt.subplots()
        sns.regplot(x='x', y='y', data=df, ax=ax, line_kws={"color": "red"})
        ax.set_title("Hubungan x dan y")
        st.pyplot(fig)

        # Prediksi nilai baru
        st.subheader("Prediksi Nilai Baru")
        input_x = st.number_input("Masukkan nilai x", value=0.0)
        if st.button("Prediksi y"):
            pred_y = model.predict([[input_x]])[0]
            st.success(f"Prediksi y: {pred_y:.2f}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
