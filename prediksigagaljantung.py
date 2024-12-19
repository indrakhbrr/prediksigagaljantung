import time
import os
import pandas as pd
import joblib
import json
import logging
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Konfigurasi Logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Simpan history training
TRAIN_HISTORY_FILE = "train_history.json"

# Fungsi untuk memuat riwayat training
def load_train_history():
    if os.path.exists(TRAIN_HISTORY_FILE):
        with open(TRAIN_HISTORY_FILE, "r") as file:
            return json.load(file)
    return []

# Fungsi untuk menyimpan riwayat pelatihan
def save_train_history(history):
    with open(TRAIN_HISTORY_FILE, "w") as file:
        json.dump(history, file, indent=4)

# Fungsi load uploaded file
def load_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

# Streamlit app
st.title("📊 Sistem Prediksi Keberlangsungan Hidup Pasien Gagal Jantung")
st.sidebar.title("📋 Menu")

# Sidebar Menu
menu = st.sidebar.selectbox("Pilih Menu", ["Beranda", "Train Model", "Test Model", "Lihat Log"])

# Menu: Beranda
if menu == "Beranda":
    st.subheader("Selamat Datang di Sistem Prediksi")
    st.write("""
    Pilih menu di sidebar untuk:
    - **Train Model**: Melatih model K-NN dan Naive Bayes.
    - **Test Model**: Memanfaatkan model untuk prediksi data baru.
    - **Lihat Log**: Menampilkan log aplikasi.
    """)

# Menu: Train Model
elif menu == "Train Model":
    st.subheader("🔍 Train Models (K-NN & Naive Bayes)")

    # Layout dengan kolom
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Langkah 1**: Upload dataset Anda dalam format CSV/Excel.")
        uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])
    with col2:
        st.info("**Langkah 2**: Klik **Train Models** untuk memulai pelatihan.")

    # Display training history
    st.write("### 📜 Riwayat Pelatihan")
    history = load_train_history()
    if history:
        st.write(pd.DataFrame(history))
    else:
        st.write("Belum ada riwayat pelatihan.")

    if uploaded_file:
        try:
            data = load_file(uploaded_file)
            logging.info(f"Dataset '{uploaded_file.name}' berhasil diunggah.")
            st.write("### Dataset Preview")
            st.write(data.head())

            if "DEATH_EVENT" not in data.columns:
                st.error("Dataset harus memiliki kolom 'DEATH_EVENT'")
                logging.error("Kolom 'DEATH_EVENT' tidak ditemukan dalam dataset.")
            else:
                if st.button("Train Models"):
                    start_time = time.time()

                    # Pembagian data
                    X = data.drop(columns=["DEATH_EVENT"])
                    y = data["DEATH_EVENT"]
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )

                    # Scaling data
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Train K-NN
                    knn = KNeighborsClassifier(n_neighbors=5)
                    knn.fit(X_train_scaled, y_train)

                    # Train Naive Bayes
                    nb = GaussianNB()
                    nb.fit(X_train, y_train)

                    # Save model and scaler
                    joblib.dump(knn, "knn_model.pkl")
                    joblib.dump(nb, "nb_model.pkl")
                    joblib.dump(scaler, "scaler.pkl")

                    end_time = time.time()
                    duration = end_time - start_time
                    logging.info("Model berhasil dilatih.")

                    st.success("Model berhasil dilatih!")

                    # Visualisasi distribusi kelas
                    st.write("### Target Distribusi Kelas")
                    class_distribution = y.value_counts().rename({0: 'No Death', 1: 'Death'})
                    fig, ax = plt.subplots()
                    class_distribution.plot(kind='bar', ax=ax, color=['blue', 'orange'])
                    ax.set_title("Distribusi Kelas")
                    ax.set_xlabel("Kelas")
                    ax.set_ylabel("Jumlah")
                    st.pyplot(fig)

                    # Akurasi Training
                    knn_train_acc = accuracy_score(y_train, knn.predict(X_train_scaled))
                    nb_train_acc = accuracy_score(y_train, nb.predict(X_train))
                    st.write("### Akurasi Training")
                    st.write(f"K-NN: {knn_train_acc:.2f}")
                    st.write(f"Naive Bayes: {nb_train_acc:.2f}")

                    # Perbandingan Akurasi
                    st.write("### Perbandingan Akurasi Training")
                    fig, ax = plt.subplots()
                    model_names = ['K-NN', 'Naive Bayes']
                    accuracies = [knn_train_acc, nb_train_acc]
                    ax.bar(model_names, accuracies, color=['blue', 'orange'])
                    ax.set_ylim(0, 1)
                    ax.set_ylabel("Akurasi")
                    ax.set_title("Perbandingan Akurasi Model")
                    st.pyplot(fig)

                    # Save history
                    new_entry = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "dataset_name": uploaded_file.name,
                        "knn_train_accuracy": knn_train_acc,
                        "nb_train_accuracy": nb_train_acc,
                        "duration_seconds": duration,
                    }
                    history.append(new_entry)
                    save_train_history(history)

        except Exception as e:
            st.error("Terjadi kesalahan saat memproses dataset.")
            logging.error(f"Error: {e}")

# Menu: Test Model
elif menu == "Test Model":
    st.subheader("Test Models (K-NN & Naive Bayes)")
    uploaded_file = st.file_uploader("Upload Tes Dataset (CSV / Excel)", type=["csv", "xlsx"])

    if uploaded_file:
        data = load_file(uploaded_file)
        st.write("### Uploaded Dataset Preview")
        st.write(data.head())

        # Load model and scaler
        try:
            knn = joblib.load("knn_model.pkl")
            nb = joblib.load("nb_model.pkl")
            scaler = joblib.load("scaler.pkl")

            if "DEATH_EVENT" in data.columns:
                X = data.drop(columns=["DEATH_EVENT"])
                y = data["DEATH_EVENT"]
                X_scaled = scaler.transform(X)

                # Prediksi
                knn_predictions = knn.predict(X_scaled)
                nb_predictions = nb.predict(X)

                # Akurasi
                knn_accuracy = accuracy_score(y, knn_predictions)
                nb_accuracy = accuracy_score(y, nb_predictions)

                st.write("### Akurasi Model")
                st.write(f"Akurasi KNN: {knn_accuracy:.2f}")
                st.write(f"Akurasi Naive Bayes: {nb_accuracy:.2f}")

                # Confusion Matrix & Classification Reports
                def plot_confusion_matrix(y_true, y_pred, model_name):
                    cm = confusion_matrix(y_true, y_pred)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Death', 'Death'], yticklabels=['No Death', 'Death'])
                    ax.set_title(f'Confusion Matrix - {model_name}')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    return fig

                st.write("### Confusion Matrix - K-NN")
                st.pyplot(plot_confusion_matrix(y, knn_predictions, "K-NN"))

                st.write("### Confusion Matrix - Naive Bayes")
                st.pyplot(plot_confusion_matrix(y, nb_predictions, "Naive Bayes"))

                st.write("### Classification Reports")
                st.write("#### K-NN:")
                st.text(classification_report(y, knn_predictions))
                st.write("#### Naive Bayes:")
                st.text(classification_report(y, nb_predictions))

            else:
                # Hanya tampil prediksi saja jika tidak ada kolom DEATH_EVENT
                X = data
                X_scaled = scaler.transform(X)

                # Prediksi
                knn_predictions = knn.predict(X_scaled)
                nb_predictions = nb.predict(X)
                data["K-NN Prediction"] = knn_predictions
                data["Naive Bayes Prediction"] = nb_predictions
                st.write("### Predictions")
                st.write(data)

                # Download hasil prediksi
                output_file = "predictions.csv"
                data.to_csv(output_file, index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=open(output_file, "rb"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )

        except FileNotFoundError:
            st.error("Models tidak ditemukan! Silahkan latih model dulu di menu 'Train Model'")

# Menu: Lihat Log
elif menu == "Lihat Log":
    st.subheader("📄 Log Aplikasi")
    with open("app.log", "r") as log_file:
        st.text(log_file.read())