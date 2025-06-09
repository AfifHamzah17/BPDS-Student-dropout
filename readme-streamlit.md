# Student Dropout Predictor

## 📖 Deskripsi

`Student Dropout Predictor` adalah prototype aplikasi berbasis Streamlit untuk memprediksi risiko dropout siswa menggunakan model machine learning yang telah dilatih (Logistic Regression). Aplikasi ini menyediakan antarmuka input data siswa, visualisasi probabilitas dropout, serta opsi tabel referensi fitur.

Aplikasi telah dideploy di Streamlit Community Cloud:

> 🌐 [Student Dropout Predictor pada Streamlit Community Cloud](https://student-dropout-afifhamzah17.streamlit.app/)

---

## ✔️ Fitur Utama

1. Input data siswa melalui sidebar:
   - Jenis Kelamin
   - Usia Saat Pendaftaran
   - Kode Course
   - Kualifikasi Sebelumnya
   - Admission Grade
   - Negara Asal
2. Preview tabel referensi fitur (course, qualification, nationality) dengan tombol `Tampilkan/Sembunyikan Tabel Informasi`.
3. Visualisasi probabilitas dropout menggunakan gauge Plotly.
4. Peringatan risiko dropout dengan kategori:
   - Risiko rendah (≤ 40%)
   - Monitoring lanjutan (40% < p ≤ 70%)
   - Intervensi segera (p > 70%)

---

## 🛠️ Persiapan Lingkungan

1. Pastikan Python 3.8+ terinstal.
2. Clone repositori atau unduh folder project.
3. Buat virtual environment (opsional).
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\\Scripts\\activate    # Windows
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## requirements.txt
streamlit==1.45.1
plotly==6.1.2
numpy==2.1.3
joblib==1.5.1
scikit-learn==1.6.1
pandas==2.3.0

## 🚀 Menjalankan Aplikasi

1. Pastikan file app.py, data.csv, dan model_dropout_predictor.joblib berada di direktori kerja.

2. Jalankan perintah:
```bash
streamlit run app.py
```
3. Browser akan terbuka otomatis menuju http://localhost:8501.

## 📂 Struktur Folder

├── app.py
├── data.csv
├── model_dropout_predictor.joblib
├── requirements.txt
├── afifhamzah17-streamlit-1.png
├── afifhamzah17-streamlit-2.png
└── afifhamzah17-streamlit-3.png

## 🎯 Kriteria Proyek

1. Minimal Satu Solusi Machine Learning:

* Pipeline preprocessing + model (LogisticRegression) telah dilatih pada dataset student performance.

* Pipeline disimpan dalam model_dropout_predictor.joblib.

2. Prototype Streamlit:

* Antarmuka interaktif untuk prediksi dropout.

* Visualisasi probabilitas menggunakan Plotly.

3. Deployment Cloud:

* Aplikasi berhasil di-deploy di Streamlit Community Cloud.

* Link deployment: [LINK](https://student-dropout-afifhamzah17.streamlit.app/)

4. Dokumentasi:

* README.md ini menjelaskan cara instalasi, penggunaan, serta link akses