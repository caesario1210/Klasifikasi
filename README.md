# 🎯 Klasifikasi Prediksi Langganan Nasabah Bank

Repositori ini merupakan proyek data science yang bertujuan untuk memprediksi apakah seorang nasabah akan berlangganan produk deposito berjangka (term deposit) dari sebuah bank. Model klasifikasi yang digunakan berbasis *Decision Tree*, dengan aplikasi berbasis **Streamlit** untuk visualisasi prediksi secara interaktif.

---

## 🧠 Dataset
Dataset yang digunakan adalah `bank_marketing.csv`, berisi data pemasaran dari sebuah bank.

**Fitur dalam dataset:**
- `age`: Umur nasabah
- `job`: Jenis pekerjaan
- `marital`: Status pernikahan
- `education`: Tingkat pendidikan
- `default`: Status kredit sebelumnya
- `balance`: Saldo akun
- `housing`: Pinjaman perumahan
- `loan`: Pinjaman pribadi
- `contact`: Media kontak
- `month`: Bulan terakhir kontak
- `duration`: Durasi panggilan terakhir (detik)
- `campaign`: Jumlah kontak dalam kampanye ini
- `pdays`: Hari sejak kontak terakhir
- `previous`: Jumlah kontak sebelumnya
- `poutcome`: Hasil kampanye pemasaran sebelumnya
- `y`: Target variabel (langganan atau tidak)

---

## 🚀 Teknologi dan Tools
- Python 3
- Streamlit
- Scikit-learn
- Pandas, NumPy
- Joblib
- Matplotlib / Seaborn

---

## 🧾 Struktur File
```bash
.
├── streamlit_subscribe_app.py         # Aplikasi utama Streamlit
├── bank_marketing.csv                 # Dataset utama
├── y_prediction_components.joblib     # Model tanpa Hyperparameter
├── _034411_dt_tuned_comps.joblib      # Model dengan Hyperparameter
├── feature_importance.png             # Visualisasi fitur penting
├── histogramafter_coding.png          # Histogram fitur setelah preprocessing
├── requirements.txt                   # Dependensi Python
├── README.md                          # Dokumentasi proyek ini
└── myenv/                             # (Opsional) Lingkungan virtual
