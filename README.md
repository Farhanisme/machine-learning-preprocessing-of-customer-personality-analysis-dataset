# Simulasi Pra-pemrosesan pada Dataset Customer Personality Analysis

> **Nama:** Muhammad Zaky Farhan
> **NIM:** 105841110523
> **Kelas:** 5AI-A
> **Mata Kuliah:** Applied Machine Learning
> **Dosen Pengajar:** Runal Rezkiawan, S.Kom., M.T

Repositori ini berisi alur kerja (pipeline) pra-pemrosesan data lengkap untuk dataset "Customer Personality Analysis". Tujuannya adalah untuk mengubah data mentah (`.csv`) menjadi data yang bersih, terstruktur, dan teroptimasi yang siap untuk pemodelan *supervised learning* (Klasifikasi dan Regresi).

Proses ini didokumentasikan dalam dua laporan (Tugas 1 dan Tugas 2) dan dieksekusi dalam dua *notebook* Jupyter (`preprocessing1.ipynb` dan `preprocessing2.ipynb`).

**Dataset Asli:** [Customer Personality Analysis via Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data)

---

## 📁 Struktur Direktori

Proyek ini dibagi menjadi dua tahap utama, yang tercermin dalam struktur folder:

```

preprocessing-simulation-of-customer-personality-analysis-dataset/
│
├── marketing\_campaign.csv
│
├── 📂 Pre-processing I - Data Cleaning & Data Transformation/
│   ├── preprocessing1.ipynb
│   ├── Pre-processing I - Documentation Report.pdf
│   ├── data\_train\_preprocessed.csv
│   └── data\_test\_preprocessed.csv
│
└── 📂 Pre-processing II - Feature Engineering/
├── preprocessing2.ipynb
├── Pre-processing II - Documentation Report.pdf
├── X\_class\_train\_selected.csv
├── X\_class\_test\_selected.csv
├── X\_class\_train\_pca.csv
├── X\_class\_test\_pca.csv
├── X\_reg\_train\_selected.csv
├── X\_reg\_test\_selected.csv
├── X\_reg\_train\_pca.csv
├── X\_reg\_test\_pca.csv
├── y\_class\_train.csv
├── y\_class\_test.csv
├── y\_reg\_train.csv
└── y\_reg\_test.csv

```

---

## 🌊 Alur Kerja (Pipeline) Proyek

Alur kerja proyek ini bersifat sekuensial, di mana output dari Tahap I menjadi input untuk Tahap II.

### Tahap 1: Data Cleaning & Transformation (`preprocessing1.ipynb`)

*Notebook* ini mengambil `marketing_campaign.csv` mentah dan melakukan 5 langkah pembersihan dan transformasi data dasar:

1.  **Inspeksi Data:** Mengidentifikasi 24 *missing values* di `Income` dan 3 kolom `object` (Teks).
2.  **Penanganan Nilai Hilang:** Mengisi `Income` yang hilang menggunakan **Median**, karena Median bersifat *robust* (kuat) terhadap *outlier* pendapatan.
3.  **Penanganan Kategorikal:** Mengelompokkan kategori `Marital_Status` yang sangat langka ('Alone', 'Absurd', 'YOLO') ke 'Lain-Lain' dan menerapkan **One-Hot Encoding** pada `Education` dan `Marital_Status`.
4.  **Penanganan Outlier:** Menggunakan **Box Plot** untuk mendeteksi *outlier* anomali di `Year_Birth` (< 1900) dan `Income` (> 600k), kemudian menerapkan **Capping (1.5*IQR)** sebagai keputusan teknis untuk melindungi model dari *noise* statistik tanpa menghapus data.
5.  **Penskalaan & Pemisahan:** Menerapkan `train_test_split` **sebelum** *scaling* (untuk mencegah *data leakage*) dan menggunakan **`StandardScaler`**. Parameter **`stratify=y`** digunakan untuk memastikan rasio kelas `Response` yang tidak seimbang tetap terjaga di set latih dan uji.

**Output Tahap 1:**
* `data_train_preprocessed.csv` (1792 baris, data bersih & ter-scale)
* `data_test_preprocessed.csv` (448 baris, data bersih & ter-scale)

### Tahap 2: Feature Engineering & Optimization (`preprocessing2.ipynb`)

*Notebook* ini mengambil data bersih dari Tahap 1 dan menerapkan strategi optimasi fitur yang canggih.

#### Justifikasi Strategi

*Notebook* ini tidak menjalankan satu *pipeline* sederhana, melainkan:
1.  **Strategi Paralel (Seleksi vs. PCA):** Menjalankan *Feature Selection* (Skenario A) dan *Reduksi Dimensi* (Skenario B) secara paralel. Ini memungkinkan perbandingan di tahap *modelling* untuk melihat filosofi mana (membuang fitur vs. merangkum fitur) yang bekerja lebih baik.
2.  **Pipeline Ganda (Klasifikasi vs. Regresi):** Menjalankan kedua skenario untuk dua tugas terpisah, karena susunan fitur (`X`) dan target (`y`) untuk kedua tugas tersebut berbeda secara fundamental.

#### Eksekusi Tugas Klasifikasi (Target: `Response`)

* `X` berisi 31 fitur (termasuk `Income`).
* **Skenario A (Seleksi Fitur):** Menggunakan **LASSO (L1)**, yang memilih **15 fitur** paling relevan (fitur teratas: `AcceptedCmp3`, `Recency`).
* **Skenario B (PCA):** Menerapkan PCA pada 16 fitur numerik. "Scree Plot" menunjukkan **13 komponen** diperlukan untuk menangkap 95% varians.
* **Output:** `X_class_train_selected.csv`, `X_class_test_selected.csv`, `X_class_train_pca.csv`, `X_class_test_pca.csv`, `y_class_train.csv`, `y_class_test.csv`.

#### Eksekusi Tugas Regresi (Target: `Income`)

* `X` berisi 31 fitur (tidak termasuk `Income`, tapi termasuk `Response`).
* **Skenario A (Seleksi Fitur):** Menggunakan **LASSO (L1)**, yang memilih **7 fitur** paling relevan (fitur teratas: `NumWebVisitsMonth`, `MntWines`).
* **Skenario B (PCA):** Menerapkan PCA pada 15 fitur numerik. "Scree Plot" menunjukkan **13 komponen** diperlukan untuk menangkap 95% varians.
* **Output:** `X_reg_train_selected.csv`, `X_reg_test_selected.csv`, `X_reg_train_pca.csv`, `X_reg_test_pca.csv`, `y_reg_train.csv`, `y_reg_test.csv`.

---

## 🚀 Langkah Selanjutnya (Alur Kerja Modelling)

12 file `.csv` yang dihasilkan dari `preprocessing2.ipynb` siap digunakan untuk agenda *modelling* (Klasifikasi dan Regresi).

#### Untuk Agenda Klasifikasi (KNN, Naive Bayes, dll.)
* **Eksperimen A:** Latih model pada `X_class_train_selected.csv` dan `y_class_train.csv`.
* **Eksperimen B:** Latih model pada `X_class_train_pca.csv` dan `y_class_train.csv`.
* **PERHATIAN:** `y_class_train` sangat tidak seimbang (imbalanced). Teknik *resampling* seperti **SMOTE** harus diterapkan pada data latih (A dan B) sebelum `.fit()` untuk mendapatkan hasil yang valid.

#### Untuk Agenda Regresi (Regresi Linier)
* **Eksperimen A:** Latih model pada `X_reg_train_selected.csv` dan `y_reg_train.csv`.
* **Eksperimen B:** Latih model pada `X_reg_train_pca.csv` dan `y_reg_train.csv`.

Bandingkan metrik evaluasi (F1-Score, R-Squared) dari keempat eksperimen ini untuk menyimpulkan strategi pra-pemrosesan mana yang paling optimal.

---

## 🛠️ Cara Menjalankan

1.  *Clone* repositori ini.
2.  Letakkan `marketing_campaign.csv` di direktori *root* (sesuai struktur di atas).
3.  Buka dan jalankan `preprocessing1.ipynb` dari awal hingga akhir. Ini akan menghasilkan `data_train_preprocessed.csv` dan `data_test_preprocessed.csv` di dalam folder `Pre-processing I/`.
4.  Pindahkan kedua file tersebut ke direktori *root* (atau ubah path di *notebook* kedua).
5.  Buka dan jalankan `preprocessing2.ipynb` dari awal hingga akhir. Ini akan menghasilkan 12 file `.csv` yang siap untuk *modelling*.
```