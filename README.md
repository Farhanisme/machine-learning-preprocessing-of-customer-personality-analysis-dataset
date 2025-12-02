# Simulasi Pra-pemrosesan pada Dataset Customer Personality Analysis

> **Nama:** Muhammad Zaky Farhan
> 
> **NIM:** 105841110523
> 
> **Kelas:** 5AI-A
> 
> **Mata Kuliah:** Applied Machine Learning
> 
> **Dosen Pengajar:** Runal Rezkiawan, S.Kom., M.T

Repositori ini berisi alur kerja (pipeline) pra-pemrosesan hingga pemodelan data lengkap untuk dataset "Customer Personality Analysis". Tujuannya adalah untuk mengubah data mentah (`.csv`) menjadi data yang bersih, terstruktur, dan teroptimasi, kemudian mengujinya menggunakan algoritma *supervised learning* (Klasifikasi dan Regresi).

Proses ini didokumentasikan secara bertahap dan dieksekusi dalam beberapa *notebook* Jupyter terpisah.

**Dataset Asli:** [Customer Personality Analysis via Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data)

---

## 📁 Struktur Direktori

Proyek ini dibagi menjadi tiga tahap utama (Data Cleaning, Feature Engineering, dan Modelling Klasifikasi), yang tercermin dalam struktur folder:

```text
📂preprocessing-simulation-of-customer-personality-analysis-dataset/
│
├── marketing_campaign.csv
│
├── 📂 Pre-processing I - Data Cleaning & Data Transformation/
│   ├── preprocessing1.ipynb
│   ├── Pre-processing I - Documentation Report.pdf
│   ├── data_train_preprocessed.csv
│   └── data_test_preprocessed.csv
│
├── 📂 Pre-processing II - Feature Engineering/
│   ├── preprocessing2.ipynb
│   ├── Pre-processing II - Documentation Report.pdf
│   ├── X_class_train_selected.csv
│   ├── X_class_test_selected.csv
│   ├── X_class_train_pca.csv
│   ├── X_class_test_pca.csv
│   ├── X_reg_train_selected.csv
│   ├── X_reg_test_selected.csv
│   ├── X_reg_train_pca.csv
│   ├── X_reg_test_pca.csv
│   ├── y_class_train.csv
│   ├── y_class_test.csv
│   ├── y_reg_train.csv
│   └── y_reg_test.csv
│
└── 📂 Classification - KNN & NB/
    ├── classification.ipynb
    ├── KNN & NB Classification - Documentation Report.pdf
    ├── X_class_train_selected.csv
    ├── X_class_test_selected.csv
    ├── X_class_train_pca.csv
    ├── X_class_test_pca.csv
    ├── y_class_train.csv
    └── y_class_test.csv
````

-----

## 🌊 Alur Kerja (Pipeline) Proyek

Alur kerja proyek ini bersifat sekuensial, di mana output dari tahap sebelumnya menjadi input untuk tahap berikutnya.

### Tahap 1: Data Cleaning & Transformation (`preprocessing1.ipynb`)

*Notebook* ini mengambil `marketing_campaign.csv` mentah dan melakukan 5 langkah pembersihan dan transformasi data dasar:

1.  **Inspeksi Data:** Mengidentifikasi 24 *missing values* di `Income` dan 3 kolom `object` (Teks).
2.  **Penanganan Nilai Hilang:** Mengisi `Income` yang hilang menggunakan **Median**, karena Median bersifat *robust* (kuat) terhadap *outlier* pendapatan.
3.  **Penanganan Kategorikal:** Mengelompokkan kategori `Marital_Status` yang sangat langka ('Alone', 'Absurd', 'YOLO') ke 'Lain-Lain' dan menerapkan **One-Hot Encoding** pada `Education` dan `Marital_Status`.
4.  **Penanganan Outlier:** Menggunakan **Box Plot** untuk mendeteksi *outlier* anomali di `Year_Birth` (\< 1900) dan `Income` (\> 600k), kemudian menerapkan **Capping (1.5\*IQR)** sebagai keputusan teknis untuk melindungi model dari *noise* statistik tanpa menghapus data.
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
  * **Output:** Dataset `X_class...` dan `y_class...` (total 6 file).

#### Eksekusi Tugas Regresi (Target: `Income`)

  * `X` berisi 31 fitur (tidak termasuk `Income`, tapi termasuk `Response`).
  * **Skenario A (Seleksi Fitur):** Menggunakan **LASSO (L1)**, yang memilih **7 fitur** paling relevan (fitur teratas: `NumWebVisitsMonth`, `MntWines`).
  * **Skenario B (PCA):** Menerapkan PCA pada 15 fitur numerik. "Scree Plot" menunjukkan **13 komponen** diperlukan untuk menangkap 95% varians.
  * **Output:** Dataset `X_reg...` dan `y_reg...` (total 6 file).

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

### Tahap 3: Modelling - Klasifikasi (`classification.ipynb`)

*Notebook* ini mengeksekusi pelatihan model klasifikasi menggunakan output dari Tahap 2. Tujuannya adalah memprediksi apakah pelanggan akan menerima tawaran kampanye (`Response`).

#### Metodologi Eksekusi

1.  **Deep Dive EDA:** Memvalidasi karakteristik data sebelum pelatihan.
      * *Heatmap* membuktikan bahwa dataset LASSO memiliki multikolinearitas (blok merah), sedangkan dataset PCA bersifat ortogonal (bersih/netral).
      * *Boxplot* membuktikan fitur seperti `MntWines` dan `Recency` memiliki daya pembeda yang kuat.
2.  **Handling Imbalance (SMOTE):** Karena data latih sangat timpang (85% : 15%), teknik **SMOTE** diterapkan khusus pada data latih, meningkatkan sampel minoritas dari 267 menjadi 1525 sampel.
3.  **Model & Tuning:**
      * **K-Nearest Neighbors (KNN):** Dioptimasi dengan `GridSearchCV` (mencari k dan metrik jarak) serta *Threshold Tuning* (menggeser batas probabilitas).
      * **Naive Bayes (Gaussian):** Dioptimasi dengan parameter `var_smoothing` untuk menguji hipotesis independensi fitur pada data PCA.

---

#### Hasil Eksperimen (Key Findings)
Evaluasi dilakukan pada data uji murni (tanpa SMOTE) untuk melihat performa dunia nyata:

| Peringkat | Model & Data | F1-Score | Akurasi | Analisis |
| :--- | :--- | :--- | :--- | :--- |
| **#1** | **Naive Bayes + PCA** | **0.534** | 84.8% | **Paling Seimbang & Efisien.** Transformasi PCA sukses menghilangkan korelasi antar-fitur, membuat Naive Bayes bekerja optimal dengan tingkat kesalahan (*False Positive*) terendah. |
| **#2** | **KNN + PCA** | 0.527 | **86.4%** | **Akurasi Tertinggi, namun Konservatif.** Model menjadi terlalu berhati-hati (threshold tinggi) sehingga Akurasi naik, tetapi banyak melewatkan pelanggan potensial (*Recall* rendah). |
| **#3** | **KNN + LASSO** | 0.519 | 81.0% | **Paling Agresif (*Recall Champion*).** Mencatat **Recall tertinggi (0.69)**. Sangat sensitif mendeteksi pelanggan potensial, namun dengan biaya kesalahan tebak yang lebih tinggi sehingga akurasi turun. |

**Kesimpulan:**
* Gunakan **Naive Bayes + PCA** untuk efisiensi biaya (minim salah sasaran).
* Gunakan **KNN + LASSO** untuk ekspansi agresif (menangkap pelanggan sebanyak mungkin).

---

## 🛠️ Cara Menjalankan Proyek (Step-by-Step)

Ikuti panduan langkah demi langkah ini untuk mereproduksi hasil eksperimen dari awal hingga akhir tanpa *error*.

### 1\. Persiapan Lingkungan (*Environment Setup*)

Pastikan Python (versi 3.8+) sudah terinstal. Sangat disarankan menggunakan *virtual environment* (seperti Anaconda atau `venv`).

Install *library* yang dibutuhkan dengan menjalankan perintah berikut di terminal:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyterlab
```

### 2\. Persiapan Data Mentah

1.  Clone repositori ini ke komputer lokal Anda.
2.  Unduh dataset asli dari Kaggle (link di atas).
3.  Ubah nama file menjadi **`marketing_campaign.csv`**.
4.  **PENTING:** Letakkan file `marketing_campaign.csv` tersebut di **direktori root** (folder paling luar dari proyek ini), sejajar dengan folder-folder `Pre-processing...`.

### 3\. Eksekusi Pipeline

Proses ini harus dijalankan secara berurutan (Sekuensial).

#### **Langkah 1: Pembersihan Data (Tahap I)**

1.  Buka folder `📂 Pre-processing I - Data Cleaning & Data Transformation/`.
2.  Jalankan *notebook* **`preprocessing1.ipynb`**.
3.  Setelah selesai, *notebook* akan menghasilkan dua file baru di folder yang sama:
      * `data_train_preprocessed.csv`
      * `data_test_preprocessed.csv`
4.  **Tindakan Manual:** Salin (Copy) atau Pindahkan (Move) kedua file `.csv` tersebut ke folder **`📂 Pre-processing II - Feature Engineering/`**.

#### **Langkah 2: Optimasi Fitur (Tahap II)**

1.  Buka folder `📂 Pre-processing II - Feature Engineering/`.
2.  Pastikan file input dari Langkah 1 sudah ada di folder ini.
3.  Jalankan *notebook* **`preprocessing2.ipynb`**.
4.  Notebook ini akan menghasilkan **12 file CSV** (kombinasi data latih/uji untuk Klasifikasi dan Regresi).
5.  **Tindakan Manual:** Untuk melanjutkan ke tahap klasifikasi, cari 6 file berikut dan **pindahkan** ke folder **`📂 Classification - KNN & NB/`**:
      * `X_class_train_selected.csv`, `X_class_test_selected.csv`
      * `X_class_train_pca.csv`, `X_class_test_pca.csv`
      * `y_class_train.csv`, `y_class_test.csv`

#### **Langkah 3: Pemodelan Klasifikasi (Tahap III)**

1.  Buka folder `📂 Classification - KNN & NB/`.
2.  Pastikan ke-6 file CSV dari langkah sebelumnya sudah berada di folder ini (sejajar dengan notebook).
3.  Jalankan *notebook* **`classification.ipynb`**.
4.  Notebook akan otomatis memuat data, melakukan validasi EDA, menyeimbangkan data dengan SMOTE, melatih model KNN & Naive Bayes, serta menampilkan tabel peringkat hasil akhirnya.

**Catatan Troubleshooting:**

  * Jika Anda menemui error `FileNotFoundError`, periksa kembali apakah Anda sudah memindahkan file `.csv` output dari tahap sebelumnya ke folder tahap yang sedang Anda jalankan.
  * Pastikan nama file tidak diubah-ubah (harus sesuai persis dengan daftar di atas).

----
