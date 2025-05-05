# Analisis-Perbandingan-Kinerja-CNN-DT-RF-Untuk-Deteksi-Intrusi-Jaringan-Pada-Dataset-NSL-KDD
**Deskripsi Proyek**

Proyek ini bertujuan untuk membandingkan kinerja Algoritma, Convolutional Network (CNN), Decision Tree (DT) dan Random Forest (RF) dalam mendeteksi jaringan dengan berbasis dataset NSL-KDD

**Abstrak**

Di era transformasi digital, serangan siber menjadi ancaman serius terhadap keamanan infrastruktur jaringan. Intrusion Detection System (IDS) merupakan salah satu solusi penting untuk mendeteksi aktivitas jaringan yang mencurigakan. Namun, metode deteksi konvensional seperti signature-based dan anomaly-based masih memiliki keterbatasan, terutama dalam mendeteksi serangan baru dan dalam mengurangi tingkat false positive. Oleh karena itu, pendekatan berbasis machine learning semakin banyak digunakan untuk meningkatkan efektivitas deteksi intrusi.
Penelitian ini bertujuan untuk menganalisis dan membandingkan kinerja tiga algoritma klasifikasi, yaitu Convolutional Neural Network (CNN), Decision Tree (DT), dan Random Forest (RF), dalam mendeteksi intrusi jaringan menggunakan dataset NSL-KDD. Dataset ini merupakan versi perbaikan dari KDD Cup 99 dan banyak digunakan sebagai benchmark dalam penelitian IDS. Setiap algoritma diuji menggunakan metrik evaluasi seperti akurasi, presisi, recall, dan F1-score untuk mengetahui tingkat efektivitas dalam mengklasifikasi trafik jaringan sebagai normal atau serangan.
Hasil penelitian menunjukkan bahwa CNN memiliki performa yang lebih baik dalam mendeteksi pola serangan secara kompleks dibandingkan DT dan RF, meskipun membutuhkan waktu pelatihan yang lebih lama. Sementara itu, Random Forest menunjukkan kinerja yang stabil dan cukup tinggi, sedangkan Decision Tree memiliki waktu pelatihan tercepat namun dengan akurasi yang lebih rendah.

**Dataset**

NSL-KDD  Dataset bechmark yang banyak digunakan untuk penelitian dalam bidang Intrusion Detection System (IDS). Dataset ini merupakan versi perbaikan dari KDD Cup 99 dengan menghilangkan data duplikat dan redundan, sehingga lebih seimbang dan realistis untuk pelatihan model machine learning.

NSL-KDD berisi data lalu lintas jaringan yang diklasifikasikan ke dalam dua kategori yaitu Normal dan Intrusi (misalnya DoS, Probe, R2L, dan U2R)

Setiap entri dalam dataset terdiri dari 41 fitur, termasuk fitur numerik dan kategorikal, serta satu label target.
- KDDTrain+ : Data Pelatihan  
- KDDTest+ : Data Pengujian

Sumber dataset:  [NSL-KDD (UNB)](https://www.unb.ca/cic/datasets/nsl.html)

**Fitur Proyek**

- Preprocessing data
- Pembagian data train dan test
- Pelatihan model:
  - Decision Tree (DT)
  - Random Forest (RF)
  - Convolutional Neural Network (CNN)
- Evaluasi akurasi
- Visualisasi hasil
- Perbandingan kinerja model

**PreProcessing**

- Menggabungkan Data Train dan Test
  - def load_and_preprocess_dataset(train_path, test_path)
- Pelabelan columns (meski pada dataset tidak ada label, tapi ada keterangan tentang labelnya dari dokumentasi asli dataset NSL-KDD)
  - columns = [....]
- Dropping (Column Removal)
  - df = pd.concat([train_df, test_df], ignore_index=True)
- Mengubah fitur kategorikal menjadi bentuk numerik (karena model tidak bisa memproses data bertipe string, LabelEncoder memberikan angka mulai dari 0)
  - for col in ['protocol_type', 'service', 'flag']:
    - df[col] = LabelEncoder().fit_transform(df[col])
- Mengubah label menjadi numerik biner (0 untuk normal, 1 untuk serangan/intrusi)
  - df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
- Memisahkan fitur dan target
  - X = df.drop('label', axis=1)
  - y = df['label']
- Normalisasi atau Standarisasi fitur
  - scaler = StandardScaler()
  - X_scaled = scaler.fit_transform(X)

File :  [preprocessing.py](./preprocessing.py) dan [columns.txt](./columns.txt)

**Pembagian Data Train dan Test**

Setelah data digabungkan pada Tahap preprocessing, kemudian data dipisahkan kembali untuk dibagi menjadi data Train dan Test  
- Membagi Data (Train 80% dan Test 20%)
  - X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

**Pelatihan model:**

- Decision Tree (DT)
  - Menggunakan kelas DecisionTreeClassifer pada library scikit-learn
    - from sklearn.tree import DecisionTreeClassifier
  - Menggunakan akurasi prediksi untuk fungsi evaluasinya
    - from sklearn.metrics import accuracy_score
- Random Forest (RF)
  - Menggunakan kelas RandomForestClassifer pada library scikit-learn
    - from sklearn.ensemble import RandomForestClassifier
  - Menggunakan akurasi prediksi untuk fungsi evaluasinya
    - from sklearn.metrics import accuracy_score
- Convolutional Neural Network (CNN)
  - Menggunakan framework Tensorflow dengan interface keras yang menambahkan layer Sequential (satu per satu secara berurutan)
    - from tensorflow.keras.models import Sequential
  - Menggunakan layer konvolusi 1 Dimensi karena Dataset Tabular (berbentuk tabel) bukan data gambar
    - from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
  - Menggunakan optimizer
    - from tensorflow.keras import optimizers
  - Menggunakan konvolusi 32 filter, kernel ukuran 3 dan fungsi aktivasi ReLU (Rectified Linear Unit) membuat fungsi non-linear karena dataset komplek
    - Conv1D(32, kernel_size=3, activation='relu')

File :  [decision_tree.py](./decision_tree.py) , [random_forest.py](./random_forest.py) dan [cnn.py](./cnn.py)
