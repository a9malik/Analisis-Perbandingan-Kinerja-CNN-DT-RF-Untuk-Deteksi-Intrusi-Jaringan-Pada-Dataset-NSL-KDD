# Analisis-Perbandingan-Kinerja-CNN-DT-RF-Untuk-Deteksi-Intrusi-Jaringan-Pada-Dataset-NSL-KDD
**Deskripsi Proyek**

Proyek ini bertujuan untuk membandingkan kinerja Algoritma, Convolutional Network (CNN), Decision Tree (DT) dan Random Forest (RF) dalam mendeteksi jaringan dengan berbasis dataset NSL-KDD

**Abstrak**

Di era transformasi digital, serangan siber menjadi ancaman serius terhadap keamanan infrastruktur jaringan. Intrusion Detection System (IDS) merupakan salah satu solusi penting untuk mendeteksi aktivitas jaringan yang mencurigakan. Namun, metode deteksi konvensional seperti signature-based dan anomaly-based masih memiliki keterbatasan, terutama dalam mendeteksi serangan baru dan dalam mengurangi tingkat false positive. Oleh karena itu, pendekatan berbasis machine learning semakin banyak digunakan untuk meningkatkan efektivitas deteksi intrusi.
Penelitian ini bertujuan untuk menganalisis dan membandingkan kinerja tiga algoritma, yaitu dua algoritma klasifikasi klasik yaitu Decision Tree (DT) dan Random Forest (RF), dengan algoritma Deep Learning yaitu Convolutional Neural Network (CNN) dalam mendeteksi intrusi jaringan menggunakan dataset NSL-KDD. Dataset ini merupakan versi perbaikan dari KDD Cup 99 dan banyak digunakan sebagai benchmark dalam penelitian IDS. Setiap algoritma diuji menggunakan evaluasi akurasi untuk mengetahui tingkat efektivitas dalam mengklasifikasi trafik jaringan sebagai normal atau serangan.
Hasil penelitian menunjukkan bahwa tingkat akurasi Random Forest (RF) lebih tinggi diikut oleh Decision Tree (DT) dan terakhir Convolutional Neural Network (CNN). CNN meskipun lebih canggih tapi bukan berarti selalu lebih baik untuk semua jenis data. Dataset NSL-KDD tidak cocok dengan CNN karena karakteristik data tabularnya tidak sesuai dengan cara kerja CNN. CNN dirancang untuk data spasial atau sekuensial seperti Gambar (image Recognition and Detection), Video, Sinyal 1D (suara, EKG), dan Teks berurutan.

**Dataset**

NSL-KDD  Dataset bechmark yang banyak digunakan untuk penelitian dalam bidang Intrusion Detection System (IDS). Dataset ini merupakan versi perbaikan dari KDD Cup 99 dengan menghilangkan data duplikat dan redundan, sehingga lebih seimbang dan realistis untuk pelatihan model machine learning.

NSL-KDD berisi data lalu lintas jaringan yang diklasifikasikan ke dalam dua kategori yaitu Normal dan Intrusi (misalnya DoS, Probe, R2L, dan U2R)

Setiap entri dalam dataset terdiri dari 41 fitur, termasuk fitur numerik dan kategorikal, serta satu label target.
- KDDTrain+ : Data Pelatihan  [KDDTrain+.txt](./KDDTrain+.txt)
- KDDTest+ : Data Pengujian   [KDDTest+.txt](./KDDTest+.txt)

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
  - Menggunakan 2 layer Conv1D (untuk menangkap fitur lebih kompleks), pada dataset NSL KDD cukup menggunakan 2 layer karena dataset ini kecil dibandingkan dengan dataset gambar, seperti VGG16 (16 layer), ResNet (152 layer), atau InceptionV3 (48 layer), semakin banyak layer semakin banyak membutuhkan daya komputasi
    - Conv1D(64, kernel_size=3, activation='relu')
  - Mengelola fitur-fitur secara menyeluruh setelah fitur diekstrasi oleh layer Conv1D
    - Dense(128, activation='relu')
  - Output akhir dalam bentuk biner (probabilitas antara 0-1)
    - Dense(1, activation='sigmoid')
  - Menggunakan opimizer Adam (Adaptive Momen Estimation), cocok untuk data kompleks dan noise sepert NSL-KDD karena cepat belajar dan stabil, metric evaluasi menggunakan accuracy (presentasi prediksi yang benar)
    - model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  - Melatih model dengan epoch=15 (1 epoch=1 kali seluruh dataset dilatih), batch_size=128 (jumlah sampe data yang diproses sekaligus, lebih besar maka lebih cepat proses pelatihannya tapi membutuhkan lebih banyak memori)
    - model.fit(X_train, y_train, epochs=15, batch_size=128, validation_split=0.2, verbose=1)
- Evaluasi Akurasi
    Dilakukan disetiap modul model nya
    - dt_accuracy = evaluate_decision_tree(dt_model, X_test, y_test)
    - rf_accuracy = evaluate_random_forest(rf_model, X_test, y_test)
    - cnn_accuracy = evaluate_cnn(cnn_model, X_test_cnn, y_test_cnn)
- Visualisasi hasil
    - print(f" Akurasi Decision Tree: {dt_accuracy:.4f}")
    - print(f" Akurasi Random Forest: {rf_accuracy:.4f}")
    - print(f" Akurasi CNN: {cnn_accuracy:.4f}")

File :  [decision_tree.py](./decision_tree.py) , [random_forest.py](./random_forest.py) , [cnn.py](./cnn.py) dan [main.py](./main.py) 
