# Computer-Vision-Natural-Language-Processing
Repository ini berisi source code mengenai penugasan kelompok pada Assignment 3 - Computer Vision &amp; Natural Language Processing. Antara lain:
1. Case 1 (03-Tim 5-1.ipynb): 
Program ini adalah sebuah notebook Python yang digunakan dalam tugas Computer Vision di Startup Campus, Indonesia, untuk mempelajari pemrosesan gambar digital. Fokus utama tugas ini adalah meningkatkan kualitas gambar (terutama yang gelap) menggunakan teknik Max Pooling, Min Pooling, Average Pooling, dan Contrast Limited Adaptive Histogram Equalization (CLAHE). Berikut adalah penjelasan singkat mengenai cara kerja program ini:
- Library Import: Program ini menggunakan beberapa pustaka utama seperti OpenCV, Scikit-image, NumPy, Matplotlib, dan PyTorch. Library-library ini mendukung operasi pemrosesan gambar dan tampilan grafik.
- User-Defined Functions: 
plot : Menampilkan beberapa gambar dalam satu figure dengan judul dan peta warna tertentu.
apply_clahe_rgb: Mengubah gambar ke model warna LAB, menerapkan CLAHE pada komponen kecerahan (L), dan mengembalikannya ke model warna BGR.
- Image Processing:
Mengubah gambar asli menjadi gambar grayscale, kemudian menjadi gambar biner.Menampilkan histogram gambar asli dan grayscale untuk menunjukkan distribusi intensitas warna.
- Max Pooling
- Min Pooling dan Average Pooling
- CLAHE
Output: Hasil akhir gambar yang disempurnakan dengan CLAHE disimpan sebagai file PNG.

2. Case 2 (03-Tim 5-2.ipynb):
Program ini menggunakan transfer learning untuk mengklasifikasikan dataset angka tulisan tangan MNIST (0-9) menggunakan beberapa model CNN yang telah dilatih sebelumnya, seperti ResNet18, DenseNet121, dan Vision Transformer (ViT). Tujuan utamanya adalah untuk menguji pengaruh pembekuan beberapa lapisan jaringan saraf (layer) terhadap performa model dalam tugas klasifikasi. Berikut adalah penjelasan singkat dan cara kerja dari program ini:
- Persiapan Data: 
Dataset MNIST digunakan untuk melatih dan menguji model. Gambar diubah ukurannya menjadi 224x224 piksel dan dinormalisasi. Fungsi get_dataloaders mempersiapkan data pelatihan dan validasi.
- Model:
VisionModel adalah kelas model yang memungkinkan pemilihan model yang berbeda: ResNet, DenseNet, atau ViT. Model yang dipilih dimodifikasi untuk menerima gambar berwarna abu-abu (grayscale) sebagai input (MNIST), karena model awalnya dilatih dengan gambar berwarna RGB. Lapisan terakhir model disesuaikan untuk menghasilkan output 10 kelas (angka 0-9) sesuai dengan dataset MNIST.
- Transfer Learning dan Pembekuan Lapisan:
Model dimulai dengan semua lapisan dapat dilatih (trainable). Setelah pelatihan awal, model diuji dengan beberapa lapisan yang dibekukan. Pada eksperimen pertama, lapisan pertama (denseblock1) dibekukan, lalu pada eksperimen kedua, lapisan pertama dan kedua (denseblock1 dan denseblock2) dibekukan. Tujuannya adalah untuk melihat bagaimana pembekuan lapisan-lapisan tersebut mempengaruhi performa model.
- Pelatihan dan Evaluasi: 
Fungsi fit digunakan untuk melatih model, menghitung akurasi dan kehilangan (loss) selama pelatihan dan validasi, serta menampilkan hasil pada setiap epoch. Setelah pelatihan, performa model divisualisasikan dengan grafik akurasi dan loss.

Output: Setelah melatih model dengan semua lapisan yang dapat dilatih, model diuji dengan lapisan yang dibekukan untuk melihat dampaknya pada akurasi dan waktu pelatihan. Hasil perbandingan antara model yang dilatih sepenuhnya dan model yang dibekukan beberapa lapisannya dianalisis. Lapisan yang dibekukan mungkin memiliki akurasi yang lebih buruk, serta semakin banyak lapisan yang dibekukan, waktu pelatihan menjadi lebih cepat.

3. Case 3 (03-Tim 5-3.ipynb):
Program ini bertujuan untuk melakukan deteksi objek secara real-time menggunakan model CNN berbasis YOLOv5 yang sudah dilatih sebelumnya. Program ini bekerja dengan memanfaatkan video YouTube sebagai sumber input dan mendeteksi objek-objek yang ada dalam video.
- Instalasi dan Pemasangan Dependensi:
Menggunakan pustaka cap-from-youtube untuk mengakses dan memutar video dari YouTube. Library lain yang digunakan adalah PyTorch, OpenCV, dan Numpy untuk pemrosesan video dan deteksi objek.
- Kelas ObjectDetection:
Inisialisasi: Kelas ini memerlukan URL video YouTube dan nama file output untuk menyimpan video yang telah diproses.
Metode get_video_from_url: Mengambil video dari URL YouTube dan mengonversinya menjadi objek video yang dapat diproses frame-by-frame menggunakan OpenCV.
Metode load_model: Memuat model YOLOv5 menggunakan PyTorch Hub. Model ini sudah dilatih untuk mendeteksi berbagai objek umum.
Metode score_frame: Menggunakan model YOLOv5 untuk mendeteksi objek dalam setiap frame video, mengembalikan label objek dan koordinat bounding box.
Metode plot_boxes: Menambahkan bounding box dan label ke dalam frame video berdasarkan hasil deteksi.
Metode __call__: Menjalankan loop untuk membaca frame video, mendeteksi objek, dan menyimpan hasil deteksi ke file output.
- Deteksi Objek:
Untuk setiap frame, model YOLOv5 mendeteksi objek-objek yang ada, dan setiap objek yang terdeteksi akan diberi bounding box dan label sesuai dengan kelas objek tersebut (misalnya, mobil, orang, dll).

Output: 
Setelah diproses, video yang telah diberi bounding box dan label disimpan sebagai file video baru yang dapat diputar untuk melihat hasil deteksi objek secara real-time.

4. Case 4 (03-Tim 5-1.ipynb):
Program ini adalah implementasi untuk mengklasifikasikan tweet terkait bencana menggunakan model BERT (Bidirectional Encoder Representations for Transformers) yang telah dilatih sebelumnya. Tujuan utama dari proyek ini adalah untuk mengklasifikasikan tweet apakah terkait dengan bencana atau tidak, yang dapat digunakan dalam analisis pola komunikasi pada situasi darurat atau sistem peringatan dini berbasis Twitter.
- Library dan Data Setup:
Program mengimpor berbagai library seperti nltk, pandas, torch, dan transformers untuk keperluan pemrosesan teks dan pelatihan model. Dataset yang digunakan adalah dataset "Disaster Tweets", di mana setiap tweet diberi label apakah itu terkait dengan bencana atau tidak.
- Preprocessing Data:
Fungsi clean_text digunakan untuk membersihkan teks tweet, menghapus URL, HTML tags, tanda baca, stopwords (kata yang tidak bermakna), dan emoji. Setelah itu, data dibagi menjadi dua bagian: data training dan data validation (80% untuk training, 20% untuk validation).
- Tokenisasi dengan BERT: 
Fungsi tokenizer_encode mempersiapkan teks agar dapat diproses oleh model BERT. Teks yang sudah dibersihkan diubah menjadi token, yang kemudian dikonversi menjadi input IDs dan attention masks yang diperlukan oleh model BERT.
- Pembuatan DataLoader:
Dataset yang telah diproses dibagi menjadi batch menggunakan DataLoader untuk proses training dan validasi. Batch training diproses dengan RandomSampler dan batch validasi diproses dengan SequentialSampler.
- Model BERT: 
Model BERT pre-trained bert-base-uncased digunakan untuk klasifikasi. Model ini kemudian di-fine-tune menggunakan data training. Optimizer yang digunakan adalah AdamW, dan learning rate scheduler digunakan untuk menyesuaikan learning rate selama proses training.
- Training dan Evaluasi: 
Selama training, model dihitung loss-nya dan diupdate menggunakan backpropagation. Setiap epoch dilalui dengan training dan evaluasi untuk memonitor akurasi dan loss pada data validasi. Jika akurasi validasi meningkat, model yang terbaik disimpan.
- Testing: Setelah model selesai dilatih, data test digunakan untuk memprediksi kelas tweet baru. 

Output: Hasil prediksi disimpan dalam dataframe yang berisi teks tweet dan prediksi apakah tweet tersebut terkait dengan bencana atau tidak.

