# README.md

## (Bahasa Indonesia 🇮🇩)

# Pemrosesan Citra Invoice dan Struk untuk Ekstraksi Informasi

Repositori ini berisi kode dan hasil dari penelitian skripsi yang berfokus pada deteksi objek dan ekstraksi informasi dari citra invoice (faktur) dan struk belanja. Penelitian ini mengeksplorasi beberapa pendekatan dan teknologi untuk mencapai hasil yang optimal.

---

### **Deskripsi Proyek**

Tujuan utama dari penelitian ini adalah untuk membangun dan mengevaluasi model yang mampu mendeteksi area spesifik pada struk dan invoice, seperti **nama item, kuantitas (qty), dan harga**, untuk kemudian diekstraksi informasinya. Proyek ini membandingkan beberapa metode, mulai dari model Deep Learning hingga pemanfaatan library OCR yang sudah ada.

---

### **Metodologi & Pendekatan**

Penelitian ini menggunakan tiga pendekatan utama untuk model deteksi objek:

1.  **Model Berbasis TensorFlow**
    * Menggunakan arsitektur dari TensorFlow Object Detection API untuk melatih model deteksi pada dataset struk dan invoice.

2.  **YOLOv8 dengan Region of Interest (ROI) Sempit**
    * Model ini dilatih menggunakan YOLOv8 untuk mendeteksi tiga kelas label secara terpisah: `item_name`, `price`, dan `qty`.
    * Pendekatan ini berfokus pada deteksi setiap komponen secara individual dengan bounding box yang presisi.

3.  **YOLOv8 dengan Region of Interest (ROI) Luas**
    * Sebuah pendekatan alternatif menggunakan YOLOv8 di mana seluruh area yang memuat item, harga, dan kuantitas digabungkan menjadi satu label tunggal, yaitu `table`.
    * Tujuannya adalah untuk mendeteksi seluruh tabel transaksi sebagai satu kesatuan, yang kemudian dapat diproses lebih lanjut.

---

### **Eksperimen Tambahan**

#### **1. Pengujian dengan OCR Konvensional**

Sebagai perbandingan, dilakukan juga eksperimen ekstraksi teks menggunakan library populer:
* **EasyOCR**: Sebuah library OCR berbasis Python yang mudah digunakan.
* **PaddleOCR**: Alat OCR yang dikembangkan oleh Baidu dengan performa tinggi.

Eksperimen ini bertujuan untuk membandingkan hasil ekstraksi dari model deteksi objek + OCR dengan hasil dari library OCR secara langsung.

#### **2. Analisis Explainable AI (XAI)**

Untuk memahami bagaimana model "melihat" dan membuat keputusan, dilakukan analisis XAI pada citra struk dan invoice menggunakan metode:
* **Grad-CAM (Gradient-weighted Class Activation Mapping)**: Menghasilkan peta panas (heatmap) untuk menunjukkan area pada citra yang paling berpengaruh terhadap prediksi kelas tertentu.
* **Saliency Map**: Menyoroti piksel-piksel pada citra input yang paling signifikan dalam proses pengambilan keputusan model.

#### **3. Eksplorasi Model NER (Named Entity Recognition)**

Sebuah model NER juga sempat dikembangkan untuk mengidentifikasi entitas seperti nama produk dan harga dari teks mentah. Namun, setelah evaluasi, disimpulkan bahwa pendekatan ini **kurang cocok** untuk struktur data pada struk/invoice dalam konteks penelitian ini, sehingga tidak diadopsi dalam solusi akhir.

---

### **Struktur Repositori**

* `/bbox_detection_model.ipynb`: Kode, notebook, dan hasil pelatihan model TensorFlow.
* `/yolo_small_bbox.ipynb`: Konfigurasi dan hasil pelatihan untuk model YOLOv8 dengan label `item_name`, `price`, `qty`.
* `/final_model.ipynb`: Konfigurasi dan hasil pelatihan untuk model YOLOv8 dengan label `table`.
* `/explainable_ai_invoice`: Notebook dan hasil visualisasi Grad-CAM dan Saliency Map pada invoice.
* `/explainable_ai_receipt`: Notebook dan hasil visualisasi Grad-CAM dan Saliency Map pada receipt.

---

## (English 🇬🇧/🇺🇸)

# Invoice and Receipt Image Processing for Information Extraction

This repository contains the code and findings of a thesis research project focused on object detection and information extraction from invoice and receipt images. This study explores several approaches and technologies to achieve optimal results.

---

### **Project Description**

The main objective of this research is to build and evaluate models capable of detecting specific regions on receipts and invoices, such as **item name, quantity (qty), and price**, for subsequent information extraction. The project compares several methods, ranging from Deep Learning models to the use of existing OCR libraries.

---

### **Methodologies & Approaches**

This research employs three primary approaches for the object detection model:

1.  **TensorFlow-Based Model**
    * Utilizes the TensorFlow Object Detection API architecture to train a detection model on a dataset of receipts and invoices.

2.  **YOLOv8 with a Narrow Region of Interest (ROI)**
    * This model is trained using YOLOv8 to detect three separate class labels: `item_name`, `price`, and `qty`.
    * This approach focuses on detecting each component individually with precise bounding boxes.

3.  **YOLOv8 with a Wide Region of Interest (ROI)**
    * An alternative approach using YOLOv8 where the entire area containing items, prices, and quantities is merged into a single label: `table`.
    * The goal is to detect the entire transaction table as a single unit, which can then be further processed.

---

### **Additional Experiments**

#### **1. Conventional OCR Testing**

For comparison, text extraction experiments were also conducted using popular libraries:
* **EasyOCR**: A user-friendly OCR library based on Python.
* **PaddleOCR**: A high-performance OCR tool developed by Baidu.

This experiment aimed to compare the extraction results from the object detection + OCR pipeline with the results from using these OCR libraries directly.

#### **2. Explainable AI (XAI) Analysis**

To understand how the model "sees" and makes decisions, an XAI analysis was performed on receipt and invoice images using the following methods:
* **Grad-CAM (Gradient-weighted Class Activation Mapping)**: Generates a heatmap to show which areas in the image were most influential for a particular class prediction.
* **Saliency Map**: Highlights the pixels in the input image that were most significant in the model's decision-making process.

#### **3. Named Entity Recognition (NER) Model Exploration**

An NER model was also developed to identify entities like product names and prices from raw text. However, after evaluation, it was concluded that this approach was **not well-suited** for the data structure of receipts/invoices in the context of this research and was therefore not adopted in the final solution.

---

### **Repository Structure**

* `/bbox_detection_model.ipynb`: Code, notebooks, and training results for the TensorFlow model.
* `/yolo_small_bbox.ipynb`: Configuration and training results for the YOLOv8 model with `item_name`, `price`, `qty` labels.
* `/final_model.ipynb`: Configuration and training results for the YOLOv8 model with the `table` label.
* `/explainable_ai_invoice`: Notebooks and visualization results for Grad-CAM and Saliency Maps on invoice.
* `/explainable_ai_receipt`: Notebooks and visualization results for Grad-CAM and Saliency Maps on receipt.
