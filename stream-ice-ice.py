import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
from PIL import Image
import zipfile
import io

# Load model dan scaler
with open('ice-ice_model.sav', 'rb') as model_file:
    knn = pickle.load(model_file)

with open('scaler.sav', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Fungsi untuk resize gambar
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# Fungsi untuk meningkatkan kualitas gambar
def auto_enhance(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    enhanced_image = enhanced_image.astype('uint8')
    return enhanced_image

# Fungsi untuk mengekstrak fitur dari gambar
def extract_features(image_path):
    img = cv2.imread(image_path)
    img = image_resize(img, height=400)
    img2 = img.copy()
    img2 = auto_enhance(img2)

    mean_color = cv2.mean(img2)[:3]

    std_deviation_blue = np.std(img2[:, :, 0])
    std_deviation_green = np.std(img2[:, :, 1])
    std_deviation_red = np.std(img2[:, :, 2])

    skewness_blue = skew(img2[:, :, 0].ravel())
    skewness_green = skew(img2[:, :, 1].ravel())
    skewness_red = skew(img2[:, :, 2].ravel())

    kurtosis_blue = kurtosis(img2[:, :, 0].ravel())
    kurtosis_green = kurtosis(img2[:, :, 1].ravel())
    kurtosis_red = kurtosis(img2[:, :, 2].ravel())

    grayscale_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(grayscale_img2, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    # Calculate texture features
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    entropy = -np.sum(glcm * np.log(glcm + np.finfo(float).eps))
    variance = np.var(img2)

    features = [
        mean_color[0], mean_color[1], mean_color[2],
        std_deviation_red, std_deviation_green, std_deviation_blue,
        skewness_red, skewness_green, skewness_blue,
        kurtosis_red, kurtosis_green, kurtosis_blue,
        contrast, correlation, homogeneity, energy, entropy, variance
    ]
    return features

# Fungsi untuk mengklasifikasikan gambar dan menyimpan hasil
def classify_and_save(image_path, filename):
    features = extract_features(image_path)
    features_scaled = scaler.transform([features])
    hasil = knn.predict(features_scaled)
    hasilProbabilitas = knn.predict_proba(features_scaled)
    return {
        'Nama Gambar': filename,
        'Mean_R': features[0], 'Mean_G': features[1], 'Mean_B': features[2],
        'Std_Dev_R': features[3], 'Std_Dev_G': features[4], 'Std_Dev_B': features[5],
        'Skewness_R': features[6], 'Skewness_G': features[7], 'Skewness_B': features[8],
        'Kurtosis_R': features[9], 'Kurtosis_G': features[10], 'Kurtosis_B': features[11],
        'Contrast': features[12], 'Correlation': features[13], 'Homogeneity': features[14],
        'Energy': features[15], 'Entropy': features[16], 'Variance': features[17],
        'Hasil_KNN': hasil[0], 'Probabilitas Sehat': hasilProbabilitas[0][1] * 100,
        'Probabilitas Sakit': hasilProbabilitas[0][0] * 100
    }

# Fungsi untuk membaca file ice-ice.xlsx
def read_excel():
    excel_file = 'hasil-klasifikasi/ice-ice.xlsx'
    if os.path.exists(excel_file):
        df = pd.read_excel(excel_file)
        df_display = df[['Nama Gambar', 'Homogeneity', 'Energy', 'Entropy', 'Variance', 'Hasil_KNN', 'Probabilitas Sehat', 'Probabilitas Sakit']]
        
        for i, row in df_display.iterrows():
            image_path = os.path.join('gambar-dataset', row['Nama Gambar'])
            if os.path.exists(image_path):
                st.image(image_path, caption=row['Nama Gambar'], use_column_width=True)
            st.write(f"Nama Gambar: {row['Nama Gambar']}")
            st.write(f"Homogeneity: {row['Homogeneity']}")
            st.write(f"Energy: {row['Energy']}")
            st.write(f"Entropy: {row['Entropy']}")
            st.write(f"Variance: {row['Variance']}")
            st.write(f"Hasil KNN: {row['Hasil_KNN']}")
            st.write(f"Probabilitas Sehat: {row['Probabilitas Sehat']:.2f}%")
            st.write(f"Probabilitas Sakit: {row['Probabilitas Sakit']:.2f}%")
            st.write('---')
    else:
        st.write("Belum ada hasil klasifikasi yang tersimpan.")

# fungsi untuk menyimpan gambar yang di upload ke folder gambar-dataset
def upload_image(gambar):
    save_path = os.path.join('gambar-dataset', gambar.name)
    with open(save_path, 'wb') as f:
        f.write(gambar.getbuffer())
    
    st.image(gambar, caption='Gambar yang diunggah.', use_column_width=True)
    
    # Lakukan klasifikasi dan tampilkan hasilnya
    result = classify_and_save(save_path, gambar.name)
    st.write(f"Prediksi: {result['Hasil_KNN']}")
    st.write(f"Probabilitas Sehat: {result['Probabilitas Sehat']:.2f}% | Probabilitas Sakit: {result['Probabilitas Sakit']:.2f}%")
    
    # Simpan hasil klasifikasi ke file Excel
    excel_file = 'hasil-klasifikasi/ice-ice.xlsx'
    if os.path.exists(excel_file):
        df = pd.read_excel(excel_file)
    else:
        df = pd.DataFrame(columns=[
            'Nama Gambar', 'Mean_R', 'Mean_G', 'Mean_B',
            'Std_Dev_R', 'Std_Dev_G', 'Std_Dev_B',
            'Skewness_R', 'Skewness_G', 'Skewness_B',
            'Kurtosis_R', 'Kurtosis_G', 'Kurtosis_B',
            'Contrast', 'Correlation', 'Homogeneity', 'Energy', 'Entropy', 'Variance',
            'Hasil_KNN', 'Probabilitas Sehat', 'Probabilitas Sakit'
        ])
    
    # Menggunakan pd.concat untuk menambahkan baris baru ke DataFrame
    df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    df.to_excel(excel_file, index=False)
    
    st.success("Hasil klasifikasi telah disimpan.")

# Download Excel
def download_excel():
    excel_file = 'hasil-klasifikasi/ice-ice.xlsx'
    if os.path.exists(excel_file):
        with open(excel_file, 'rb') as f:
            st.download_button(
                label="Download file Excel",
                data=f,
                file_name=excel_file,
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
    else:
        st.write("Tidak ada data untuk diunduh.")

# Download Gambar
def download_image():
    image_dir = 'gambar-dataset'
    images = [f for f in os.listdir(image_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
    
    if images:
        # Membuat file ZIP di memori menggunakan BytesIO
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zipf:
            for image in images:
                image_path = os.path.join(image_dir, image)
                zipf.write(image_path, arcname=image)
        
        # Mengatur ulang posisi pointer ke awal
        zip_buffer.seek(0)
        
        st.download_button(
            label="Download semua gambar sebagai ZIP",
            data=zip_buffer,
            file_name='images.zip',
            mime='application/zip'
        )
    else:
        st.write("Tidak ada gambar untuk diunduh.")

# Streamlit layout
st.set_page_config(page_title="Ice-Ice KNN Classifier", layout="wide")
st.title("Ice-Ice KNN Classifier")

# Sidebar untuk navigasi halaman
page = st.sidebar.selectbox("Pilih Halaman", ["Halaman Utama", "Klasifikasi Gambar"])

if page == "Halaman Utama":
    st.header("Hasil Klasifikasi Gambar")
    
    # Download Excel
    download_excel()

    # Download Gambar
    download_image()
    
    # Membaca file Excel yang berisi hasil klasifikasi dari fungsi read_excel()
    read_excel()

elif page == "Klasifikasi Gambar":
    st.header("Unggah Gambar untuk Klasifikasi")
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Simpan gambar yang diunggah ke direktori yang ditentukan menggunakan fungsi upload_image()
        upload_image(uploaded_file)
