# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 22:01:20 2023

@author: Lenovo
"""

import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os
import random
from tkinter import *
from PIL import Image, ImageTk

# Duygu ifadeleri listesi
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Eğitilmiş modeli yükle
model_path = 'D:/tespit/model_optimal.h5'
model = load_model(model_path)

# Yüz algılama için Cascade sınıflandırıcısını yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# GUI penceresini oluştur
window = Tk()
window.title("Duygu Tanıma Uygulaması")

# Kamera akışını aç
video_capture = cv2.VideoCapture(0)

# Kamera için frame'i güncelleme fonksiyonu
def update_frame():
    # Kameradan bir görüntü al
    ret, frame = video_capture.read()

    # Görüntüyü gri tonlamalı hale getir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri algıla
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Algılanan yüzleri döngüye al
    for (x, y, w, h) in faces:
        # Yüz bölgesini kırp
        face_image = gray[y:y+h, x:x+w]
        # Yüzü 48x48 boyutuna yeniden boyutlandır
        face_image = cv2.resize(face_image, (48, 48))
        # Piksel değerlerini normalize et
        face_image = face_image / 255.0
        # Dizi şeklini uygun hale getir
        face_image = np.expand_dims(face_image, axis=0)
        face_image = np.expand_dims(face_image, axis=-1)

        # Duygu tahminini yap
        emotion_prediction = model.predict(face_image)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_label = emotion_labels[emotion_label_arg]
        
        # Yüzün etrafına dikdörtgen çiz
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Duygu ifadesini metin olarak ekle
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Görüntüyü PIL formatına dönüştür
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    

    # Resmi güncelle
    img_tk = ImageTk.PhotoImage(img)
    camera_label.config(image=img_tk)
    camera_label.image = img_tk
    


    # Yeniden güncelleme için işleme çağrısı yap
    window.after(10, update_frame)

# Başlatma butonu için fonksiyon
def start_camera():
    video_capture.release()
    
    # Kamerayı aç
    video_capture.open(0)
    
    # Frame'i güncelle
    update_frame()

# Müzik önerisi butonu için fonksiyon
def recommend_music():
    # Kameradan bir görüntü al
    ret, frame = video_capture.read()

    # Görüntüyü gri tonlamalı hale getir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri algıla
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Algılanan yüzleri döngüye al
    for (x, y, w, h) in faces:
        # Yüz bölgesini kırp
        face_image = gray[y:y+h, x:x+w]
        # Yüzü 48x48 boyutuna yeniden boyutlandır
        face_image = cv2.resize(face_image, (48, 48))
        # Piksel değerlerini normalize et
        face_image = face_image / 255.0
        # Dizi şeklini uygun hale getir
        face_image = np.expand_dims(face_image, axis=0)
        face_image = np.expand_dims(face_image, axis=-1)

        # Duygu tahminini yap
        emotion_prediction = model.predict(face_image)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_label = emotion_labels[emotion_label_arg]
        
        # Yüzün etrafına dikdörtgen çiz
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Duygu ifadesini metin olarak ekle
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Müzik resmi öner
        if emotion_label == "Happy":
            # Müzik resimleri dosya yolunu belirtin
            music_images_dir = "D:/tespit/music_dataset/happy_music"
            # Müzik resimlerini al
            music_images = os.listdir(music_images_dir)
            # Rastgele bir müzik resmi seç
            random_music_image = random.choice(music_images)
            # Müzik resmini yükle
            music_image_path = os.path.join(music_images_dir, random_music_image)
            music_img = Image.open(music_image_path)
            # Müzik resmini sonuç frame'ine yükle
            music_img_tk = ImageTk.PhotoImage(music_img)
            result_label.config(image=music_img_tk)
            result_label.image = music_img_tk
            break
                
            
            
        if emotion_label == "Sad":
            # Müzik resimleri dosya yolunu belirtin
            music_images_dir = "D:/tespit/music_dataset/sad_music"
            # Müzik resimlerini al
            music_images = os.listdir(music_images_dir)
            # Rastgele bir müzik resmi seç
            random_music_image = random.choice(music_images)
            # Müzik resmini yükle
            music_image_path = os.path.join(music_images_dir, random_music_image)
            music_img = Image.open(music_image_path)
            # Müzik resmini sonuç frame'ine yükle
            music_img_tk = ImageTk.PhotoImage(music_img)
            result_label.config(image=music_img_tk)
            result_label.image = music_img_tk
            break
            

# Kamera için frame'i oluştur
camera_label = Label(window)
camera_label.pack(side=LEFT)

# Sonuç için frame'i oluştur
result_label = Label(window)
result_label.pack(side=RIGHT)

# Başlatma butonunu oluştur
start_button = Button(window, text="Kamerayı Başlat", command=start_camera)
start_button.pack()

# Müzik önerisi butonunu oluştur
recommend_button = Button(window, text="Müzik Öner", command=recommend_music)
recommend_button.pack()

# GUI penceresini güncelle
window.mainloop()



    
   
