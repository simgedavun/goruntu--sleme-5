import os
import cv2
import numpy as np

klasor = os.path.dirname(os.path.abspath(__file__))
yol = os.path.join(klasor, "winniethepooh.jpg")

img = cv2.imread(yol)  # BGR sırasıyla okunur
if img is None:
    raise FileNotFoundError("winniethepooh.jpg okunamadı.")


# 1) OpenCV hazır fonksiyonu
gri_opencv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# 2) Kendi yazdığımız fonksiyon
def my_rgb_to_gray(bgr_img):
    """BGR görüntüyü ağırlıklı toplamla tek kanallı griye çevirir."""
    yukseklik, genislik, _ = bgr_img.shape
    gri = np.zeros((yukseklik, genislik), dtype=np.uint8)
    for y in range(yukseklik):
        for x in range(genislik):
            b = float(bgr_img[y, x, 0])
            g = float(bgr_img[y, x, 1])
            r = float(bgr_img[y, x, 2])
            deger = 0.299 * r + 0.587 * g + 0.114 * b
            gri[y, x] = int(round(deger))
    return gri


gri_kendi = my_rgb_to_gray(img)


# 3) Hata değeri: eşleşen piksellerin mutlak farklarının toplamı
fark = np.abs(gri_opencv.astype(np.int32) - gri_kendi.astype(np.int32))
hata = fark.sum()

print("OpenCV gri şekli :", gri_opencv.shape)
print("Kendi gri şekli  :", gri_kendi.shape)
print("Toplam piksel    :", gri_opencv.size)
print("Farklı piksel sayısı:", np.count_nonzero(fark))
print("En büyük piksel farkı:", fark.max())
print("HATA (mutlak fark toplamı):", hata)

# Görselleri kaydet
cv2.imwrite(os.path.join(klasor, "gri_opencv.png"), gri_opencv)
cv2.imwrite(os.path.join(klasor, "gri_kendi.png"), gri_kendi)