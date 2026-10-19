import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

klasor = os.path.dirname(os.path.abspath(__file__))
yol = os.path.join(klasor, "winniethepooh.jpg")

# 1) Görüntüyü oku (BGR sırasıyla gelir)
img = cv2.imread(yol)
if img is None:
    raise FileNotFoundError("winniethepooh.jpg okunamadı.")


# 2) Kendi yazdığımız calcHist fonksiyonu
def my_calc_hist(kanal):
    """Tek kanallı (2B) görüntü için 256 kutulu histogram hesaplar."""
    hist = np.zeros(256, dtype=np.int64)
    yukseklik, genislik = kanal.shape
    for y in range(yukseklik):
        for x in range(genislik):
            hist[kanal[y, x]] += 1
    return hist


# 3) Kanalları ayır ve her biri için ayrı ayrı çalıştır
b, g, r = cv2.split(img)

hist_b = my_calc_hist(b)
hist_g = my_calc_hist(g)
hist_r = my_calc_hist(r)

# 4) Doğruluk kontrolü
toplam = img.shape[0] * img.shape[1]
print("Toplam piksel:", toplam)
print("Mavi toplam  :", hist_b.sum())
print("Yeşil toplam :", hist_g.sum())
print("Kırmızı toplam:", hist_r.sum())

# Hazır fonksiyonla karşılaştırma
for ad, kanal, h in [("Mavi", b, hist_b), ("Yeşil", g, hist_g), ("Kırmızı", r, hist_r)]:
    hazir = cv2.calcHist([kanal], [0], None, [256], [0, 256]).flatten()
    print(f"{ad}: hazır fonksiyonla aynı mı? ->", np.array_equal(h, hazir))

# 5) Her kanalı ayrı grafikte çiz
kanallar = [(hist_b, "Mavi", "blue"), (hist_g, "Yeşil", "green"), (hist_r, "Kırmızı", "red")]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (hist, ad, renk) in zip(axes, kanallar):
    ax.bar(range(256), hist, color=renk, width=1.0)
    ax.set_title(f"{ad} Kanal Histogramı")
    ax.set_xlabel("Piksel değeri (0-255)")
    ax.set_ylabel("Piksel sayısı")
    ax.set_xlim([0, 255])

plt.tight_layout()
plt.savefig(os.path.join(klasor, "histogramlar.png"), dpi=150)
plt.show()