import os
import cv2

klasor = os.path.dirname(os.path.abspath(__file__))
yol = os.path.join(klasor, "winniethepooh.jpg")

img = cv2.imread(yol)

if img is None:
    raise FileNotFoundError("winniethepooh.jpg okunamadı. Dosya adını ve klasörünü kontrol et.")

print("Şekil:", img.shape)
print("Veri tipi:", img.dtype)
