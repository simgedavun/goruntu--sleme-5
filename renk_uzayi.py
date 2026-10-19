import os
import cv2
import numpy as np

klasor = os.path.dirname(os.path.abspath(__file__))
img = cv2.imread(os.path.join(klasor, "winniethepooh.jpg"))  # BGR
if img is None:
    raise FileNotFoundError("winniethepooh.jpg okunamadı.")

print("OpenCV sürümü:", cv2.__version__)
print("Görüntü şekli:", img.shape, "| toplam piksel:", img.shape[0] * img.shape[1])
print()


def descale(x, n):
    """Sağa kaydırma ile yuvarlayarak bölme (OpenCV'deki gibi)."""
    return (x + (1 << (n - 1))) >> n


# ---------- 1) BGR -> RGB ----------
def my_bgr2rgb(bgr):
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    return np.dstack((r, g, b))


# ---------- 2) BGR -> GRAY (sabit noktalı) ----------
def my_bgr2gray(bgr):
    b = bgr[..., 0].astype(np.int32)
    g = bgr[..., 1].astype(np.int32)
    r = bgr[..., 2].astype(np.int32)
    gri = descale(r * 4899 + g * 9617 + b * 1868, 14)
    return gri.astype(np.uint8)


# ---------- 3) BGR -> YCrCb ----------
def my_bgr2ycrcb(bgr):
    b = bgr[..., 0].astype(np.int32)
    g = bgr[..., 1].astype(np.int32)
    r = bgr[..., 2].astype(np.int32)
    y = descale(r * 4899 + g * 9617 + b * 1868, 14)
    delta = 128 << 14
    cr = descale((r - y) * 11682 + delta, 14)
    cb = descale((b - y) * 9241 + delta, 14)
    return np.dstack((y, cr, cb)).astype(np.uint8)   # OpenCV sırası: Y, Cr, Cb


# ---------- 4) BGR -> HSV (H: 0-180, S,V: 0-255) ----------
def my_bgr2hsv(bgr):
    b = bgr[..., 0].astype(np.int32)
    g = bgr[..., 1].astype(np.int32)
    r = bgr[..., 2].astype(np.int32)

    v = np.maximum(np.maximum(b, g), r)
    vmin = np.minimum(np.minimum(b, g), r)
    diff = v - vmin

    # OpenCV'nin kullandığı tamsayı bölme tabloları (2^12 ölçekli)
    idx = np.arange(256, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        sdiv = np.where(idx == 0, 0, np.rint(np.float32(255 << 12) / idx)).astype(np.int32)
        hdiv = np.where(idx == 0, 0, np.rint(np.float32(180 << 12) / (np.float32(6) * idx))).astype(np.int32)

    s = descale(diff * sdiv[v], 12)

    h = np.where(v == r, g - b,
        np.where(v == g, b - r + 2 * diff, r - g + 4 * diff))
    h = descale(h * hdiv[diff], 12)
    h = np.where(h < 0, h + 180, h)

    return np.dstack((h, s, v)).astype(np.uint8)


# ---------- Fonksiyon "işaretçileri" (C++'taki function pointer karşılığı) ----------
donusumler = {
    "BGR -> RGB":   (my_bgr2rgb,   cv2.COLOR_BGR2RGB),
    "BGR -> GRAY":  (my_bgr2gray,  cv2.COLOR_BGR2GRAY),
    "BGR -> YCrCb": (my_bgr2ycrcb, cv2.COLOR_BGR2YCrCb),
    "BGR -> HSV":   (my_bgr2hsv,   cv2.COLOR_BGR2HSV),
}

# ---------- Karşılaştırma ----------
print(f"{'Dönüşüm':14s} | {'Toplam mutlak hata':>18s} | {'Farklı değer':>12s} | {'En büyük fark':>13s}")
print("-" * 68)
for ad, (fonk, cv_kodu) in donusumler.items():
    benim = fonk(img)
    opencv = cv2.cvtColor(img, cv_kodu)
    fark = np.abs(benim.astype(np.int32) - opencv.astype(np.int32))
    print(f"{ad:14s} | {fark.sum():18d} | {np.count_nonzero(fark):12d} | {fark.max():13d}")