import cv2
import numpy as np

def resize_to_height(img, height):
    """Resize keeping the aspect ratio. Returns (resized, ratio)."""
    ratio = height / img.shape[0]
    width = int(img.shape[1] * ratio)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA), ratio

def load_image(path):
    """REad an image from disk (BGR format). Raises an error if it fails."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read images: {path}")
    return img

def image_info(img):
    """Return (height, width, channels)."""
    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1
    return h, w, channels

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGRHSV)

def crop(img, x, y, w, h):
    return img[:y + h, x:x + w]

def rotate(img, angle):
    """Rotate around the cetre by 'angle' degrees (counter-clockwise)."""
    h,w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.wrapAffine(img, matrix, (w, h))

def blur_gaussian(img, k=5):
    """Smooths noise. k must be odd."""
    return cv2.GaussianBlur(img, (k, k), 0)

def blur_median(img, k=5):
    """Good at removing salt-and-pepper noise while keeping edges. k must be odd."""
    return cv2.medianBlur(img, k)

def sharpen(img, kernel):
    """Sharpend using a 3x3 concolution kernel."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)

def threshold_simple(gray, value=127):
    _, out =cv2.threshold(gray, value, 255, cv2.THRESH_BINARY)
    return out

def threshold_otsu(gray):
    """Otsu picks the threshold automatically from the histogram."""
    _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out

def threshold_adaptive(gray, block_size=11, c=2):
    """Threshold computed per small region, so it handles uneven lighting."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, c,
    )

def detect_edges(gray, low=75, high=200):
    """Canny edge detection (expects a blurred grayscale image)."""
    return cv2.Canny(gray, low, high)
 
 
def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    """out = alpha * img + beta. alpha = contrast, beta = brightness."""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def equalize_histogram(gray):
    """Spread ouyt the intensity values to improve contrast."""
    return cv2.equalizeHist(gray)

def to_color(img):
    """Make sure the image has 3 colour channels (BGR). Grayscale input is converted."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def hdr_detect(img, clip_limit=2.0):
    """HDR-style local contrast boost (CLAHE on the lightness channel).
 
    Helps edge detection when the photo has shadows or uneven light.
    clip_limit: higher = stronger contrast (2 is a good start).
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)