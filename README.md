OpenCV Document Scanner

Turns a photo of a paper into a flat, clean scanned page using Python, OpenCV and NumPy.

Features
Detects the page outline with Gaussian blur, Canny edges and contour analysis
Corrects perspective with a 4-point perspective transform
Outputs a black-and-white (adaptive threshold), grayscale or colour scan
Small image processing toolkit (toolkit.py): colour spaces, resize, crop, rotate, Gaussian/median blur, sharpen, simple/Otsu/adaptive threshold, brightness and contrast, histogram equalisation
--hdr option boosts local contrast (CLAHE) for photos with shadows
--debug mode saves every pipeline step as an image
Install
bash
pip install -r requirements.txt
Run
bash
python scan.py --input samples/photo.jpg --output samples/scan.png
python scan.py --input samples/photo.jpg --output samples/scan.png --mode color --debug
python scan.py --input samples/photo.jpg --output samples/scan.png --hdr
Pipeline
Resize to a working height of 500 px
Grayscale and Gaussian blur to reduce noise
Canny edge detection
Find contours, keep the largest one with 4 corners (approxPolyDP)
Order the corners (top-left, top-right, bottom-right, bottom-left)
Perspective transform on the original full-size image
Adaptive threshold for the scanned look
Limitations
Needs the whole page visible against a contrasting background
Fails with a clear message if no 4-corner outline is found
Future improvements
Port the edge and contour step to C++ with OpenCV
OCR with Tesseract
Webcam mode with cv2.VideoCapture
