""" Document scanner: photo of a paper -> flat,Black and white,Color page,Scanned page.
Usage:
    python scan.py --input <input_image> --output <output_image> --mode <mode>
    color debug: python scan.py --input <input_image< --output <output_image> --mode <mode> --debug
    """

import argparse
import os
import sys

import numpy as np
import cv2

import toolkit as tk

DETECT_HEIGHT = 500 # work on  a smaller copy for faster processing, more stable edge detection

def order_points(pts):
    """order points in the order, top-left, top-right, botton-right, bottom-left"""

    pts = pts.reshape(4,2). astype("float32")
    rect = np.zeros((4,2), dtype="float32")

    s = pts.sum(axis=1) # x+y
    rect[0] = pts[np.argmin(s)] # top-left has the smallest sum
    rect[2] = pts[np.argmax(s)] # bottom-right has the largest sum

    d = np.diff(pts, axis = 1) # y-x
    rect[1] = pts[np.argmin(d)] # top-right has the smallest difference
    rect[3] =pts[np.argmax(d)] # bottom-left has the largest difference

    return rect

def four_point_transform(image, pts):

    """wrap the area  inside 4  corner points  into a flat  rectangle."""

    rect = order_points(pts)
    tl, tr, br, bl = rect

    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl -bl)))

    dst = np.array([
        [0,0], #top-left corner
        [width -1, 0], # top-right corder
        [width -1, height -1], # bottom-right corner
        [0, height -1], #bottom-left corner
    ],dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (width, height))

def find_document_contour(edges):
    """find the contour of the document in the image, return 4 corner points"""

    kernel = cv2.getStructuringElement(cv2. MORPH_RECT, (3,3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    Contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    Contours = sorted(Contours, key=cv2.contourArea, reverse=True)[:5]

    for c in Contours:

        per = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * per, True)
        if len(approx) == 4:
            return approx
    return None

def find_by_brightness(blur, min_area_ratio=0.15):
    """Fallback: find the page as a big bright (or dark) region instead of using edges."""
    h, w = blur.shape
    total = h * w
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    best, best_area = None, 0
 
    for flag in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, mask = cv2.threshold(blur, 0, 255, flag + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area_ratio * total or area > 0.98 * total or area <= best_area:
                continue
            peri = cv2.arcLength(c, True)
            candidate = None
            for eps in (0.02, 0.03, 0.04, 0.05):
                approx = cv2.approxPolyDP(c, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    candidate = approx
                    break
            if candidate is None:
                # use the tightest rotated rectangle, only if the shape is already close to a rectangle
                box = cv2.boxPoints(cv2.minAreaRect(c))
                if area / cv2.contourArea(box.astype(np.float32)) >= 0.85:
                    candidate = box.astype(np.int32).reshape(4, 1, 2)
            if candidate is not None:
                best, best_area = candidate, area
    return best

def scan_document(img, mode="bw", debug_dir=None, hdr=False ):
    """full pipeline to scan a document form an image. Return the scanned image or None if no page is found."""
    small, ratio = tk.resize_to_height(img, DETECT_HEIGHT)

    color = tk.to_color(small)
    if hdr:
       color = tk.hdr_detect(color, 2)

    gray =tk.to_gray(color)
    blur =tk.blur_gaussian(gray, 5)

    #try the normal edge settings first, then mire sensitive ones.
    corners = None
    for low, high in ((75, 200), (50, 150), (30, 100)):
        edges = tk.detect_edges(blur, low, high)
        corners = find_document_contour(edges)
        if corners is not None:
            break

    # Last resort: look for a big brigh/dark rectangle instead of edges.
    if corners is None:
        corners = find_by_brightness(blur)
        if corners is None:
            return None

    corners = find_document_contour(edges)
    if corners is None:
        return None

    #scale corners back to the original image size, then wrap the full-res image.
    corners = corners.reshape(4,2) / ratio
    wraped = four_point_transform(img, corners)

    if mode == "bw":
        result = tk.threshold_adaptive(tk.to_gray(wraped), block_size=21, c=10)
    elif mode == "gray":
        result= tk.to_gray(wraped)
    elif mode == "color":
        result = tk.to_color(wraped)
    else:
        result = wraped

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        outline = small.copy()
        cv2.drawContours(outline, [(corners * ratio).astype(int).reshape(4,1,2)], -1,(0, 255,0),2)
        cv2.imwrite(os.path.join(debug_dir, "1_gray.png"), gray),
        cv2.imwrite(os.path.join(debug_dir, "2_blur.png"), blur),
        cv2.imwrite(os.path.join(debug_dir, "5_edges.png"), edges),
        cv2.imwrite(os.path.join(debug_dir, "6_outline.png"), outline),
        cv2.imwrite(os.path.join(debug_dir, "7_warped.png"), wraped)

        return result

def main():
        parser = argparse.ArgumentParser(description="scan a photo of a document.")
        parser.add_argument("--input", required=True, help="path to the photo")
        parser.add_argument("--output",required=True, help="where to save the scan")
        parser.add_argument("--mode", default="bw", choices=["bw", "gray", "color"],
                           help="bw = black and white scan, gray, hdr,  or color"),
        parser.add_argument("--debug", action="store_true",
                             help="save intermediate steps next to  the output")
        parser.add_argument("--hdr", action="store_true",
                             help="boost local contrast before edge detection (helps with shadows)")
        args = parser.parse_args()

        try:
            img = tk.load_image(args.input)
        except FileNotFoundError as err:
            print(err)
            sys.exit(1)
            
        debug_dir = os.path.splitext(args.output)[0] + "_steps" if args.debug else None
        result = scan_document(img, args.mode, debug_dir, args.hdr)
        
        if result is None:
          print("could not find a 4 corner document outline."
              "Try a photo with a clear background and whole page visible.")
        sys.exit(2)

        cv2.imwrite(args.output, result)
        print(f"scan saved to {args.output}")

if __name__ == "__main__":
    main()
