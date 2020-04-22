import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import matplotlib.pyplot as plt
import cv2

def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]
    return img

def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    img = draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


# def normalize_range(img):
#     '''
#     Normalizes range of input image
#     @param img: input image
#     @return img: image in range [0,255]
    
#     '''
#     img = 255*(img - np.min(img))/(np.max(img) - np.min(img))
#     img = img.astype("uint8")
#     return img


# def clean_circle(params, size=200):
#     '''
#     Creates an image of the clean circle for model training
    
#     '''
#     row, col, rad = params
#     img = np.zeros((size, size), dtype=np.float)
#     img = draw_circle(img, row, col, rad)
#     return img


def find_circle(img):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 100,
                  param1=30,
                  param2=15,
                  minRadius=0,
                  maxRadius=0)
    if circles is not None:
        params = circles[0][0]
        col, row, rad = int(params[0]), int(params[1]), int(params[2])
        return (row, col, rad)
    else:
        return (1,1,1)


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )


def main():
    results = []
    training_dataset = []
    for i in range(1000):
        params, img = noisy_circle(200, 50, 2)
        
        tmp = {}
        tmp['params'] = params
        tmp['img'] = img
        training_dataset.append(tmp)
        
        detected = find_circle(img)
        results.append(iou(params, detected))
    results = np.array(results)
    print((results > 0.7).mean())
    



