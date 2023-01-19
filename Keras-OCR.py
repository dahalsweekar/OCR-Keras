import cv2
import keras_ocr
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rotate
import deskew


def dskew_img(img):
    angle = deskew.determine_skew(img)
    img = rotate(img, angle, resize=True) * 255
    print(f'Angle of Rotation:{angle}')
    return img.astype(np.uint8)


def noise_removal(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 12)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def keras_pipeline():
    return keras_ocr.pipeline.Pipeline()


def get_image():
    image = [keras_ocr.tools.read(ocr_img_keras) for ocr_img_keras in
             [  # 'Resources/High_res/test1.png',
                 # 'Resources/c_backside/document.png',
                 'Resources/test5.png']]
    return image


def recognize_image(img):
    prediction_groups = pipeline.recognize(img)
    return prediction_groups


def plot_result(image, prediction_groups):
    for text, bbox in prediction_groups[0]:
        cv2.polylines(image, np.int32([bbox]), 2, (0, 255, 0), 3)
    plt.imshow(image)
    plt.show()


def get_text(prediction_groups):
    predicted = prediction_groups[0]
    for text, bbox in predicted:
        print(text)


if __name__ == '__main__':
    pipeline = keras_pipeline()
    image = get_image()
    img = noise_removal(image[0])
    img_d = dskew_img(img)
    image_o = image[0]
    image.pop()
    image.append(img_d)
    prediction_groups = recognize_image(image)
    plot_result(img_d, prediction_groups)
    get_text(prediction_groups)

    # cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
    # cv2.imshow('Image',image)
    # cv2.waitKey(0)
