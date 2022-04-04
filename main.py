import cv2
import numpy as np

from faces import faces

# image path
image1 = faces(path=r"C:/exapmle.png",
               info=True)
image1w, image1h, image1_color_channels = image1.image.shape

if image1w > 560 and image1h > 700:
    resize_percentage_w = int(np.round((560 * 100) / image1w))
    resize_percentage_h = int(np.round((700 * 100) / image1h))

    image1.resize_percentage = (resize_percentage_w + resize_percentage_h) // 2
    print(image1.resize_percentage)

image1.find_faces()
image1.highlight_faces(display=True)

image1.extract_faces()
for num, face in enumerate(image1.faces):
    cv2.imshow("face" + str(num), face)


cv2.waitKey()
