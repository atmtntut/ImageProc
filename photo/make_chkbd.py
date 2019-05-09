import cv2 
import numpy as np

w = 5
h = 8
step = 40
width = w*step
height = h*step

image = np.zeros((height, width),dtype = np.uint8)
print(image.shape)

for i in range(width):
    for j in range(height):
        if((int)(i/step) + (int)(j/step))%2:
            image[j,i] = 255;
cv2.imwrite("chess.jpg",image)
cv2.imshow("chess",image)
cv2.waitKey(0)
