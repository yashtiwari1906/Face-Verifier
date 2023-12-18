import cv2
from PIL import Image 
import numpy as np 

# img = cv2.imread("/home/dracarys/Desktop/cars24/tiwari_proejcts/chehra/howard.jpg", cv2.IMREAD_COLOR)
# print(img.shape)
# cv2.imshow("image", img)

file = "/home/dracarys/Desktop/cars24/tiwari_proejcts/chehra/howard.jpg" 
img = Image.open(file).convert('RGB')
arrImg = np.array(img)

cvImg = cv2.cvtColor(arrImg, cv2.COLOR_RGB2BGR)
cv2.imshow("image", cvImg)
cv2.waitKey(0)
cv2.destroyAllWindows()