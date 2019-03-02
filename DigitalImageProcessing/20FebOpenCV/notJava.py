import cv2



srcImg = cv2.imread("greyscale.png",cv2.IMREAD_GRAYSCALE)

dstIMG = cv2.GaussianBlur(srcImg, (45,45),0)

cv2.imwrite("blur.png",dstIMG)
