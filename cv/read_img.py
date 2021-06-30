import cv2 as cv
image = cv.imread("images/img.jpg")
#bgr_image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
cv.imwrite("output/img.jpg",rgb_image)
# cv.imshow("Image", rgb_image)
# cv.waitKey(0)
# cv.destroyAllWindows()