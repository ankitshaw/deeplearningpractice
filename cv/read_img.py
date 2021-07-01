import cv2 as cv
image = cv.imread("images/img.jpg")

#1
#bgr_image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

#2
#rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

#3
#cropped = image[10:500, 500:2000]

#4
# scale_percent = 20 # percent of original size
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)
# resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)

#5
# (h, w, d) = image.shape
# center = (w // 2, h // 2)
# M = cv.getRotationMatrix2D(center, 180, 1.0)
# rotated = cv.warpAffine(image, M, (w, h))


#6
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
ret, threshold_image = cv.threshold(image, 150, 255, 0)
cv.imwrite("output/img.jpg",gray_image)
cv.imwrite("output/img2.jpg",threshold_image)


def viewImage(image, name_of_window):
    cv.namedWindow(name_of_window, cv.WINDOW_NORMAL)
    cv.imshow(name_of_window, image)
    cv.waitKey(0)
    cv.destroyAllWindows()