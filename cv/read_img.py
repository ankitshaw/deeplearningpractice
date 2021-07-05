import cv2 as cv
image = cv.imread("images/faces.jpg")

#1
#image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

#2
#rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

#3
#image = image[50:1000, 500:1500]

#4
# scale_percent = 20 # percent of original size
# width = int(image.shape[1] * scale_percent / 100)
# height = int(image.shape[0] * scale_percent / 100)
# dim = (width, height)
# image = cv.resize(image, dim, interpolation = cv.INTER_AREA)

#5
# (h, w, d) = image.shape
# center = (w // 2, h // 2)
# M = cv.getRotationMatrix2D(center, 180, 1.0)
# image = cv.warpAffine(image, M, (w, h))


#6
#gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# ret, gray_image = cv.threshold(image, 150, 255, 0)
# cv.imwrite("output/threshold.jpg",gray_image)

#7
# image = cv.GaussianBlur(image, (51, 51), 0)

#8
#cv.rectangle(image, (600, 200), (1300, 800), (0, 255, 255), 10)

#9
#cv.line(image, (600, 800), (1300, 800), (0, 0, 255), 5)

#10
#cv.putText(image, "Crowd", (600, 1000),cv.FONT_HERSHEY_SIMPLEX, 5, (30, 105, 210), 20) 



#11
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor= 1.1,
    minNeighbors= 5,
    minSize=(10, 10)
)
faces_detected = format(len(faces)) + " faces detected!"
print(faces_detected)
# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)

#12
# imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(imgray, 127, 255, 0)
# contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(image, contours, -1, (0,255,0), 3)

cv.imwrite("output/text.jpg",image)


def viewImage(image, name_of_window):
    cv.namedWindow(name_of_window, cv.WINDOW_NORMAL)
    cv.imshow(name_of_window, image)
    cv.waitKey(0)
    cv.destroyAllWindows()