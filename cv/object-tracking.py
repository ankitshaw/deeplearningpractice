import cv2 as cv
import cv2.legacy as cvl


print("--------- Select Tracker Types ---------")
print("1. Boosting")
print("2. MIL")
print("3. KCF")
print("4. TLD")
print("5. MedianFlow")
print("6. CSRT")
print("7. MOSSE")
type = int(input("Tracker no: "))

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE']
tracker_type = tracker_types[type-1]

print("Tracker Initialised: ",tracker_type)
if tracker_type == 'BOOSTING':
    tracker = cvl.TrackerBoosting_create()
if tracker_type == 'MIL':
    tracker = cv.TrackerMIL_create()
if tracker_type == 'KCF':
    tracker = cv.TrackerKCF_create()
if tracker_type == 'TLD':
    tracker = cvl.TrackerTLD_create()
if tracker_type == 'MEDIANFLOW':
    tracker = cvl.TrackerMedianFlow_create()
if tracker_type == 'CSRT':
    tracker = cv.TrackerCSRT_create()
if tracker_type == 'MOSSE':
    tracker = cvl.TrackerMOSSE_create()

########################################################


cap = cv.VideoCapture(0)

success, frame = cap.read()
bbox = cv.selectROI("Tracking...",frame, False)
tracker.init(frame, bbox)


def drawBox(img,bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 3 )
    cv.putText(img, "Tracking...", (100, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


while True:

    timer = cv.getTickCount()
    success, img = cap.read()
    success, bbox = tracker.update(img)

    if success:
        drawBox(img,bbox)
    else:
        cv.putText(img, "Lost", (100, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv.rectangle(img,(15,15),(200,90),(255,0,255),2)
    cv.putText(img, "Fps:", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2);
    cv.putText(img, "Status:", (20, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2);


    fps = cv.getTickFrequency() / (cv.getTickCount() - timer);
    if fps>60: myColor = (20,230,20)
    elif fps>20: myColor = (230,20,20)
    else: myColor = (20,20,230)
    cv.putText(img,str(int(fps)), (75, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, myColor, 2);

    cv.imshow("Tracking", img)
    if cv.waitKey(1) & 0xff == ord('q'):
       break