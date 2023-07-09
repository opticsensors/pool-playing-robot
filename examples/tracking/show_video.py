import cv2

cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
#cap.set(cv2.CAP_PROP_BRIGHTNESS, 120)
#cap.set(cv2.CAP_PROP_EXPOSURE, -8)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(3,640)
cap.set(4,480)


while(True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    c = cv2.waitKey(1)
    if c & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()