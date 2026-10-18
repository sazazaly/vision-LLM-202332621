import cv2
import numpy as np

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

# TODO: 배운 기술(이미지 전처리, 특징 추출)을 활용하여 실시간으로 객체를 검출 /// bring by any cell!!
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# TODO: 각 프레임 이미지를 전처리 // any cell
    frame_blur = cv2.GaussianBlur(frame, (7, 7), 0)
    frame_blur_lab = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2LAB)

# TODO: Color Segmentation으로 Binary Mask 생성 //motor mask cell 
    green_mask = cv2.inRange(frame_blur_lab, green_lower, green_upper)

# TODO: Morphological Operation을 적용하여 객체 영역 추출 //motor contour cell
    green_mask_opened = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(
    green_mask_opened,        
    cv2.RETR_EXTERNAL,       
    cv2.CHAIN_APPROX_SIMPLE  
)

# TODO: 객체의 중심과 반지름을 구해 프레임 위에 overlay //1017todo cell
    if len(contours) > 0:
    # 가장 큰 contour
        largest_contour = max(contours, key=cv2.contourArea)

        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center_circle = (int(x), int(y))
        radius = int(radius)

        cv2.circle(frame, center_circle, radius, (0,255,255), 3)
        cv2.circle(frame, center_circle, 5, (0,0,255), -1)

    cv2.imshow("VideoCapture with GStreamer", frame)

cap.release()
cv2.destroyAllWindows()