import cv2
import numpy as np
import random
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

# 초록 LAB lower/upper range
green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# 과녁 (설정 2단계)
target_center_x = 320  # 과녁 중심의 가로 위치 (화면 가로 640의 가운데)
target_center_y = 240  # 과녁 중심의 세로 위치 (화면 세로 480의 가운데)
target_radius = 40     # 과녁 원의 반지름
score = 0  # 점수 (처음엔 0점)

# 손 인식 설정 (강사님 코드)
base_option = python.BaseOptions(model_asset_path="src/models/MediaPipe/hand_landmarker.task")  # 모델 경로 지정하는 옵션
options = vision.HandLandmarkerOptions(base_options=base_option, num_hands=2)                   # 모델 경로와 최대 손 개수 지정
hand_detector = vision.HandLandmarker.create_from_options(options)                              # 해당 옵션으로 손 검출하는 객체 생성
connections = vision.HandLandmarksConnections.HAND_CONNECTIONS                                  # 각 landmark를 잇는 연결선 정보
finger_tips = (4, 8, 12, 16, 20) 

# 카메라 연결: 맥북에서는 첫 번째 줄, 젯슨에서는 두 번째 줄을 사용
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

cv2.namedWindow("Target Game", cv2.WINDOW_NORMAL)  # 크기를 바꿀 수 있는 창으로 만듦
cv2.resizeWindow("Target Game", 640, 480)  

while True:
    ret, frame = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # 지금 카메라 화면의 세로, 가로 크기를 읽어옴
    frame_height, frame_width = frame.shape[:2]

     # 프레임 내 손 탐지 (강사님 코드, 좌우 반전은 뺌)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    # 초록색 공 검출
    frame_blur = cv2.GaussianBlur(frame, (7, 7), 0)
    frame_lab = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2LAB)
    green_mask = cv2.inRange(frame_lab, green_lower, green_upper)
    green_mask_opened = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(green_mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 과녁 색: 기본은 보라색 
    target_color = (255, 0, 255) 

    # 손이 공을 만졌는지 (처음엔 안 만진 상태)
    hand_touching_ball = False

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        (ball_x, ball_y), ball_radius = cv2.minEnclosingCircle(largest_contour)

        # 검출된 공에 노란 원과 빨간 점 overlay
        ball_center = (int(ball_x), int(ball_y))
        cv2.circle(frame, ball_center, int(ball_radius), (0, 255, 255), 3)
        cv2.circle(frame, ball_center, 5, (0, 0, 255), -1)

        # 공 중심과 과녁 중심 사이의 거리 (3단계)
        horizontal_difference = ball_x - target_center_x
        vertical_difference = ball_y - target_center_y
        distance_to_target = (horizontal_difference ** 2 + vertical_difference ** 2) ** 0.5

        # 두 원의 반지름을 더한 값보다 거리가 작으면 = 두 원이 닿음 (3단계)
        if distance_to_target < target_radius + ball_radius:
            target_color = (0, 0, 255)  # 빨간색
            score = score + 1  # 닿았으니 1점 추가

            # 닿으면 과녁을 화면 안의 새로운 랜덤 위치로 옮김
            target_center_x = random.randint(target_radius, frame_width - target_radius)
            target_center_y = random.randint(target_radius, frame_height - target_radius)

    # 탐지 결과의 각 손마다 선과 점 그리기 (강사님 코드)
    for hand in result.hand_landmarks:
        h, w = frame.shape[:2]  # 프레임 높이와 너비
        points = [(int(p.x * w), int(p.y * h)) for p in hand]  # 프레임 높이와 너비 길이 기준 각 landmark 좌표

                # 손 관절 점이 공의 원 안에 들어오면 = 손이 공을 만진 것
        if len(contours) > 0:
            for point in points:
                horizontal_difference_to_ball = point[0] - ball_x
                vertical_difference_to_ball = point[1] - ball_y
                distance_to_ball = (horizontal_difference_to_ball ** 2 + vertical_difference_to_ball ** 2) ** 0.5
                if distance_to_ball < ball_radius:
                    hand_touching_ball = True

        # landmark를 연결하는 선 (skeleton) 그리기
        for c in connections:
            cv2.line(frame, points[c.start], points[c.end], (0,255,0), 2)

        # 각 관절 (landmark)에 점 그리기 (손가락 끝은 빨간 점, 그 외에는 파란 점)
        for i, point in enumerate(points):
            color = (0,0,255) if i in finger_tips else (255,0,0)
            cv2.circle(frame, point, 6 if i in finger_tips else 4, color, -1)

    # 과녁을 보라색 원으로 그림 (2단계)
    cv2.circle(frame, (target_center_x, target_center_y), target_radius, target_color, 3)

    # 점수를 화면 왼쪽 위에 글자로 씀
    cv2.putText(frame, "SCORE: " + str(score), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 손이 공을 만졌으면 글자로 알려줌
    if hand_touching_ball:
        cv2.putText(frame, "HAND TOUCH BALL", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    cv2.imshow("Target Game", frame)

hand_detector.close()
cap.release()
cv2.destroyAllWindows()
