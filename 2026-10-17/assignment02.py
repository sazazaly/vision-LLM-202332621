import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# src찾기
image_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "images")

# ===== 앞 준비: 농구공 이미지 불러오기 + LAB CLAHE 마스크 (TODO셀 실행하기 위해) =====
lower_basketball_lab = np.array([70, 130, 70])
upper_basketball_lab = np.array([180, 230, 185])

basketball = cv2.imread(os.path.join(image_folder, "basketball_crop.jpg"))

basketball_lab = cv2.cvtColor(basketball, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(basketball_lab)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l_clahe_enhanced = clahe.apply(l)
basketball_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))

mask_enhanced = cv2.inRange(basketball_enhanced_lab, lower_basketball_lab, upper_basketball_lab)
basketball_enhanced_rgb = cv2.cvtColor(basketball_enhanced_lab, cv2.COLOR_LAB2RGB)

# ===== 과제 셀 =====


# TODO: Morphological operation을 적용하여 최종 binary mask 생성
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
ball_mask_opened = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel)

# TODO: Binary mask로 외곽선 검출
contours, _ = cv2.findContours(ball_mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# TODO: 최종 농구공 영역 및 외곽선 선택
largest_contour = max(contours, key=cv2.contourArea)

# TODO: 최종 binary mask
final_ball_mask = ball_mask_opened

# Contour 및 외접원을 그릴 원본 이미지 copy
final_img_enhanced = basketball_enhanced_rgb.copy()

if len(contours) > 0:
    # 가장 큰 contour
    largest_contour = max(contours, key=cv2.contourArea)

    # 가장 큰 contour만 채운 binary mask 생성
    largest_mask = np.zeros_like(final_ball_mask)
    cv2.drawContours(
        largest_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # 가장 큰 contour만 채운 binary mask를 반투명 빨간색으로 overlay
    mask_color = np.zeros_like(final_img_enhanced)
    mask_color[:, :, 0] = largest_mask
    alpha = 0.15
    final_img_enhanced = cv2.addWeighted(final_img_enhanced, 1.0, mask_color, alpha, 0)

    # Contour 그리기 (빨간색 외곽선)
    cv2.drawContours(final_img_enhanced, [largest_contour], -1, (255,0,0), 2)

    # 최소 외접원 중심 및 반지름
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center_circle = (int(x), int(y))
    radius = int(radius)

    # 파란색 원
    cv2.circle(final_img_enhanced, center_circle, radius, (0,0,255), 3)
    cv2.circle(final_img_enhanced, center_circle, 5, (0,0,255), -1)

plt.figure(figsize=(16,8))
plt.subplot(1,2,1), plt.imshow(basketball_enhanced_rgb)
plt.subplot(1,2,2), plt.imshow(final_img_enhanced)

for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()

# TODO: 최종 binary mask
final_ball_mask = ball_mask_opened

# Contour 및 무게중심 기준 원을 그릴 원본 이미지 copy
final_img_enhanced = basketball_enhanced_rgb.copy()

if len(contours) > 0:
    # 가장 큰 contour
    largest_contour = max(contours, key=cv2.contourArea)

    # 가장 큰 contour만 채운 binary mask 생성
    largest_mask = np.zeros_like(final_ball_mask)
    cv2.drawContours(
        largest_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # 가장 큰 contour만 채운 binary mask를 반투명 빨간색으로 overlay
    mask_color = np.zeros_like(final_img_enhanced)
    mask_color[:, :, 0] = largest_mask
    alpha = 0.15
    final_img_enhanced = cv2.addWeighted(final_img_enhanced, 1.0, mask_color, alpha, 0)

    # Contour 그리기 (빨간색 외곽선)
    cv2.drawContours(final_img_enhanced, [largest_contour], -1, (255,0,0), 2)

    # Moments를 통해 무게중심 (Centroid) 산출
    M = cv2.moments(largest_contour)

    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    centroid = (cX, cY)

    # 노란색 무게중심
    cv2.circle(final_img_enhanced, centroid, 5, (255,255,0), -1)

plt.figure(figsize=(16,8))
plt.subplot(1,2,1), plt.imshow(basketball_enhanced_rgb)
plt.subplot(1,2,2), plt.imshow(final_img_enhanced)

for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()

# TODO: 최종 binary mask
final_ball_mask = ball_mask_opened

# Contour 및 무게중심 기준 원을 그릴 원본 이미지 copy
final_img_enhanced = basketball_enhanced_rgb.copy()

if len(contours) > 0:
    # 가장 큰 contour
    largest_contour = max(contours, key=cv2.contourArea)

    # 가장 큰 contour만 채운 binary mask 생성
    largest_mask = np.zeros_like(final_ball_mask)
    cv2.drawContours(
        largest_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # 가장 큰 contour만 채운 binary mask를 반투명 빨간색으로 overlay
    mask_color = np.zeros_like(final_img_enhanced)
    mask_color[:, :, 0] = largest_mask
    alpha = 0.15
    final_img_enhanced = cv2.addWeighted(final_img_enhanced, 1.0, mask_color, alpha, 0)

    # Contour 그리기 (빨간색 외곽선)
    cv2.drawContours(final_img_enhanced, [largest_contour], -1, (255,0,0), 2)

    # 최소 외접원 중심 및 반지름
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center_circle = (int(x), int(y))
    radius = int(radius)

    # Moments를 통해 무게중심 (Centroid) 산출
    M = cv2.moments(largest_contour)

    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = center_circle

    centroid = (cX, cY)

    # 노란색 무게중심 기반 원
    cv2.circle(final_img_enhanced, centroid, radius, (255,255,0), 3)
    cv2.circle(final_img_enhanced, centroid, 5, (255,255,0), -1)

plt.figure(figsize=(16,8))
plt.subplot(1,2,1), plt.imshow(basketball_enhanced_rgb)
plt.subplot(1,2,2), plt.imshow(final_img_enhanced)

for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()

# TODO: 최종 binary mask
final_ball_mask = ball_mask_opened

# Contour 및 무게중심 기준 원을 그릴 원본 이미지 copy
final_img_enhanced = basketball_enhanced_rgb.copy()

if len(contours) > 0:
    # 가장 큰 contour
    largest_contour = max(contours, key=cv2.contourArea)

    # 가장 큰 contour만 채운 binary mask 생성
    largest_mask = np.zeros_like(final_ball_mask)
    cv2.drawContours(
        largest_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # 가장 큰 contour만 채운 binary mask를 반투명 빨간색으로 overlay
    mask_color = np.zeros_like(final_img_enhanced)
    mask_color[:, :, 0] = largest_mask
    alpha = 0.15
    final_img_enhanced = cv2.addWeighted(final_img_enhanced, 1.0, mask_color, alpha, 0)

    # Contour 그리기 (빨간색 외곽선)
    cv2.drawContours(final_img_enhanced, [largest_contour], -1, (255,0,0), 2)

    # 최소 외접원 중심 및 반지름
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center_circle = (int(x), int(y))
    radius = int(radius)

    # 파란색 최소 외접원
    cv2.circle(final_img_enhanced, center_circle, radius, (0,0,255), 3)
    cv2.circle(final_img_enhanced, center_circle, 5, (0,0,255), -1)

    # Moments를 통해 무게중심 (Centroid) 산출
    M = cv2.moments(largest_contour)

    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = center_circle

    centroid = (cX, cY)

    # 노란색 무게중심 기반 원
    cv2.circle(final_img_enhanced, centroid, radius, (255,255,0), 3)
    cv2.circle(final_img_enhanced, centroid, 5, (255,255,0), -1)

plt.figure(figsize=(16,8))
plt.subplot(1,2,1), plt.imshow(basketball_enhanced_rgb)
plt.subplot(1,2,2), plt.imshow(final_img_enhanced)

for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()
