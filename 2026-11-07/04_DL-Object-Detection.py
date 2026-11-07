# 04_DL-Object-Detection 숙제 (30, 51, 52번)
# 필수 실행 = 노트북에서 가져온 코드, TODO 답 = 숙제로 채운 코드

# 필수 실행: A에서 가져옴 - import
import numpy as np
import skimage.data


# B에서 
# TODO: 1) BBox 좌표를 (x,y,w,h)에서 (x1,y1,x2,y2)로 변환하는 함수 정의
# TODO: 2) IoU 계산 함수 정의

def convert_bbox_coord(bbox):
    x, y, w, h = bbox
    return x, y, x + w, y + h

def calculate_iou(bbox_a, bbox_b):
    xa_1, ya_1, xa_2, ya_2 = bbox_a
    xb_1, yb_1, xb_2, yb_2 = bbox_b

    intersection_x1 = max(xa_1, xb_1)
    intersection_y1 = max(ya_1, yb_1)
    intersection_x2 = min(xa_2, xb_2)
    intersection_y2 = min(ya_2, yb_2)

    w = max(0, intersection_x2 - intersection_x1)
    h = max(0, intersection_y2 - intersection_y1)

    intersection_area = w * h
    union_area = (xa_2 - xa_1) * (ya_2 - ya_1) + (xb_2 - xb_1) * (yb_2 - yb_1) - intersection_area

    # TODO: ZeroDivisionError 주의
    if union_area == 0:
        return 0.0

    return intersection_area / union_area


# 필수 실행: C에서 가져옴 - 랜덤 BBox 만드는 함수
from random import randint, gauss

def random_bbox(img, min_w=100, min_h=400, max_w=600, max_h=600):
    img_h, img_w = img.shape[:2]

    w = randint(min_w, min(max_w, img_w))
    h = randint(min_h, min(max_h, img_h))

    x1 = randint(0, img_w - w)
    y1 = randint(0, img_h - h)

    return x1, y1, x1 + w, y1 + h

def random_conf():
    return min(1.0, max(0.0, gauss(0.6, 0.1)))


# 필수 실행: C에서 가져옴 - 랜덤 BBox 20개 만들기 (그리는 부분은 뺌)
hubble = skimage.data.hubble_deep_field()

N = 20

boxes = []
scores = []

for i in range(N):
    boxes.append(random_bbox(hubble))
    scores.append(random_conf())




# C의 NMS 과정 설명(46번) 
# TODO: Confidence Thresholding 함수 정의
# TODO: Confidence Score가 특정 임계값보다 낮으면 해당 BBox 제거

def prune_low_conf(boxes, scores, conf_threshold=0.5):
    """
    Confidence Thresholding 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    conf_threshold : float
        Bounding Box를 유지하기 위한 최소 Confidence Score

    Returns
    -------
    pruned_boxes : list
        Confidence Score가 임계값 이상인 Bounding Box 목록

    pruned_scores : list
        Confidence Score가 임계값 이상인 Bounding Box의 Confidence Score 목록
    """

    pruned_boxes = []
    pruned_scores = []

    for index in range(len(boxes)):
        if scores[index] >= conf_threshold:
            pruned_boxes.append(boxes[index])
            pruned_scores.append(scores[index])

    return pruned_boxes, pruned_scores



# C의 NMS 과정 설명(46번) 
# TODO: NMS 함수 정의
# TODO: np.sort() 함수를 사용하면 1차원 배열을 정렬
# TODO: np.argsort() 함수를 사용하면 1차원 배열의 정렬된 인덱스를 반환

def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    NMS 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    iou_threshold : float
        중복 Bounding Box를 제거할 IoU 기준값

    Returns
    -------
    nms_boxes : list
        NMS 적용 후 유지된 Bounding Box 목록

    nms_scores : list
        NMS 적용 후 유지된 Bounding Box의 Confidence Score 목록
    """

    sorted_indices = list(np.argsort(scores))

    keep_indices = []

    while len(sorted_indices) > 0:
        best_index = sorted_indices.pop()
        keep_indices.append(best_index)

        remaining_indices = []

        for other_index in sorted_indices:
            iou = calculate_iou(boxes[best_index], boxes[other_index])

            if iou <= iou_threshold:
                remaining_indices.append(other_index)

        sorted_indices = remaining_indices

    nms_boxes = []
    nms_scores = []

    for index in keep_indices:
        nms_boxes.append(boxes[index])
        nms_scores.append(scores[index])

    return nms_boxes, nms_scores


# 필수 실행: C에서 가져옴 - 함수 적용
print(len(boxes), len(scores))

new_boxes, new_scores = prune_low_conf(boxes, scores)
new_boxes, new_scores = apply_nms(boxes, scores, iou_threshold=0.2)

print(len(new_boxes), len(new_scores))
