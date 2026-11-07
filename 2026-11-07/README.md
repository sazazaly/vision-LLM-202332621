# 04. 객체 검출 (IoU, NMS) 숙제

## 소개
- `04_DL-Object-Detection-2.ipynb`의 TODO 중 30번, 51번, 52번 칸
- 노트북에서 가져온 코드와 직접 채운 코드를 나눠서 표시함
  - `필수 실행`: 노트북에서 가져온 코드. 어느 부분에서 가져왔는지 위에 표시
  - `TODO 답`: 숙제로 채운 코드

## 실행 방식
```bash
python 04_DL-Object-Detection.py
```
- numpy, scikit-image가 설치된 환경에서 실행함
- 이미지 파일은 필요 없음 (skimage 기본 이미지 사용)

## 채운 TODO
- 30번 (B. IoU)
  - `convert_bbox_coord`: (x, y, w, h)를 (x1, y1, x2, y2)로 바꿈
  - `calculate_iou`: 겹친 넓이 / 합친 넓이. 합친 넓이가 0이면 0을 돌려줌 (ZeroDivisionError 방지)
- 51번 (C. NMS) `prune_low_conf`: confidence score가 기준값보다 낮은 박스를 지움
- 52번 (C. NMS) `apply_nms`: score 높은 박스부터 남기고, 그 박스와 IoU가 기준값보다 큰 박스는 지움. 박스가 없어질 때까지 반복함

## 어디서 가져왔는지
- A에서 가져옴: import
- C에서 가져옴: `random_bbox`, `random_conf`, 랜덤 박스 20개 만들기, 함수 적용
- TODO 답은 노트북 설명에서 가져옴
  - 30번: B의 설명 칸(29번). 교집합 좌표, 너비 w, 높이 h, 합집합 계산 순서 그대로
  - 51번: C의 NMS 과정 1번 (신뢰도 낮은 BBox 제거)
  - 52번: C의 NMS 과정 2~6번, `keep_indices` 이름은 58번 `apply_nms_cv2`에서 가져옴

## 참고 사항
- 18, 19, 24번은 이미 되어 있던 칸이라 이번에 제출하는 파일에는 넣지 않음
- 사진 보여주는 코드는 TODO에 해당하지 않는 것 같아서 뺌