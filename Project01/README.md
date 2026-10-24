# Project01 - 초록 공 과녁 게임

## 프로젝트 소개
- 카메라로 초록색 탁구공을 검출하고, 화면의 과녁에 공이 닿으면 점수가 오르고 과녁이 랜덤 위치로 이동하는 게임

## 파일 구조 
```
Project01/
├── target_game.py
└── README.md
```
- `target_game.py`: 실행 파일 (공 검출과 과녁 게임이 모두 들어 있음)

## 실행 방식
1. 젯슨에서 가상환경(.venv)을 켠 상태로 `Project01` 폴더가 있는 위치로 이동함
2. 아래 명령어로 실행함
```bash
python Project01/target_game.py
```
3. 카메라 창이 뜨면 초록 공을 과녁에 대어 점수를 올림
4. 카메라 창에서 `q`를 누르면 종료됨

## 필요한 패키지
- Python 3.10 (젯슨 환경)
- opencv (GStreamer 사용 가능한 버전), numpy

## 사용한 기술
- 색공간 변환(BGR → LAB), 필터링(GaussianBlur), Color Segmentation(inRange)
- 모폴로지(OPEN), Contour Detection, 최소 외접원(minEnclosingCircle)
- 결과를 이미지 위에 그리는 overlay(원, 점수 글자)

## 참고 사항
- 카메라 연결은 젯슨용 GStreamer pipeline을 사용함 (코드 안 `cap` 줄 참고)
- 객체 추적은 칼만 필터 없이, 매 프레임마다 공을 검출하여 위치를 따라가는 방식으로 구현함
- 공을 검출하는 초록색 LAB 범위는 수업에서 제공받은 값을 사용함
- 과녁판과 공 사이의 직선거리는 피타고라스 정리를 사용하였으며 이후 직선 거리가 두 원의 반지름의 합보다 작으면 점수가 인정됨
