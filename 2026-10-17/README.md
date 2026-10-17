# 02. Computer Vision 과제

## 파일 구조 (필수)
- **assignment02.py**: 농구공 사진에서 공 부분만 뽑아서 외곽선을 찾고, 공을 감싸는 원(최소 외접원)이랑 무게중심을 구해서 비교함
- **README.md**: 설명 파일

## 실행 방식 (필수)
저장소 맨 위 폴더(`src` 폴더가 있는 곳)에서 실행

```bash
python 2026-10-17/assignment02.py
```

그림창이 하나씩 뜨며, 창을 닫으면 다음 그림이 나옴 (총 4개)

## 필요한 패키지
- Python 3.13
- opencv-python
- numpy
- matplotlib

```bash
pip install opencv-python numpy matplotlib
```

## 참고 사항
- 이미지는 `src/images` 폴더에서 불러옴. (코드에서는 `../src/images`로 찾음)
- 실행에 필요한 앞 코드를 앞부분에 넣음
