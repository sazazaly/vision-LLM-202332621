# 03. 딥러닝 기초 및 GPU 가속 숙제

## 소개
- `03_DL-and-GPU.ipynb`의 TODO 숙제를 `03_DL-and-GPU.py` 파일 하나로 정리함

## 실행 방식
```bash
python 03_DL-and-GPU.py
```
- `src/images/seagull.jpg`, `src/datasets/`가 있는 노트북과 같은 폴더 구조에서 실행함
- NVIDIA GPU(CUDA)가 있고 CuPy, CUDA용 PyTorch가 설치된 환경(젯슨 등)에서 실행함
- 그림 창이 뜨면 닫아야 다음 코드가 실행됨
- MNIST CPU 학습이 두 번 들어 있어서 실행 시간이 긴 편임

## 채운 TODO (03만, 04는 다음주 숙제)
- F. MNIST: Loss, Accuracy 그래프 / 학습된 모델의 Feature Map 확인
- G. CuPy 잔상 효과: CPU 함수, GPU 함수, CuPy 배열로 변환, GPU 결과를 CPU로 복사
- G. CuPy Convolution: `convolution2d_cp` 함수, CPU 처리, GPU 처리, 결과 출력
- I. `main_gpu()`: `main_cpu()`를 복사해서 GPU용으로 수정
- J. CIFAR-10: GPU 학습 반복문

## 잔상 효과
- 이미지를 오른쪽으로 조금씩 밀어서 여러 장 만들고, 전부 겹쳐서 더함
- 많이 민 이미지일수록 약하게 더해서, 오른쪽으로 갈수록 흐려지는 꼬리가 생김
- CPU 방식: 픽셀을 하나씩 돌면서 계산함. 픽셀이 너무 많아서 오래 걸림
- GPU 방식: 미는 횟수(32번)만 반복하고, 이미지 전체는 한 번에 계산함. 그래서 훨씬 빠름
- GPU는 계산을 시키고 바로 다음 줄로 넘어가기 때문에, 시간을 잴 때는 GPU가 끝날 때까지 기다리게 함

## Convolution
- 원래 방식(CPU)은 이미지 위에서 3×3 칸을 하나씩 옮기면서 곱하고 더함. 옮길 곳이 많아서 느림
- GPU 방식은 커널의 숫자 9개를 하나씩 꺼내서, 이미지 전체에 그 숫자를 곱해서 더함. 9번만 반복하면 끝남
- 계산 순서만 다르고 결과는 같음. 결과 이미지를 CPU와 GPU로 비교해서 같은 것을 확인함
- 세로선, 가로선 필터를 적용해서 이미지의 윤곽선을 찾음

## 참고 사항
- 결과 시간은 실행 환경에 따라 다름 (아래는 실행 후 채움)
  - 잔상 효과 CPU / GPU:
  - Convolution CPU / GPU:
  - MNIST 학습 CPU / GPU:
- GPU 워밍업은 잔상 효과와 CIFAR-10 칸에서만 사용함
- 맥북에서 실행이 불가해서 실습실 컴퓨터로 옮겨서 테스트 했는데 어려웠음 잘 돌아가는지 확인이 힘듦
