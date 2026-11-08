# 공부 집중 도우미 (Final Project)

## 소개
- 책상 위 핸드폰이 10초 넘게 보이면 음성으로 경고하고, 음성으로 "공부 얼마나 했어?"라고 물으면 Gemma가 음성으로 대답하는 프로그램임
- Vision-LLM 융합 시청각 멀티모달 파이프라인
  - 객체 탐지: YOLO가 카메라 화면에서 핸드폰을 찾음
  - 이벤트 감지: 핸드폰이 10초 넘게 보이면 경고
  - 음성 입력: 마이크 녹음 후 Whisper가 글자로 바꿈
  - 음성 출력: Piper가 글자를 음성으로 읽어줌

## 실행 환경
- Jetson (카메라, 마이크, 스피커 연결)
- 사용 모델
  - YOLO11n (TensorRT int8 `.engine`)
  - Gemma 4 E2B Q4_K_M (llama.cpp)
  - Whisper.cpp base (한국어)
  - Piper ko_KR-kss-medium

## 실행 방식
```bash
python study_helper.py
```
- 모델을 불러오는 데 30~40초 걸림
- `src/`, `whisper.cpp/`, `.piper_venv/` 폴더가 있는 위치에서 실행함 (경로가 상대경로)

## 사용 방법
- 카메라 화면에 핸드폰이 10초 넘게 보이면 "핸드폰 치우고 집중하세요"라고 말해줌
- `l` 키를 누르고 5초 안에 질문하면 Gemma가 음성으로 대답함 (예: "공부 얼마나 했어?")
- `q` 키를 누르면 종료함

## 동작 순서
1. 카메라 한 프레임을 읽음
2. YOLO가 핸드폰만 찾음 (COCO 클래스 67번 cell phone)
3. 핸드폰이 연속으로 10초 넘게 보이면 경고 횟수를 1 올리고 음성으로 경고함. 핸드폰이 사라지면 시간을 처음부터 다시 셈
4. `l` 키를 누르면 녹음, Whisper로 글자 변환
5. 공부 시간, 경고 횟수, 지금 핸드폰이 있는지를 문장으로 만들어 질문과 함께 Gemma에게 줌 (Vision-to-Text)
6. Gemma의 대답을 Piper가 음성으로 읽어줌

## 어디서 가져왔는지
- 노트북 6번에서 가져옴: 경로 설정, `speech_to_text`, `text_to_speech`, YOLO와 Llama 불러오기, 반복문 뼈대, Gemma 프롬프트 틀 (Instruction, Context, Constraint, Output Format)
- 내 GStreamer 카메라 코드 그대로 사용
- 직접 쓴 부분
  - 핸드폰 기준 (`PHONE_CLASS_ID`, `PHONE_SECONDS`)
  - `study_record_to_text` (노트북 6번 A의 `detections_to_text`와 같은 방식)
  - 10초 판단과 경고 횟수 세기
  - `l` 키 질문 처리
  - 프롬프트 안의 내용

## 참고 사항
- 공부 시간은 프로그램을 켠 뒤부터 지난 시간임 (쉬는 시간은 따로 빼지 않음)
- 질문을 녹음하고 대답하는 동안 카메라 화면이 잠깐 멈춤
- 다음에 할 것: `s` 키로 쉬는 시간 빼기
