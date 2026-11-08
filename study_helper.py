# 공부 집중 도우미
# 책상 위 핸드폰이 10초 넘게 보이면 음성으로 알려주고, l 키를 누르고 "공부 얼마나 했어?"라고 물으면 Gemma가 음성으로 대답함

import cv2
import time
import subprocess

from ultralytics import YOLO
from llama_cpp import Llama


# 모델, 파일 경로 (노트북 6번 A, C 파트에서 가져옴)
YOLO_MODEL_PATH = "src/models/YOLO/yolo11n_int8.engine"
GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"

CONTEXT_WINDOW = 2048
MAX_TOKENS = 150

# 음성 (노트북 6번 C 파트에서 가져옴, 마이크 번호는 arecord -l 로 확인해서 바꿈)
MIC_DEVICE = "plughw:2,0"
AUDIO_FILE = "src/audio/input.wav"
WHISPER_PATH = "whisper.cpp/build-cpu/bin/whisper-cli"
WHISPER_MODEL = "whisper.cpp/models/ggml-base.bin"
RECORD_SECONDS = 5

PIPER_PYTHON = ".piper_venv/bin/python"
PIPER_MODEL = "src/models/Piper/ko_KR-kss-medium.onnx"
OUTPUT_FILE = "src/audio/response.wav"

# 핸드폰 경고 기준 (직접 정함)
PHONE_CLASS_ID = 67      # YOLO(COCO)에서 cell phone 번호
PHONE_SECONDS = 10       # 핸드폰이 이 시간 넘게 보이면 경고


# 노트북 6번 C 파트에서 가져옴
def speech_to_text():
    subprocess.run(
        [
            "pasuspender", "--",
            "arecord",
            "-D", MIC_DEVICE,
            "-f", "S16_LE",
            "-r", "16000",
            "-c", "1",
            "-d", str(RECORD_SECONDS),
            AUDIO_FILE,
        ],
        check=True,
    )

    result = subprocess.run(
        [
            WHISPER_PATH,
            "-m", WHISPER_MODEL,
            "-f", AUDIO_FILE,
            "-l", "ko",
            "--no-timestamps",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    return result.stdout.strip()


# 노트북 6번 C 파트에서 가져옴
def text_to_speech(text):
    subprocess.run(
        [
            PIPER_PYTHON,
            "-m",
            "piper",

            "-m",
            PIPER_MODEL,

            "-f",
            OUTPUT_FILE,

            "--",
            text,
        ],
        check=True,
    )

    subprocess.run(
        [
            "aplay",
            OUTPUT_FILE,
        ],
        check=True,
    )


# 공부 기록을 문장으로 만듦 (노트북 6번 A 파트의 detections_to_text와 같은 방식)
def study_record_to_text(study_seconds, warning_count, phone_visible):
    minutes = int(study_seconds // 60)
    seconds = int(study_seconds % 60)

    if phone_visible:
        phone_text = "있습니다"
    else:
        phone_text = "없습니다"

    sentences = [
        f"공부한 시간은 {minutes}분 {seconds}초입니다.",
        f"핸드폰 경고를 받은 횟수는 {warning_count}번입니다.",
        f"지금 책상 위에 핸드폰이 {phone_text}.",
    ]

    return "\n".join(sentences)


yolo = YOLO(YOLO_MODEL_PATH)

# 노트북 6번 A 파트의 Llama 설정과 같음
llm = Llama(
    model_path=GEMMA_MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=CONTEXT_WINDOW,
    n_batch=32,
    n_ubatch=32,
    verbose=False,
)

# 카메라 파이프라인 (내가 쓰던 GStreamer 코드 그대로)
pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)


if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()


print("q : 종료")
print("l : 음성으로 질문하기 (예: 공부 얼마나 했어?)")


start_time = time.time()
warning_count = 0
phone_start_time = None


while True:
    ret, frame = cap.read()
    key = cv2.waitKey(1) & 0xFF

    if not ret:
        break
    if key == ord("q"):
        break


    # 1. 현재 프레임에서 핸드폰만 YOLO 탐지
    results = yolo.predict(
        source=frame,
        conf=0.25,
        iou=0.5,
        classes=[PHONE_CLASS_ID],
        verbose=False,
    )

    result = results[0]
    phone_visible = len(result.boxes) > 0


    # 2. YOLO 결과 화면 출력
    output_frame = result.plot()
    cv2.imshow("Study Helper", output_frame)


    # 3. 핸드폰이 10초 넘게 보이면 음성으로 경고 (이벤트)
    current_time = time.time()

    if phone_visible:
        if phone_start_time is None:
            phone_start_time = current_time

        phone_seen_seconds = current_time - phone_start_time

        if phone_seen_seconds >= PHONE_SECONDS:
            warning_count = warning_count + 1
            phone_start_time = None

            print("[경고] 핸드폰 치우고 집중하세요")
            text_to_speech("핸드폰 치우고 집중하세요")
    else:
        phone_start_time = None


    # 4. l 키를 누르면 음성 질문 -> Gemma -> 음성 대답
    if key == ord("l"):
        print("질문을 말씀해 주세요.")
        question = speech_to_text()
        print("[질문]", question)

        if question == "":
            print("음성이 인식되지 않았습니다.")
            continue

        study_seconds = time.time() - start_time
        vision_text = study_record_to_text(study_seconds, warning_count, phone_visible)

        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": f"""
                                Instruction:
                                주어진 공부 기록을 바탕으로 사용자의 질문에 답하시오.

                                Context:
                                {vision_text}

                                Constraint:
                                공부 기록에 없는 내용을 지어내지 마시오.

                                Output Format:
                                한국어 두 문장 이내.
                               """
                },
                {
                    "role": "user",
                    "content": question,
                }
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.7,
        )

        answer = response["choices"][0]["message"]["content"].strip()

        print("\n[Gemma]")
        print(answer)

        text_to_speech(answer)


cap.release()
cv2.destroyAllWindows()
