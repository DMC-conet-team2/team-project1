from dotenv import load_dotenv

import json
import openai
import os
import torch
import whisper

def load_whisper_model(model_size = "large"):
    '''
    Whisper 모델을 return 하는 함수(model_size 다운그레이드 시 발음이 부정확할 경우 이상하게 인식되는 경우가 있어 large 고정)

    호출 예시: model = load_whisper_model()
    '''
    
    model = whisper.load_model(model_size) # Whisper 모델 로드(tiny, base, small, medium, large 중 선택)
    print(f"호출된 모델 크기: {model_size}")

    # GPU 사용 가능하면 모델을 GPU로
    if torch.cuda.is_available():
        print("✅ CUDA 사용 가능 — GPU로 모델 로드 중")
        model = model.to("cuda")
    else:
        print("⚠️ CUDA 사용 불가 — CPU로 실행됩니다")
    
    return model

def analize_audio(model, client, audio_dir):
    """
    주어진 음성 파일을 Whisper로 분석하여 언어를 감지하고, 문장별 시작/끝 시간과 텍스트를 출력하는 함수

    실행 전 세팅은 아래 링크 참조
    - https://velog.io/@strurao/Python-OpenAI-Whisper-%EC%9D%8C%EC%84%B1%EC%9D%B8%EC%8B%9D#-%EC%84%A4%EC%B9%98-%EA%B3%BC%EC%A0%95

    위 링크 세팅 중 필수 여부 정리
    - torch         | ✅ 필수           | Whisper는 PyTorch 기반, 반드시 설치
    - torchaudio    | ❌ 선택           | Whisper 기본 작동에는 필요 없지만, 오디오 전처리 등을 위해 유용
    - torchvision   | ❌ 불필요         | 이미지 처리를 위한 것으로, Whisper에서는 사용되지 않습니다.
    - ffmpeg        | ✅ 사실상 필수    | Whisper는 ffmpeg를 백엔드로 사용하여 다양한 오디오 포맷을 처리하므로 시스템에 설치되어 있어야 함

    파라미터
    - model: Whisper 모델
    - client: OpenAI API 클라이언트
    - audio_dir (str): 분석 할 음성 파일 경로

    호출 예시: analize_audio('./audio/voice-sample.mp3')
    """

    # 입력한 파일 경로 검증
    if not audio_dir:
        print('음성 파일 경로 지정이 잘못되었습니다.')
        return
    if not os.path.exists(audio_dir):
        print('존재하지 않는 음성파일입니다.\n파일 위치를 다시 확인하세요.')
        return

    # transcribe()는 오디오를 30초 단위로 분할하여 자동으로 텍스트로 변환
    result = model.transcribe(
        audio_dir,
        fp16=torch.cuda.is_available(),
        temperature=0, 
        initial_prompt="이 오디오는 면접자의 답변입니다. 이유, 동기, 배경, 성격, 경험 등의 단어가 포함될 수 있습니다."
    ) 
    print(f"모델 transcribe 완료. 감지된 언어: {result['language']}") # 감지된 언어 출력

    # 문장별 시간 출력
    segments = []
    for segment in result["segments"]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]

        # 부정확한 발음이나 문맥 상 자연스럽지 않은 표현을 GPT를 통해 후처리 교정
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 면접자의 음성 인식 결과를 맞춤법과 흐름 위주로 교정하는 교정 도우미입니다. \
                        사용자의 발화 스타일을 보존하되, 문법 오류와 어색한 표현만 자연스럽게 수정하세요. \
                        오타나 잘못 인식된 단어는 문맥상 자연스럽게 복원하세요. 예: '이후' → '이유'. \
                        문장은 하나로 출력하고, 면접 말투(자연스럽고 공손한 구어체)를 유지하세요."
                    )
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        )

        corrected = response.choices[0].message.content.strip()

        print(f"[{start:.1f}s --> {end:.1f}s] {corrected}")
        # 분석 된 문장 파일 저장을 위해 append
        segments.append({
            "original_sentence": text,
            "corrected_sentence": corrected,
            "start": start,
            "end": end
        })

    # JSON 데이터 구성
    output_data = {"interview": segments}

    # JSON 파일 저장 경로 구성
    os.makedirs("./json", exist_ok=True)  # 디렉토리 없으면 생성
    base_filename = os.path.splitext(os.path.basename(audio_dir))[0]
    json_path = f"./json/{base_filename}.json"

    # JSON 저장
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ JSON 파일이 저장되었습니다: {json_path}")

# 음성 분석기 활용 예시
load_dotenv() # .env 파일 로드
api_key = os.getenv("OPENAI_API_KEY") # 키 읽어오기
client = openai.OpenAI(api_key=api_key) # 클라이언트 생성

model = load_whisper_model()

 # 여러 파일 반복 처리
for audio_file in [
    './audio/ckmk_a_bm_f_e_47109.wav',
    './audio/ckmk_a_bm_f_e_47110.wav',
    './audio/ckmk_a_bm_f_e_47111.wav',
    './audio/ckmk_a_bm_f_e_47112.wav'
]:
    print(f"audio_file: {audio_file}")
    analize_audio(model, client, audio_file)