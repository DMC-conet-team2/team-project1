from dotenv import load_dotenv

import json
import openai
import os
import whisper

def analize_audio(audio_dir):
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

    # Whisper 모델 로드(tiny, base, small, medium, large 중 선택)
    model = whisper.load_model("base")

    # transcribe()는 오디오를 30초 단위로 분할하여 자동으로 텍스트로 변환
    result = model.transcribe(audio_dir, temperature=0.1, initial_prompt="이 오디오는 면접자의 답변입니다.") 
    print(f"Detected language: {result['language']}") # 감지된 언어 출력

    # 문장별 시간 출력
    segments = []
    for segment in result["segments"]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]

        load_dotenv() # .env 파일 로드
        api_key = os.getenv("OPENAI_API_KEY") # 키 읽어오기
        client = openai.OpenAI(api_key=api_key) # 클라이언트 생성

        # 부정확한 발음이나 문맥 상 자연스럽지 않은 표현을 GPT를 통해 후처리 교정
        response = client.chat.completions.create(
            model = "gpt-4.1",
            messages = [
                {"role": "system", "content": "다음 문장은 면접자의 답변 음성파일을 텍스트화 한 문장입니다. 부정확한 발음이나 문맥 상 자연스럽지 않은 내용을 교정해주세요."},
                {"role": "user", "content": text}
            ]
        )

        corrected = response.choices[0].message.content

        print(f"[{start:.1f}s --> {end:.1f}s] {corrected}")
        # 분석 된 문장 파일 저장을 위해 append
        segments.append({
            "sentence": corrected.strip(),
            "start": start,
            "end": end
        })

    # 전체 텍스트 출력
    print("\n--- 전체 텍스트 ---\n")
    print(result["text"])

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

analize_audio('./audio/ckmk_a_bm_f_e_47109.wav')
analize_audio('./audio/ckmk_a_bm_f_e_47110.wav')
analize_audio('./audio/ckmk_a_bm_f_e_47111.wav')
analize_audio('./audio/ckmk_a_bm_f_e_47112.wav')