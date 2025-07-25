from audio_utils import extract_audio_features
from sklearn.metrics.pairwise import cosine_similarity
from utils import OpenAIClient, OpenAIEmbedding, dump_json, read_json
from queue import Queue
from vosk import Model as VoskModel, KaldiRecognizer

import numpy as np
import json
import os
import pyloudnorm as pyln
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import tempfile
import threading
import time
import torch
import whisper

"""
### 음성 분석의 정상범위 평가 기준
- 소리 크기 : –23 LUFS ± 1.0
- 말 빠르기 : 129.7 ± 25.9 WPM WPM(≈ 1.73 ~ 2.59 WPS)

### 평가 기준 근거
- 소리 크기
    - EBU(European Broadcasting Union)의 라이브 방송 표준 범위
    - 참고 근거 : https://tech.ebu.ch/files/live/sites/tech/files/shared/r/r128.pdf
- 말 빠르기
    - (자연스러운 자발적 발화에서의 평균 WPM 추정 범위) / 60 => WPS(Words Per Second)
    - 참고 근거 : https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002099342
"""

class RealTimeAudioAnalyzerWhisper:
    def __init__(self):
        self.whisper_model = self.load_whisper_model()

        self.q = Queue()
        self.sample_rate = 16000
        self.chunk_duration = 5 # 초 단위
        self.openai_client = OpenAIClient()
        self.listening = False
        self.output_sentences = []

    def load_whisper_model(self, model_size = "large"):
        '''
        Whisper 모델을 return 하는 함수(model_size 다운그레이드 시 발음이 부정확할 경우 이상하게 인식되는 경우가 있어 large 고정)

        파라미터
        - model_size: Whisper 모델 크기(tiny, base, small, medium, large 중 선택)

        호출 예시: model = load_whisper_model()
        '''
        
        model = whisper.load_model(model_size) # Whisper 모델 로드
        print(f"호출된 모델 크기: {model_size}")

        # GPU 사용 가능하면 모델을 GPU로
        if torch.cuda.is_available():
            print("✅ CUDA 사용 가능 — GPU로 모델 로드 중")
            model = model.to("cuda")
        else:
            print("⚠️ CUDA 사용 불가 — CPU로 실행됩니다")
        
        return model
    
    def audio_callback(self, indata, frames, time_info, status):
        self.q.put(indata.copy())

    def start_stream(self):
        self.listening = True
        threading.Thread(target=self._record_audio, daemon=True).start()
        threading.Thread(target=self._transcribe_loop, daemon=True).start()
        print("🎧 실시간 음성 분석을 시작합니다. (Ctrl+C로 종료)")

    def _record_audio(self):
        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=self.audio_callback):
            while self.listening:
                time.sleep(0.1)

    def _transcribe_loop(self):
        while self.listening:
            audio_chunk = self.q.get()
            audio_chunk = np.squeeze(audio_chunk)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                filepath = f.name
                sf.write(f.name, audio_chunk, self.sample_rate)

            # Whisper 처리
            result = self.whisper_model.transcribe(
                filepath,
                fp16=torch.cuda.is_available(),
                temperature=0.2,
                initial_prompt="이 오디오는 면접자의 답변입니다."
            )

            text = result["text"].strip()
            duration = self.chunk_duration
            wps = len(text.split()) / duration if duration > 0 else 0

            # 교정
            corrected = self.openai_client.create_response(
                system_content="면접자의 말투를 유지하며 맞춤법 위주로 교정하며 교정한 내용만 출력하세요.",
                user_content=text
            )

            # 교정 후 코사인 유사도 저장
            original_embedding = OpenAIEmbedding(text)
            corrected_embedding = OpenAIEmbedding(corrected)

            similarity = cosine_similarity([original_embedding], [corrected_embedding])[0][0]

            # 감정 추정
            features = extract_audio_features(filepath, 0, self.chunk_duration)
            emotion_prompt = f"""
            문장: "{text}"
            평균 pitch: {features["pitch_mean"]:.1f}Hz, pitch 변화량: {features["pitch_std"]:.2f}
            평균 에너지: {features["energy_mean"]:.5f}, 에너지 변화량: {features["energy_std"]:.5f}
            말 빠르기(WPS): {wps:.2f}
            """
            emotion = self.openai_client.create_response(
                system_content="이 문장의 감정을 하나의 단어로 추론하세요.",
                user_content=emotion_prompt
            )

            print(f"[{emotion}] {corrected}")

            # 저장
            self.output_sentences.append({
                "original_sentence": text,
                "corrected_sentence": corrected,
                "cosine_similarity": similarity,
                "emotion": emotion,
                "wps": wps
            })

            os.remove(filepath)  # 임시 파일 삭제

    def stop(self):
        self.listening = False
        dump_json(filename="realtime_output", json_data={"interview": self.output_sentences})
        print("\n🛑 분석 종료. JSON 파일 저장 완료.")

class RealTimeAudioAnalyzerVosk:
    def __init__(self):
        self.vosk_model = VoskModel("vosk-model-small-ko-0.22")  # 한국어 모델 경로
        self.recognizer = KaldiRecognizer(self.vosk_model, 16000)

        self.q = Queue()
        self.sample_rate = 16000
        self.chunk_duration = 3  # 1~3번: shorter for quicker response
        self.openai_client = OpenAIClient()
        self.listening = False
        self.output_sentences = []

    def audio_callback(self, indata, frames, time_info, status):
        self.q.put(indata.copy())

    def start_stream(self):
        self.listening = True
        threading.Thread(target=self._record_audio, daemon=True).start()
        threading.Thread(target=self._transcribe_loop, daemon=True).start()
        print("🎧 실시간 음성 분석을 시작합니다. (Ctrl+C로 종료)")

    def _record_audio(self):
        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=self.audio_callback):
            while self.listening:
                time.sleep(0.1)

    def _transcribe_loop(self):
        buffer = b''

        while self.listening:
            chunk = self.q.get()
            buffer += chunk.tobytes()

            if self.recognizer.AcceptWaveform(buffer):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").strip()

                if not text:
                    buffer = b''  # ⚠️ 이 위치에서 초기화
                    continue

                # save audio to temp file BEFORE buffer clear
                audio_array = np.frombuffer(buffer, dtype=np.int16)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    filepath = f.name
                    sf.write(filepath, audio_array, self.sample_rate)

                buffer = b''  # ✅ buffer 초기화 위치 이쪽이 맞음

                # 분석
                duration = self.chunk_duration
                wps = len(text.split()) / duration if duration > 0 else 0

                corrected = self.openai_client.create_response(
                    system_content="면접자의 말투를 유지하며 맞춤법 위주로 교정하며 교정한 내용만 출력하세요.",
                    user_content=text
                )

                original_embedding = OpenAIEmbedding(text)
                corrected_embedding = OpenAIEmbedding(corrected)
                similarity = cosine_similarity([original_embedding], [corrected_embedding])[0][0]

                features = extract_audio_features(filepath, 0, self.chunk_duration)
                emotion_prompt = f"""
                문장: "{text}"
                평균 pitch: {features["pitch_mean"]:.1f}Hz, pitch 변화량: {features["pitch_std"]:.2f}
                평균 에너지: {features["energy_mean"]:.5f}, 에너지 변화량: {features["energy_std"]:.5f}
                말 빠르기(WPS): {wps:.2f}
                """

                emotion = self.openai_client.create_response(
                    system_content="이 문장의 감정을 하나의 단어로 추론하세요.",
                    user_content=emotion_prompt
                )

                print(f"[{emotion}] {corrected}")

                self.output_sentences.append({
                    "original_sentence": text,
                    "corrected_sentence": corrected,
                    "cosine_similarity": similarity,
                    "emotion": emotion,
                    "wps": wps
                })

                os.remove(filepath)

    def stop(self):
        self.listening = False
        dump_json(filename="realtime_output", json_data={"interview": self.output_sentences})
        print("\n🛑 분석 종료. JSON 파일 저장 완료.")

class RealTimeAudioAnalyzerGoogleSTT:
    def __init__(self, use_google_cloud=True):
        self.q = Queue()
        self.sample_rate = 16000
        self.chunk_duration = 5  # 초 단위
        self.listening = False
        self.output_sentences = []
        self.openai_client = OpenAIClient()
        self.use_google_cloud = use_google_cloud  # True면 GCP, False면 Google Web Speech API 사용
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300  # 음성 인식 민감도 설정

    def audio_callback(self, indata, frames, time_info, status):
        self.q.put(indata.copy())

    def start_stream(self):
        self.listening = True
        threading.Thread(target=self._record_audio, daemon=True).start()
        threading.Thread(target=self._transcribe_loop, daemon=True).start()
        print("🎧 Google STT 기반 실시간 음성 분석 시작 (Ctrl+C로 종료)")

    def _record_audio(self):
        import sounddevice as sd
        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=self.audio_callback):
            while self.listening:
                time.sleep(0.1)

    def _transcribe_loop(self):
        while self.listening:
            audio_chunk = self.q.get()
            audio_chunk = np.squeeze(audio_chunk)

            # WAV로 저장
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                filepath = f.name
                sf.write(filepath, audio_chunk, self.sample_rate)

            # Google STT로 음성 인식
            with sr.AudioFile(filepath) as source:
                audio = self.recognizer.record(source)
                try:
                    if self.use_google_cloud:
                        text = self.recognizer.recognize_google_cloud(audio, language="ko-KR")
                    else:
                        text = self.recognizer.recognize_google(audio, language="ko-KR")
                except sr.UnknownValueError:
                    print("❗ 음성을 인식할 수 없습니다.")
                    os.remove(filepath)
                    continue
                except sr.RequestError as e:
                    print(f"❗ 요청 실패: {e}")
                    os.remove(filepath)
                    continue

            text = text.strip()
            duration = self.chunk_duration
            wps = len(text.split()) / duration if duration > 0 else 0

            # 교정
            corrected = self.openai_client.create_response(
                system_content="면접자의 말투를 유지하며 맞춤법 위주로 교정하며 교정한 내용만 출력하세요.",
                user_content=text
            )

            original_embedding = OpenAIEmbedding(text)
            corrected_embedding = OpenAIEmbedding(corrected)
            similarity = cosine_similarity([original_embedding], [corrected_embedding])[0][0]

            # 감정 추정
            features = extract_audio_features(filepath, 0, self.chunk_duration)
            emotion_prompt = f"""
            문장: "{text}"
            평균 pitch: {features["pitch_mean"]:.1f}Hz, pitch 변화량: {features["pitch_std"]:.2f}
            평균 에너지: {features["energy_mean"]:.5f}, 에너지 변화량: {features["energy_std"]:.5f}
            말 빠르기(WPS): {wps:.2f}
            """
            emotion = self.openai_client.create_response(
                system_content="이 문장의 감정을 하나의 단어로 추론하세요.",
                user_content=emotion_prompt
            )

            print(f"[{emotion}] {corrected}")

            self.output_sentences.append({
                "original_sentence": text,
                "corrected_sentence": corrected,
                "cosine_similarity": similarity,
                "emotion": emotion,
                "wps": wps
            })

            os.remove(filepath)

    def stop(self):
        self.listening = False
        dump_json(filename="realtime_output", json_data={"interview": self.output_sentences})
        print("\n🛑 분석 종료. JSON 파일 저장 완료.")

class StaticAudioAnalyzer:
    def __init__(self):
        self.whisper_model = self.load_whisper_model()
        self.openai_client = OpenAIClient()

    def load_whisper_model(self, model_size = "large"):
        '''
        Whisper 모델을 return 하는 함수(model_size 다운그레이드 시 발음이 부정확할 경우 이상하게 인식되는 경우가 있어 large 고정)

        파라미터
        - model_size: Whisper 모델 크기(tiny, base, small, medium, large 중 선택)

        호출 예시: model = load_whisper_model()
        '''
        
        model = whisper.load_model(model_size) # Whisper 모델 로드
        print(f"호출된 모델 크기: {model_size}")

        # GPU 사용 가능하면 모델을 GPU로
        if torch.cuda.is_available():
            print("✅ CUDA 사용 가능 — GPU로 모델 로드 중")
            model = model.to("cuda")
        else:
            print("⚠️ CUDA 사용 불가 — CPU로 실행됩니다")
        
        return model

    def analize_audio(self, audio_dir):
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
        - client: OpenAI API 클라이언트
        - audio_dir (str): 분석 할 음성 파일 경로

        호출 예시: analize_audio(client, './audio/voice-sample.mp3')
        """

        # 입력한 파일 경로 검증
        if not audio_dir:
            print('음성 파일 경로 지정이 잘못되었습니다.')
            return
        if not os.path.exists(audio_dir):
            print('존재하지 않는 음성파일입니다.\n파일 위치를 다시 확인하세요.')
            return

        # 평균 소리 크기 계산
        data, rate = sf.read(audio_dir)
        meter = pyln.Meter(rate) 
        loudness = meter.integrated_loudness(data)
        print(f"Integrated Loudness: {loudness:.2f} LUFS")

        # transcribe()는 오디오를 30초 단위로 분할하여 자동으로 텍스트로 변환
        result = self.whisper_model.transcribe(
            audio_dir,
            fp16=torch.cuda.is_available(),
            temperature=0.2, 
            initial_prompt="이 오디오는 면접자의 답변이며 이유, 동기, 배경, 성격, 경험 등의 단어가 포함될 수 있습니다.",
            beam_size = 5 # beam search 사용 (더 많은 후보 탐색 → 정확도 향상)
        ) 
        print(f"모델 transcribe 완료. 감지된 언어: {result['language']}") # 감지된 언어 출력

        # 문장별 시간 출력
        sentences = []
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()

            # 말 빠르기 계산(초 당 몇 개의 단어를 말 했는지)
            duration = end - start
            word_cnt = len(text.split()) # Whisper로 읽어온 원본 텍스트 기준
            wps = word_cnt / duration if duration > 0 else 0 # words per second

            # 1. 부정확한 발음이나 문맥 상 자연스럽지 않은 표현을 GPT를 통해 후처리 교정
            corrected = self.openai_client.create_response(
                system_content = (
                    "당신은 면접자의 음성 인식 결과를 맞춤법과 흐름 위주로 교정하는 교정 도우미입니다. \
                    사용자의 어투 및 어미를 보존하고, 문법 오류와 앞 뒤 단어와 이어지지 않는 어색한 표현만 자연스럽게 수정하세요. 예: '교본 근무' → '교번근무' 혹은 '교대근무' \
                    오타나 잘못 인식된 단어는 문맥상 자연스럽게 복원하세요. 예: '이후' → '이유'. \
                    답변은 문장 하나로 출력하되, 면접 말투(자연스럽고 공손한 구어체)를 유지하며 문장 구조를 바꾸지 말고 \
                    표현만 위에 제시한대로 다듬으세요."
                ),
                user_content = text
            )

            # 교정 후 코사인 유사도 저장
            original_embedding = OpenAIEmbedding(text)
            corrected_embedding = OpenAIEmbedding(corrected)

            similarity = cosine_similarity([original_embedding], [corrected_embedding])[0][0]

            # 2. 감정 분석
            audio_features = extract_audio_features(audio_dir, start, end) # 음향적 특징 추출

            emotion_prompt = f"""
            문장: "{text}"
            평균 pitch: {audio_features["pitch_mean"]:.1f}Hz, pitch 변화량: {audio_features["pitch_std"]:.2f}
            평균 에너지: {audio_features["energy_mean"]:.5f}, 에너지 변화량: {audio_features["energy_std"]:.5f}
            말 빠르기(WPS): {wps:.2f}
            """

            emotion = self.openai_client.create_response(
                system_content = (
                    "당신은 문장의 감정을 분석하는 전문가입니다. "
                    "텍스트와 음향적 특성을 고려하여 이 문장의 감정을 추론하세요: "
                    "감정 하나의 단어만 출력하세요."
                ),
                user_content = emotion_prompt
            )

            print(f"[{start:.1f}s --> {end:.1f}s] [{emotion}] {corrected}")

            # 분석 된 문장 파일 저장을 위해 append
            sentences.append({
                "original_sentence": text,
                "corrected_sentence": corrected,
                "cosine_similarity": similarity,
                "emotion": emotion,
                "start": start,
                "end": end,
                "wps": wps
            })

        # JSON 데이터 구성
        output_data = {"interview": sentences, "LUFS": loudness}
        base_filename = os.path.splitext(os.path.basename(audio_dir))[0]

        dump_json(filename = base_filename, json_data = output_data)

class AudioEvaluator:
    def __init__(self, filename):
        self.filename = filename
        self.data = read_json(filename = filename)
    
    def evaluate(self):
        # 문장별 평가
        similarities = []
        for idx, sentence in enumerate(self.data.get("interview", []), 1):
            similarity = sentence.get("cosine_similarity")
            emotion = sentence.get("emotion")
            wps = sentence.get("wps")
            
            similarities.append(similarity)
            if similarity < 0.8:
                print(f"❌ 문장 {idx}: 의미적 차이 큼 (유사도 {similarity:.4f}) — 교정 필요")
            elif similarity < 0.9:
                print(f"⚠️ 문장 {idx}: 약간의 차이 있음 (유사도 {similarity:.4f}) — 자연스러움 개선")
            else:
                print(f"✅ 문장 {idx}: 거의 동일 (유사도 {similarity:.4f}) — 문법 교정만 필요")

            print(f"문장 {idx}의 감정 : {emotion}")

            if wps is None:
                print(f"문장 {idx}: WPS 데이터가 없습니다.")
            elif 1.73 <= wps <= 2.59:
                print(f"✅ 문장 {idx}의 WPS [{wps}] : 정상")
            else:
                print(f"⚠️ 문장 {idx}의 WPS [{wps}] : 빠르거나 느립니다.")
    
        # 전체 평가
        lufs = self.data.get("LUFS")
        if lufs is None:
            print("LUFS 데이터가 없습니다.")
        elif -24 <= lufs <= -22:
            print(f"✅ 목소리 평균 크기 [{lufs} LUFS] : 정상")
        else:
            print(f"⚠️ 목소리 평균 크기 [{lufs} LUFS] : 너무 크거나 작습니다. ")
        
        if similarities:
            avg_sim = sum(similarities) / len(similarities)
            print(f"\n📊 전체 평균 코사인 유사도: {avg_sim:.4f}")