from audio_analisys import RealTimeAudioAnalyzerWhisper, RealTimeAudioAnalyzerVosk, RealTimeAudioAnalyzerGoogleSTT # 실시간
from audio_analisys import StaticAudioAnalyzer # 음성 파일
from audio_analisys import AudioEvaluator # 평가

import time

# analyzer = StaticAudioAnalyzer()

# for audio in [
#     "./audio/ckmk_a_bm_f_e_47109.wav",
#     "./audio/ckmk_a_bm_f_e_47110.wav",
#     "./audio/ckmk_a_bm_f_e_47111.wav",
#     "./audio/ckmk_a_bm_f_e_47112.wav"
# ]:
#     analyzer.analize_audio(audio)

# for json in [
#     "ckmk_a_bm_f_e_47109",
#     "ckmk_a_bm_f_e_47110",
#     "ckmk_a_bm_f_e_47111",
#     "ckmk_a_bm_f_e_47112"
# ]:
#     eval = AudioEvaluator(json)
#     eval.evaluate()

analyzer_whisper = RealTimeAudioAnalyzerWhisper()
analyzer_vosk = RealTimeAudioAnalyzerVosk()
analyzer_stt = RealTimeAudioAnalyzerGoogleSTT()
try:
    analyzer_whisper.start_stream()
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    analyzer_whisper.stop()