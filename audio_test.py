from audio_analisys import RealtimeAudioAnalyzer # 실시간
from audio_analisys import StaticAudioAnalyzer # 음성 파일
from audio_analisys import AudioEvaluator # 평가

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

if __name__ == "__main__":
    analyzer = RealtimeAudioAnalyzer(language_code="ko-KR")
    analyzer.start()