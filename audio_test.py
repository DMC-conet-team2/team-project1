from audio_analisys import RealTimeAudioAnalyzer, StaticAudioAnalyzer, AudioEvaluator

import time

# analyzer = StaticAudioAnalyzer()

# for audio in [
#     "./audio/ckmk_a_bm_f_e_47109.wav",
#     "./audio/ckmk_a_bm_f_e_47110.wav",
#     "./audio/ckmk_a_bm_f_e_47111.wav",
#     "./audio/ckmk_a_bm_f_e_47112.wav"
# ]:
#     analyzer.analize_audio(audio)

for json in [
    "ckmk_a_bm_f_e_47109",
    "ckmk_a_bm_f_e_47110",
    "ckmk_a_bm_f_e_47111",
    "ckmk_a_bm_f_e_47112"
]:
    eval = AudioEvaluator(json)
    eval.evaluate()

# analyzer = RealTimeAudioAnalyzer()
# try:
#     analyzer.start_stream()
#     while True:
#         time.sleep(1)
# except KeyboardInterrupt:
#     analyzer.stop()