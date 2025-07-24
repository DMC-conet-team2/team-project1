from audio_analisys import RealTimeAudioAnalyzer, StaticAudioAnalyzer

import time

analyzer = StaticAudioAnalyzer()

for audio in [
    "./audio/ckmk_a_bm_f_e_47109.wav",
    "./audio/ckmk_a_bm_f_e_47110.wav",
    "./audio/ckmk_a_bm_f_e_47111.wav",
    "./audio/ckmk_a_bm_f_e_47112.wav"
]:
    analyzer.analize_audio(audio)

# analyzer = RealTimeAudioAnalyzer()
# try:
#     analyzer.start_stream()
#     while True:
#         time.sleep(1)
# except KeyboardInterrupt:
#     analyzer.stop()