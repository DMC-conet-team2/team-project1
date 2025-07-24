from audio_analisys import RealTimeAudioAnalyzer

import time

analyzer = RealTimeAudioAnalyzer()
try:
    analyzer.start_stream()
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    analyzer.stop()