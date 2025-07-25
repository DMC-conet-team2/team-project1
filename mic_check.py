from time import sleep

import pyaudio
import numpy as np

CHUNK = 1024
RATE = 16000

p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("🎧 입력 볼륨 측정 중 (Ctrl+C로 종료)")

try:
    while True:
        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        volume = np.linalg.norm(data) / CHUNK
        print(f"🔊 볼륨: {volume:.2f}")
        sleep(1)
except KeyboardInterrupt:
    print("⛔ 종료됨")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()