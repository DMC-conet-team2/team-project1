import librosa
import numpy as np

def extract_audio_features(audio_path, start, end, sr=16000):
        '''
        음성 파일에서 감정 변화를 분류하기 위해 특징들을 감지하는 함수

        파라미터
        - audio_path: 음성파일 경로
        - start: 시작 시간
        - end: 끝 시간
        - sr: sampling rate

        호출 예시: audio_features = extract_audio_features(audio_dir, start, end)
        '''

        y, sr = librosa.load(audio_path, sr=sr, offset=start, duration=end - start)

        # Pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_mean = float(np.mean(pitch_values)) if pitch_values.size > 0 else 0.0
        pitch_std = float(np.std(pitch_values)) if pitch_values.size > 0 else 0.0

        # Energy (RMS)
        rms = librosa.feature.rms(y=y)
        energy_mean = float(np.mean(rms)) if rms.size > 0 else 0.0
        energy_std = float(np.std(rms)) if rms.size > 0 else 0.0

        return {
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "energy_mean": energy_mean,
            "energy_std": energy_std
        }