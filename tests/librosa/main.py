import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


# Имя дирректории с wav файлами
dir_name = "songs"

# Файл передаётся аргументом: python main.py song.wav
# Если аргумент не указан — ищем любой wav в текущей папке
if len(sys.argv) > 1:
    audio_path = sys.argv[1]
else:
    wavs = [f for f in os.listdir(f"./{dir_name}") if f.endswith(".wav")]
    if not wavs:
        print("Не найден wav-файл. Передайте путь аргументом: python main.py song.wav")
        sys.exit(1)
    audio_path = f"./{dir_name}/{wavs[0]}"

# Обрезка: offset — с какой секунды, clip_duration — сколько секунд (None = весь файл)
clip_offset = 0.0
clip_duration = 16.0

print(f"Загружаем: {audio_path}")
y, sr = librosa.load(audio_path, mono=True, offset=clip_offset, duration=clip_duration)
duration = librosa.get_duration(y=y, sr=sr)
print(f"Отрезок: {clip_offset:.0f}–{clip_offset + duration:.0f} сек | Sample rate: {sr} Hz | Samples: {len(y)}")

fig, axes = plt.subplots(5, 1, figsize=(14, 18))
fig.suptitle(f"Анализ: {os.path.basename(audio_path)}", fontsize=14, fontweight="bold")

# 1. Waveform — временная форма сигнала
ax = axes[0]
librosa.display.waveshow(y, sr=sr, ax=ax, color="steelblue")
ax.set_title("Waveform (временная форма)")
ax.set_xlabel("Время (с)")
ax.set_ylabel("Амплитуда")

# 2. Spectrogram (Short-Time Fourier Transform)
ax = axes[1]
D = librosa.stft(y)
D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
img = librosa.display.specshow(D_db, sr=sr, x_axis="time", y_axis="hz", ax=ax, cmap="magma")
fig.colorbar(img, ax=ax, format="%+2.0f dB")
ax.set_title("Спектрограмма STFT (dB)")

# 3. Mel-spectrogram
ax = axes[2]
S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
S_mel_db = librosa.power_to_db(S_mel, ref=np.max)
img = librosa.display.specshow(S_mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="viridis")
fig.colorbar(img, ax=ax, format="%+2.0f dB")
ax.set_title("Мел-спектрограмма")

# 4. MFCCs (20 коэффициентов)
ax = axes[3]
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
img = librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=ax, cmap="coolwarm")
fig.colorbar(img, ax=ax)
ax.set_title("MFCC (20 коэффициентов)")
ax.set_ylabel("MFCC #")

# 5. Chroma (гармонические классы нот)
ax = axes[4]
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
img = librosa.display.specshow(chroma, sr=sr, x_axis="time", y_axis="chroma", ax=ax, cmap="YlOrRd")
fig.colorbar(img, ax=ax)
ax.set_title("Chroma (гармоника по нотам)")

plt.tight_layout()
out_path = os.path.splitext(audio_path)[0] + "_analysis.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"График сохранён: {out_path}")
plt.show()

# --- Числовые характеристики ---
print("\n=== Числовые характеристики ===")
print(f"RMS (громкость):          {librosa.feature.rms(y=y).mean():.4f}")
print(f"Zero Crossing Rate:       {librosa.feature.zero_crossing_rate(y).mean():.4f}")
print(f"Spectral Centroid (Гц):   {librosa.feature.spectral_centroid(y=y, sr=sr).mean():.1f}")
print(f"Spectral Bandwidth (Гц):  {librosa.feature.spectral_bandwidth(y=y, sr=sr).mean():.1f}")
print(f"Spectral Rolloff (Гц):    {librosa.feature.spectral_rolloff(y=y, sr=sr).mean():.1f}")
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
print(f"Темп (BPM):               {float(tempo):.1f}")
mfcc_means = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
for i, v in enumerate(mfcc_means, 1):
    print(f"  MFCC {i:2d} mean:           {v:.2f}")
