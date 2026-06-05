from scipy.io import wavfile
from pathlib import Path

for p in [
    "data/audio/RockThatBody_61s_30s_stereo_44k.wav",
    "data/audio/RockThatBody_61s_30s_mono_44k.wav",
]:
    ruta = Path(p)
    if ruta.exists():
        sr, x = wavfile.read(ruta)
        print(p, "sr=", sr, "shape=", x.shape)
