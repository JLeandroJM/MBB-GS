# scripts/audio

Extension de audio con atomos de Gabor. Es un modelo distinto al de video: los
atomos son estaticos sobre el eje del tiempo, sin polinomios temporales.

| script | para que sirve |
| --- | --- |
| `train_gabor.py` | entrenamiento mono sobre la waveform |
| `train_gabor_stereo.py` | entrenamiento estereo en dominio L/R o Mid-Side |
| `visualizar.py` | waveform y espectrogramas, original frente a reconstruido |
| `test_gradientes_gabor.py` | valida el backward del kernel CUDA contra autograd; corre en CPU |
