# scripts/reconstruccion

Evaluacion de un modelo ya entrenado a partir de su checkpoint. No entrenan
nada: cargan los coeficientes y rasterizan.

| script | para que sirve |
| --- | --- |
| `regenerar_clip_desde_checkpoint_streaming.py` | regenera todos los frames sin apilar el clip en GPU; es la ruta a usar en 720p o con muchas gaussianas |
| `regenerar_fps_interpolado.py` | evalua el modelo en instantes intermedios para subir los FPS (interpolacion temporal y camara lenta) |
