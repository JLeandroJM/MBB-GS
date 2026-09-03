# scripts/datos

Preparacion de los datos de entrada: pasar de un MP4 a la secuencia PNG que
consume el entrenamiento, y volver de una carpeta de frames a un MP4.

| script | para que sirve |
| --- | --- |
| `extraer_clips_720p.py` | extrae frames 720p de un video a `data/clips/<clip>/` |
| `extraer_frames_originales.py` | extrae los frames GT alineados con un experimento, desde un frame inicial dado |
| `extraer_video_gt.py` | extrae el segmento original como MP4 con la misma duracion y resolucion que un experimento |
| `extraer.py` | extraccion generica de frames |
| `frames_a_video.py` | convierte una carpeta `frame_NNNN.png` en un MP4 |
