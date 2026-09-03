# scripts/pipeline

Orquestadores que encadenan varias etapas. Llaman por subprocess a los scripts
de las demas carpetas.

| script | para que sirve |
| --- | --- |
| `run_pipeline_video_audio.py` | pipeline audiovisual completo: extraccion, entrenamiento de video y audio, pruning, UINT16 y metricas |
| `run_tests_secuencial.py` | corre en orden la ablacion de perdidas por fases (Experimento 3 de la tesis) |
