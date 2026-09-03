# scripts/compresion

Reduccion del tamano del modelo despues del entrenamiento: poda de gaussianas
y cuantizacion de los coeficientes a UINT16.

| script | para que sirve |
| --- | --- |
| `run_binary_pruning_adaptativo.py` | busca por porcentaje cuantas gaussianas se pueden podar sin perder PSNR |
| `viz_prune_checkpoint_by_stats.py` | crea un checkpoint podado a partir de IDs o de estadisticas, sin reentrenar |
| `pack_checkpoint_uint16.py` / `unpack_checkpoint_uint16.py` | UINT16 SAFE: cuantiza los coeficientes conservando los mas sensibles |
| `pack_checkpoint_uint16_all.py` / `unpack_checkpoint_uint16_all.py` | UINT16 ALL: cuantiza todos los coeficientes |
