from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ETIQUETAS = {
    "rollout/ep_rew_mean": "Recompensa media por episodio (entrenamiento)",
    "eval/mean_reward": "Recompensa media de evaluación (greedy)",
    "train/loss": "Pérdida",
}


def rutas_eventos(carpeta_modelos: str, run_id: str) -> list[str]:
    patron = os.path.join(carpeta_modelos, run_id, "tensorboard", "**", "events.out.tfevents.*")
    return sorted(glob.glob(patron, recursive=True))


def extraer_escalares(carpeta_modelos: str, run_id: str) -> dict[str, tuple[list[int], list[float]]]:
    escalares: dict[str, dict[int, float]] = {}
    for ruta in rutas_eventos(carpeta_modelos, run_id):
        acumulador = EventAccumulator(os.path.dirname(ruta), size_guidance={"scalars": 0})
        acumulador.Reload()
        for etiqueta in acumulador.Tags().get("scalars", []):
            eventos = acumulador.Scalars(etiqueta)
            serie = escalares.setdefault(etiqueta, {})
            for evento in eventos:
                serie[evento.step] = evento.value
    return {
        etiqueta: (sorted(serie), [serie[paso] for paso in sorted(serie)])
        for etiqueta, serie in escalares.items()
    }


def generar_curvas(run_id: str, carpeta_modelos: str = "modelos", salida_dir: str | None = None) -> str:
    salida_dir = salida_dir or os.path.join("bitacora", "runs", run_id)
    os.makedirs(salida_dir, exist_ok=True)
    escalares = extraer_escalares(carpeta_modelos, run_id)
    disponibles = [etiqueta for etiqueta in ETIQUETAS if etiqueta in escalares]
    fig, ejes = plt.subplots(1, len(disponibles), figsize=(6 * len(disponibles), 4), squeeze=False)
    resumen = {}
    for eje, etiqueta in zip(ejes[0], disponibles):
        pasos, valores = escalares[etiqueta]
        eje.plot(pasos, valores)
        eje.set_title(ETIQUETAS[etiqueta])
        eje.set_xlabel("Pasos")
        eje.grid(alpha=0.3)
        resumen[etiqueta] = {
            "ultimo": valores[-1],
            "maximo": max(valores),
            "minimo": min(valores),
            "n_puntos": len(valores),
        }
    fig.tight_layout()
    ruta_figura = os.path.join(salida_dir, "curvas.png")
    fig.savefig(ruta_figura, dpi=120)
    plt.close(fig)
    with open(os.path.join(salida_dir, "resumen_escalares.json"), "w", encoding="utf-8") as archivo:
        json.dump({"run_id": run_id, "escalares": resumen}, archivo, indent=2, ensure_ascii=False)
    return ruta_figura


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--carpeta-modelos", default="modelos")
    argumentos = parser.parse_args()
    ruta = generar_curvas(argumentos.run_id, argumentos.carpeta_modelos)
    print(f"curvas en {ruta}")
