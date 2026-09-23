import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
RUNS = RAIZ / "bitacora" / "runs"
FIGURAS = RAIZ / "informe" / "figuras"

ORDEN_RUNS = [
    "humo_30k",
    "e1_base",
    "e2a_lr5e5",
    "e2c_explor25",
    "e2b_buffer50k",
    "e3_final",
    "e4_ppo",
    "e5_impala",
    "e6_vgg",
    "e7_rgb",
    "e8_dueling",
]

EVAL_GREEDY = {
    "humo_30k": (420, 303.0),
    "e1_base": (1300, 788.0),
    "e2a_lr5e5": (600, 586.0),
    "e2c_explor25": (1305, 911.0),
    "e2b_buffer50k": (780, 574.0),
    "e3_final": (2040, 1395.0),
    "e4_ppo": (395, 306.0),
    "e5_impala": (1350, 1100.0),
    "e6_vgg": (285, 285.0),
    "e7_rgb": (975, 607.0),
    "e8_dueling": (1385, 894.0),
}

FPS_BACKBONES = {
    "NatureCNN": 283,
    "ImpalaResNet": 176,
    "VGG16": 43,
}


def cargar_resumen(run_id):
    ruta = RUNS / run_id / "resumen_escalares.json"
    with ruta.open(encoding="utf-8") as f:
        return json.load(f)["escalares"]


def estilo():
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150})
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def guardar(nombre):
    FIGURAS.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURAS / nombre, dpi=150)
    plt.close()


def comparacion_runs():
    maximos = [EVAL_GREEDY[r][0] for r in ORDEN_RUNS]
    medias = [EVAL_GREEDY[r][1] for r in ORDEN_RUNS]
    x = np.arange(len(ORDEN_RUNS))
    plt.figure(figsize=(11, 5))
    plt.bar(x - 0.2, maximos, 0.4, label="Recompensa máxima")
    plt.bar(x + 0.2, medias, 0.4, label="Recompensa media")
    plt.xticks(x, ORDEN_RUNS, rotation=45, ha="right")
    plt.ylabel("Recompensa en evaluación greedy")
    plt.title("Evaluación greedy por run")
    plt.legend()
    estilo()
    guardar("comparacion_runs.png")


def curvas_entrenamiento():
    runs = ["e1_base", "e2c_explor25", "e3_final"]
    ultimas = [cargar_resumen(r)["rollout/ep_rew_mean"]["ultimo"] for r in runs]
    maximas = [cargar_resumen(r)["rollout/ep_rew_mean"]["maximo"] for r in runs]
    x = np.arange(len(runs))
    plt.figure(figsize=(7, 4.5))
    plt.bar(x - 0.2, ultimas, 0.4, label="Media final")
    plt.bar(x + 0.2, maximas, 0.4, label="Media máxima")
    plt.xticks(x, runs)
    plt.ylabel("Recompensa media de entrenamiento (clipped)")
    plt.title("Recompensa de entrenamiento por run")
    plt.legend()
    estilo()
    guardar("curvas_entrenamiento.png")


def perdida_runs():
    runs = ["e1_base", "e2a_lr5e5", "e2c_explor25"]
    ultimas = [cargar_resumen(r)["train/loss"]["ultimo"] for r in runs]
    minimas = [cargar_resumen(r)["train/loss"]["minimo"] for r in runs]
    maximas = [cargar_resumen(r)["train/loss"]["maximo"] for r in runs]
    x = np.arange(len(runs))
    plt.figure(figsize=(7, 4.5))
    plt.bar(x - 0.25, minimas, 0.25, label="Pérdida mínima")
    plt.bar(x, ultimas, 0.25, label="Pérdida final")
    plt.bar(x + 0.25, maximas, 0.25, label="Pérdida máxima")
    plt.xticks(x, runs)
    plt.ylabel("Pérdida")
    plt.title("Pérdida de entrenamiento (DQN)")
    plt.legend()
    estilo()
    guardar("perdida_e1_e2a_e2c.png")


def exploracion_runs():
    pasos = np.linspace(0, 1_000_000, 200)
    e1 = np.clip(0.1 - (0.1 - 0.01) * pasos / 100_000, 0.01, 0.1)
    e2c = np.clip(0.1 - (0.1 - 0.02) * pasos / 500_000, 0.02, 0.1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(pasos / 1000, e1, label="e1_base (eps 0.1→0.01)")
    plt.plot(pasos / 1000, e2c, label="e2c_explor25 (eps 0.1→0.02 en 500k)")
    plt.xlabel("Pasos (miles)")
    plt.ylabel("Tasa de exploración (epsilon)")
    plt.title("Programas de exploración")
    plt.legend()
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)
    guardar("exploracion_e1_e2c.png")


def benchmarks_backbones():
    nombres = list(FPS_BACKBONES)
    valores = list(FPS_BACKBONES.values())
    plt.figure(figsize=(7, 4.5))
    plt.bar(nombres, valores, color=["#4C72B0", "#DD8452", "#C44E52"])
    for i, v in enumerate(valores):
        plt.text(i, v + 5, str(v), ha="center")
    plt.ylabel("Pasos/s")
    plt.title("Velocidad de entrenamiento por backbone")
    plt.ylim(0, 330)
    estilo()
    guardar("benchmarks_backbones.png")


if __name__ == "__main__":
    comparacion_runs()
    curvas_entrenamiento()
    perdida_runs()
    exploracion_runs()
    benchmarks_backbones()
    print(f"Figuras generadas en {FIGURAS}")
