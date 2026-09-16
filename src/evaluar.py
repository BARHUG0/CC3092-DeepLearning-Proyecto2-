from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import politicas
from ale_utils import crear_entorno, ejecutar_episodio
from stable_baselines3 import DQN, PPO

import extractores
from entorno import ApiladorFrames, NOMBRE_ENTORNO, crear_entorno_preprocesado, envolver_atari

EXTRACTORES_CUSTOM = {
    "impala": extractores.ImpalaResNet,
    "vgg": extractores.VGG16,
}


def leer_config(ruta_modelo: str) -> dict:
    config_ruta = os.path.join(os.path.dirname(ruta_modelo), "config.json")
    if not os.path.exists(config_ruta):
        return {}
    with open(config_ruta, encoding="utf-8") as archivo:
        return json.load(archivo)


def cargar_modelo(ruta_modelo: str) -> DQN | PPO:
    config = leer_config(ruta_modelo)
    custom_objects = {}
    if config.get("backbone") in EXTRACTORES_CUSTOM:
        custom_objects["features_extractor_class"] = EXTRACTORES_CUSTOM[config["backbone"]]
    if config.get("dueling", False):
        custom_objects["policy_class"] = politicas.PoliticaDuelingDQN
    clase = PPO if config.get("algoritmo") == "ppo" else DQN
    return clase.load(ruta_modelo, custom_objects=custom_objects)


def predecir(modelo: DQN | PPO, observacion) -> int:
    acciones, _ = modelo.predict(observacion, deterministic=True)
    return int(acciones)


def ejecutar_episodios(modelo: DQN | PPO, env, n_episodios: int, semilla: int) -> list[float]:
    puntajes = []
    env.reset(seed=semilla)
    try:
        for episodio in range(n_episodios):
            resultado = ejecutar_episodio(env, lambda observacion, ent: predecir(modelo, observacion))
            puntajes.append(resultado["recompensa_total"])
            print(f"episodio {episodio}: {resultado['recompensa_total']:.0f} puntos, {resultado['pasos']} pasos")
    finally:
        env.close()
    return puntajes


def evaluar(modelo: DQN | PPO, n_episodios: int = 5, semilla: int = 0, escala_grises: bool = True) -> list[float]:
    env = crear_entorno_preprocesado(
        escala_grises=escala_grises,
        vida_termina_episodio=False,
        recorte_recompensa=False,
    )
    return ejecutar_episodios(modelo, env, n_episodios, semilla)


def generar_video(
    modelo: DQN | PPO,
    carpeta_video: str,
    n_episodios: int = 1,
    semilla: int = 0,
    escala_grises: bool = True,
) -> list[str]:
    env = crear_entorno(
        NOMBRE_ENTORNO,
        video_folder=carpeta_video,
        name_prefix="agente",
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
        render_mode="rgb_array",
        episode_trigger=lambda episodio_id: episodio_id > 0,
    )
    env = envolver_atari(env, escala_grises=escala_grises, vida_termina_episodio=False, recorte_recompensa=False)
    env = ApiladorFrames(env, n_apilados=4)
    rutas = ejecutar_episodios(modelo, env, n_episodios, semilla)
    return sorted(os.path.normpath(r) for r in glob.glob(os.path.join(carpeta_video, "**", "*.mp4"), recursive=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", required=True, help="ruta al .zip del modelo, ej modelos/<run_id>/best_model.zip")
    parser.add_argument("--episodios", type=int, default=5)
    parser.add_argument("--semilla", type=int, default=0)
    parser.add_argument("--video", default=None, help="carpeta para el video, ej modelos/videos/<run_id>")
    parser.add_argument("--episodios_video", type=int, default=1)
    argumentos = parser.parse_args()

    modelo = cargar_modelo(argumentos.modelo)
    escala_grises = leer_config(argumentos.modelo).get("escala_grises", True)
    puntajes = evaluar(
        modelo,
        n_episodios=argumentos.episodios,
        semilla=argumentos.semilla,
        escala_grises=escala_grises,
    )
    print(f"max {max(puntajes):.0f} / media {np.mean(puntajes):.1f}")
    rutas_video = []
    if argumentos.video:
        os.makedirs(argumentos.video, exist_ok=True)
        rutas_video = generar_video(
            modelo,
            argumentos.video,
            n_episodios=argumentos.episodios_video,
            semilla=argumentos.semilla,
            escala_grises=escala_grises,
        )
    resultado = {
        "modelo": argumentos.modelo,
        "puntajes": puntajes,
        "max": float(max(puntajes)),
        "media": float(np.mean(puntajes)),
        "videos": rutas_video,
    }
    salida = os.path.splitext(argumentos.modelo)[0] + "_evaluacion.json"
    with open(salida, "w", encoding="utf-8") as archivo:
        json.dump(resultado, archivo, indent=2, ensure_ascii=False)
    print(f"resultados en {salida}")
