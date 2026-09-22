from __future__ import annotations

import glob
import os
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import ale_py
    

def crear_entorno(
    nombre_entorno: str,
    video_folder: str | None = None,
    name_prefix: str = "episodio",
    **kwargs: Any,
) -> gym.Env:
    """Crea un entorno de Gymnasium, con opcion para grabar video de todos los episodios."""
    episode_trigger = kwargs.pop("episode_trigger", None)
    if video_folder is not None:
        kwargs["render_mode"] = "rgb_array"
    env = gym.make(nombre_entorno, **kwargs)
    if video_folder is not None:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=episode_trigger if episode_trigger is not None else lambda episodio_id: True,
            name_prefix=name_prefix,
        )
    return env


def _mapear_acciones(env: gym.Env) -> dict[str, int] | None:
    """Diccionario: nombre -> significados de acción del entorno."""
    entorno_base = env.unwrapped
    if not hasattr(entorno_base, "get_action_meanings"):
        return None
    significados = entorno_base.get_action_meanings()
    return {nombre.upper(): indice for indice, nombre in enumerate(significados)}

def ejecutar_episodio(env: gym.Env, funcion_agente: Callable[[Any, gym.Env], int], max_steps: int = 10000, seed: int | None = None) -> dict:
    """Ejecuta un episodio completo y devuelve pasos y recompensa acumulada."""
    if seed is not None:
        observacion, _ = env.reset(seed=seed)
    else:
        observacion, _ = env.reset()
    recompensa_total = 0.0
    pasos = 0
    terminado = False
    while not terminado and pasos < max_steps:
        accion = funcion_agente(observacion, env)
        observacion, recompensa, terminacion, truncamiento, _ = env.step(accion)
        recompensa_total += float(recompensa)
        pasos += 1
        terminado = terminacion or truncamiento
    return {"pasos": pasos, "recompensa_total": recompensa_total}


def generar_video_agente(
    nombre_entorno: str,
    funcion_agente: Callable[[Any, gym.Env], int],
    video_folder: str,
    name_prefix: str = "episodio",
    n_episodios: int = 1,
    max_steps: int = 10000,
    **kwargs: Any,
) -> tuple[list[str], list[dict]]:
    """Ejecuta varios episodios grabando video y calculando metricas"""
    env = crear_entorno(nombre_entorno, video_folder=video_folder, name_prefix=name_prefix, **kwargs)
    metricas: list[dict] = []
    try:
        for _ in range(n_episodios):
            metricas.append(ejecutar_episodio(env, funcion_agente, max_steps=max_steps))
    finally:
        env.close()
    rutas = sorted(
        os.path.normpath(ruta) for ruta in glob.glob(os.path.join(video_folder, "**", "*.mp4"), recursive=True)
    )
    return rutas, metricas