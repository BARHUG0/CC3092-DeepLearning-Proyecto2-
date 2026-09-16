from __future__ import annotations

from functools import partial

import cv2
import gymnasium as gym
import ale_py
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecEnv

NOMBRE_ENTORNO = "ALE/SpaceInvaders-v5"
N_ACCIONES = 6
FORMA_GRIS = (4, 84, 84)
FORMA_RGB = (12, 84, 84)


class WarpFrameCanales(gym.ObservationWrapper):
    """Redimensiona a 84x84 y deja 1 canal (grises) o 3 (RGB), en formato canales-primero."""

    def __init__(self, env: gym.Env, ancho: int = 84, alto: int = 84, canales: int = 1) -> None:
        super().__init__(env)
        self.ancho = ancho
        self.alto = alto
        self.canales = canales
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(canales, alto, ancho),
            dtype=env.observation_space.dtype,
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        if self.canales == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.ancho, self.alto), interpolation=cv2.INTER_AREA)
        if self.canales == 1:
            return frame[None, :, :]
        return frame.transpose(2, 0, 1)


def envolver_atari(
    env: gym.Env,
    escala_grises: bool = True,
    vida_termina_episodio: bool = True,
    recorte_recompensa: bool = True,
) -> gym.Env:
    """Cadena de preprocesamiento de Atari equivalente a AtariWrapper de SB3, con escala de grises conmutable."""
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if vida_termina_episodio:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrameCanales(env, canales=1 if escala_grises else 3)
    if recorte_recompensa:
        env = ClipRewardEnv(env)
    return env


def crear_entorno_preprocesado(
    nombre_entorno: str = NOMBRE_ENTORNO,
    escala_grises: bool = True,
    vida_termina_episodio: bool = True,
    recorte_recompensa: bool = True,
    **kwargs,
) -> gym.Env:
    """Entorno individual preprocesado, para evaluación y video. frameskip=1 evita doble salto de frames con v5."""
    kwargs.setdefault("frameskip", 1)
    kwargs.setdefault("repeat_action_probability", 0.0)
    kwargs.setdefault("full_action_space", False)
    env = gym.make(nombre_entorno, **kwargs)
    return envolver_atari(env, escala_grises, vida_termina_episodio, recorte_recompensa)


def crear_entornos_vectorizados(
    n_entornos: int = 1,
    semilla: int = 0,
    escala_grises: bool = True,
    n_apilados: int = 4,
    vida_termina_episodio: bool = True,
    recorte_recompensa: bool = True,
) -> VecEnv:
    """VecEnv para entrenamiento, con Monitor y apilado de frames."""
    envoltorio = partial(
        envolver_atari,
        escala_grises=escala_grises,
        vida_termina_episodio=vida_termina_episodio,
        recorte_recompensa=recorte_recompensa,
    )
    vec_env = DummyVecEnv(
        [
            partial(
                _crear_env_monitoreado,
                semilla=semilla + indice,
                envoltorio=envoltorio,
            )
            for indice in range(n_entornos)
        ]
    )
    return VecFrameStack(vec_env, n_stack=n_apilados, channels_order="first")


def _crear_env_monitoreado(semilla: int, envoltorio) -> gym.Env:
    kwargs = {"frameskip": 1, "repeat_action_probability": 0.0, "full_action_space": False}
    env = gym.make(NOMBRE_ENTORNO, **kwargs)
    env.action_space.seed(semilla)
    env.reset(seed=semilla)
    env = envoltorio(env)
    return Monitor(env)


if __name__ == "__main__":
    for escala_grises, forma_esperada in [(True, FORMA_GRIS), (False, FORMA_RGB)]:
        vec_env = crear_entornos_vectorizados(semilla=42, escala_grises=escala_grises)
        observacion = vec_env.reset()
        assert observacion.shape == (1, *forma_esperada), f"{observacion.shape} != {(1, *forma_esperada)}"
        vec_env.step([0])
        print(f"escala_grises={escala_grises}: forma {observacion.shape[1:]} OK, acciones {vec_env.action_space.n}")
        vec_env.close()
