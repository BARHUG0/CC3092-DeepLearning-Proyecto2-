from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class BloqueResidual(nn.Module):
    def __init__(self, canales: int) -> None:
        super().__init__()
        self.convolucion1 = nn.Sequential(
            nn.Conv2d(canales, canales, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(canales),
            nn.ReLU(),
        )
        self.convolucion2 = nn.Sequential(
            nn.Conv2d(canales, canales, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(canales),
        )

    def forward(self, entrada: th.Tensor) -> th.Tensor:
        salida = self.convolucion2(self.convolucion1(entrada))
        return nn.functional.relu(entrada + salida)


class ImpalaResNet(BaseFeaturesExtractor):
    """Extractor de características ResNet de IMPALA: 3 bloques residuales con 16, 32 y 32 canales."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)
        canales_entrada = int(observation_space.shape[0])
        capas: list[nn.Module] = []
        for canales_salida in (16, 32, 32):
            capas.extend(
                [
                    nn.Conv2d(canales_entrada, canales_salida, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    BloqueResidual(canales_salida),
                ]
            )
            canales_entrada = canales_salida
        self.capas = nn.Sequential(*capas)
        self.proyeccion = nn.Linear(3200, features_dim)
        self.activacion = nn.ReLU()

    def forward(self, observacion: th.Tensor) -> th.Tensor:
        salida = self.capas(observacion)
        return self.activacion(self.proyeccion(salida.flatten(start_dim=1)))


class VGG16(BaseFeaturesExtractor):
    """Extractor de características VGG-16 clásico, sin normalización por lotes, sobre entrada de 84x84."""

    configuracion = [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ]

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)
        canales_entrada = int(observation_space.shape[0])
        capas: list[nn.Module] = []
        canales = canales_entrada
        for valor in self.configuracion:
            if valor == "M":
                capas.append(nn.MaxPool2d(2))
            else:
                capas.extend(
                    [
                        nn.Conv2d(canales, valor, kernel_size=3, padding=1),
                        nn.ReLU(),
                    ]
                )
                canales = valor
        self.convoluciones = nn.Sequential(*capas)
        self.proyeccion = nn.Linear(2048, features_dim)
        self.activacion = nn.ReLU()

    def forward(self, observacion: th.Tensor) -> th.Tensor:
        salida = self.convoluciones(observacion)
        return self.activacion(self.proyeccion(salida.flatten(start_dim=1)))


def contar_macs(modelo: nn.Module, forma_entrada: tuple[int, ...]) -> int:
    """Cuenta operaciones multiplicar-acumular de convoluciones y capas densas en un pase forward."""
    macs = 0

    def gancho(modulo: nn.Module, entradas: tuple[Any, ...], salida: th.Tensor) -> None:
        nonlocal macs
        if isinstance(modulo, nn.Conv2d):
            macs += int(salida.numel()) * (modulo.in_channels // modulo.groups) * modulo.kernel_size[0] * modulo.kernel_size[1]
        elif isinstance(modulo, nn.Linear):
            macs += int(np.prod(entradas[0].shape)) * salida.shape[-1]

    manijas = [modulo.register_forward_hook(gancho) for modulo in modelo.modules()]
    with th.no_grad():
        modelo(th.zeros(forma_entrada))
    for manija in manijas:
        manija.remove()
    return macs


def contar_parametros(modelo: nn.Module) -> int:
    return sum(parametro.numel() for parametro in modelo.parameters())


if __name__ == "__main__":
    from stable_baselines3.common.torch_layers import NatureCNN

    for canales, nombre_forma in [(4, "grises (4,84,84)"), (12, "RGB (12,84,84)")]:
        observation_space = spaces.Box(low=0, high=255, shape=(canales, 84, 84), dtype=np.uint8)
        print(f"--- {nombre_forma} ---")
        for nombre, clase in [
            ("NatureCNN", NatureCNN),
            ("ImpalaResNet", ImpalaResNet),
            ("VGG16", VGG16),
        ]:
            extractor = clase(observation_space, features_dim=256)
            salida = extractor(th.zeros(2, canales, 84, 84))
            assert salida.shape == (2, 256), f"{nombre}: {salida.shape}"
            macs = contar_macs(extractor, (1, canales, 84, 84))
            parametros = contar_parametros(extractor)
            print(f"{nombre}: salida {tuple(salida.shape[1:])}, parametros {parametros:,}, MACs {macs:,}")
