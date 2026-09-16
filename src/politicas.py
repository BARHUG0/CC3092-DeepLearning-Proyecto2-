from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.dqn import CnnPolicy
from stable_baselines3.dqn.policies import QNetwork
from torch import nn


class RedQDueling(QNetwork):
    """Red Q con cabeza dueling: Q(s,a) = V(s) + A(s,a) - media_a A(s,a)."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: object,
        features_dim: int,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
        )
        ocultas = self.net_arch[0] if self.net_arch else 512
        n_acciones = int(self.action_space.n)
        del self.q_net
        self.flujo_valor = nn.Sequential(
            nn.Linear(features_dim, ocultas),
            activation_fn(),
            nn.Linear(ocultas, 1),
        )
        self.flujo_ventaja = nn.Sequential(
            nn.Linear(features_dim, ocultas),
            activation_fn(),
            nn.Linear(ocultas, n_acciones),
        )

    def forward(self, observacion: th.Tensor) -> th.Tensor:
        caracteristicas = self.extract_features(observacion, self.features_extractor)
        valor = self.flujo_valor(caracteristicas)
        ventaja = self.flujo_ventaja(caracteristicas)
        return valor + ventaja - ventaja.mean(dim=1, keepdim=True)


class PoliticaDuelingDQN(CnnPolicy):
    """Política DQN para imágenes con cabeza dueling."""

    def make_q_net(self) -> QNetwork:
        argumentos = self._update_features_extractor(self.net_args, features_extractor=None)
        return RedQDueling(**argumentos).to(self.device)


if __name__ == "__main__":
    observation_space = spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
    politica = PoliticaDuelingDQN(observation_space, spaces.Discrete(6), lr_schedule=lambda progreso: 0.0001)
    observaciones = th.rand(8, 4, 84, 84)
    valores_q = politica.q_net(observaciones)
    assert valores_q.shape == (8, 6), f"forma {valores_q.shape} != (8, 6)"
    caracteristicas = politica.q_net.extract_features(observaciones, politica.q_net.features_extractor)
    valor = politica.q_net.flujo_valor(caracteristicas)
    ventaja = politica.q_net.flujo_ventaja(caracteristicas)
    reconstruccion = valor + ventaja - ventaja.mean(dim=1, keepdim=True)
    assert th.allclose(valores_q, reconstruccion), "la descomposición no coincide"
    ventajas_centradas = ventaja - ventaja.mean(dim=1, keepdim=True)
    assert th.allclose(ventajas_centradas.sum(dim=1), th.zeros(8), atol=1e-5), "las ventajas centradas no suman cero"
    print(f"Q con forma {tuple(valores_q.shape)} OK; ventajas centradas suman 0 por fila")
