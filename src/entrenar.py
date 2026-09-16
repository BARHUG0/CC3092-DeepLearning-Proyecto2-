from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import torch
import yaml
import extractores
import politicas
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import NatureCNN

from entorno import crear_entornos_vectorizados

EXTRACTORES = {
    "nature": NatureCNN,
    "impala": extractores.ImpalaResNet,
    "vgg": extractores.VGG16,
}


def construir_modelo(config: dict, vec_env) -> DQN | PPO:
    politica_kwargs = dict(
        features_extractor_class=EXTRACTORES[config["backbone"]],
        features_extractor_kwargs=dict(features_dim=config.get("features_dim", 256)),
    )
    argumentos: dict = dict(
        seed=config.get("semilla", 0),
        verbose=1,
        tensorboard_log=os.path.join("modelos", config["run_id"], "tensorboard"),
    )
    if config["algoritmo"] == "dqn":
        if config.get("dueling", False):
            politica_kwargs["policy_class"] = politicas.PoliticaDuelingDQN
        return DQN("CnnPolicy", vec_env, policy_kwargs=politica_kwargs, **config.get("hiperparametros", {}), **argumentos)
    if config["algoritmo"] == "ppo":
        return PPO("CnnPolicy", vec_env, policy_kwargs=politica_kwargs, **config.get("hiperparametros", {}), **argumentos)
    raise ValueError(f"algoritmo desconocido: {config['algoritmo']}")


def entrenar(config: dict) -> str:
    run_id = config["run_id"]
    carpeta = os.path.join("modelos", run_id)
    os.makedirs(carpeta, exist_ok=True)
    vec_env = crear_entornos_vectorizados(
        n_entornos=config.get("n_entornos", 1),
        semilla=config.get("semilla", 0),
        escala_grises=config.get("escala_grises", True),
        n_apilados=config.get("n_apilados", 4),
        vida_termina_episodio=config.get("vida_termina_episodio", True),
        recorte_recompensa=config.get("recorte_recompensa", True),
    )
    eval_env = crear_entornos_vectorizados(
        n_entornos=1,
        semilla=config.get("semilla", 0) + 1000,
        escala_grises=config.get("escala_grises", True),
        n_apilados=config.get("n_apilados", 4),
        vida_termina_episodio=False,
        recorte_recompensa=False,
    )
    modelo = construir_modelo(config, vec_env)
    with open(os.path.join(carpeta, "config.json"), "w", encoding="utf-8") as archivo:
        json.dump(
            {
                **config,
                "fecha": datetime.now().isoformat(),
                "parametros_extractor": extractores.contar_parametros(modelo.q_net.features_extractor),
                "cuda": torch.cuda.is_available(),
            },
            archivo,
            indent=2,
            ensure_ascii=False,
        )
    evaluacion = config.get("evaluacion", {})
    callbacks = [
        CheckpointCallback(
            save_freq=evaluacion.get("checkpoint_cada_pasos", 100_000),
            save_path=carpeta,
            name_prefix="checkpoint",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=carpeta,
            eval_freq=evaluacion.get("freq_pasos", 100_000),
            n_eval_episodes=evaluacion.get("n_episodios", 5),
            deterministic=True,
        ),
    ]
    modelo.learn(
        total_timesteps=config["pasos"],
        callback=callbacks,
        tb_log_name=run_id,
    )
    modelo.save(os.path.join(carpeta, "modelo_final"))
    vec_env.close()
    eval_env.close()
    return carpeta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    argumentos = parser.parse_args()
    with open(argumentos.config, encoding="utf-8") as archivo:
        config = yaml.safe_load(archivo)
    entrenar(config)
