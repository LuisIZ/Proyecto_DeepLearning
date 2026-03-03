"""
train.py  –  Entrenamiento PPO con curriculum y métricas Doom en TensorBoard
==============================================================================
Optimizado para: WSL2 + GTX 1660 Ti 6 GB + Ryzen 7 3750H (4c/8t) + 16 GB RAM

ESCENARIOS DE COMPETENCIA (lunes 2 marzo):
  Escenario 1: rocket_basic.wad  → map32, skill 2, 3 botones, 300 tics
  Escenario 2: deadly_corridor.wad → map01, skill 3, 7 botones, 2100 tics

COMANDOS PARA HOY (en orden):
  # Paso 1 — Fine-tuning rocket_basic (~2.5h)
  python3 train.py --phase rocket \
    --resume ./models/best_phase2_dtc/best_model.zip \
    --steps-rocket 300000 \
    2>&1 | tee "logs_$(date +%Y%m%d_%H%M)_rocket.txt"

  # Paso 2 — Fine-tuning deadly_corridor (~5h), lanzar cuando termine rocket
  python3 train.py --phase corridor \
    --resume ./models/best_rocket/best_model.zip \
    --steps-corridor 500000 \
    2>&1 | tee "logs_$(date +%Y%m%d_%H%M)_corridor.txt"

  # TensorBoard (terminal separada)
  tensorboard --logdir ./logs/ --host 0.0.0.0
"""

import os
import argparse
import multiprocessing as mp

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecTransposeImage,
    VecFrameStack,
    VecMonitor,
)

from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value
    return func

from doom_gym import DoomEnv


# ─────────────────────────────────────────────────────────────────────────────
# DoomMetricsCallback — kills/health/ammo/accuracy en TensorBoard
# ─────────────────────────────────────────────────────────────────────────────

class DoomMetricsCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._reset_buffers()

    def _reset_buffers(self):
        self._kills      : list = []
        self._shots      : list = []
        self._accuracy   : list = []
        self._damage_rcv : list = []
        self._health_end : list = []
        self._armor_end  : list = []
        self._ammo_end   : list = []
        self._survived   : list = []
        self._n_episodes : int  = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if done and "ep_kills" in info:
                self._kills.append(float(info["ep_kills"]))
                self._shots.append(float(info["ep_shots"]))
                self._accuracy.append(float(info["ep_accuracy"]))
                self._damage_rcv.append(float(info["ep_damage_rcv"]))
                self._health_end.append(float(info["ep_health_end"]))
                self._armor_end.append(float(info["ep_armor_end"]))
                self._ammo_end.append(float(info["ep_ammo_end"]))
                self._survived.append(float(info["ep_survived"]))
                self._n_episodes += 1
        return True

    def _on_rollout_end(self):
        if not self._kills:
            return
        def mean(lst):
            return float(np.mean(lst)) if lst else 0.0

        self.logger.record("doom/kills_per_episode",  mean(self._kills))
        self.logger.record("doom/shots_per_episode",  mean(self._shots))
        self.logger.record("doom/accuracy_pct",       mean(self._accuracy))
        self.logger.record("doom/damage_received",    mean(self._damage_rcv))
        self.logger.record("doom/health_at_end",      mean(self._health_end))
        self.logger.record("doom/armor_at_end",       mean(self._armor_end))
        self.logger.record("doom/ammo_at_end",        mean(self._ammo_end))
        self.logger.record("doom/survival_rate",      mean(self._survived) * 100.0)
        self.logger.record("doom/episodes_logged",    float(self._n_episodes))

        print(
            f"  [Doom] {len(self._kills)} eps | "
            f"kills/ep={mean(self._kills):.2f} | "
            f"accuracy={mean(self._accuracy):.1f}% | "
            f"survival={mean(self._survived)*100:.0f}% | "
            f"health_end={mean(self._health_end):.0f} | "
            f"ammo_end={mean(self._ammo_end):.0f}"
        )
        self._reset_buffers()


# ─────────────────────────────────────────────────────────────────────────────
# Construcción de entornos
# ─────────────────────────────────────────────────────────────────────────────

def build_train_env(config_path: str, n_envs: int, seed: int = 0):
    env = make_vec_env(
        lambda: DoomEnv(config_path),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=DummyVecEnv,
    )
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    env = VecMonitor(env)
    return env


def build_eval_env(config_path: str, seed: int = 99):
    env = make_vec_env(
        lambda: DoomEnv(config_path),
        n_envs=1,
        seed=seed,
        vec_env_cls=DummyVecEnv,
    )
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    env = VecMonitor(env)
    return env


# ─────────────────────────────────────────────────────────────────────────────
# Hiperparámetros
# ─────────────────────────────────────────────────────────────────────────────

BASE_HPARAMS = dict(
    policy              = "CnnPolicy",
    learning_rate       = linear_schedule(2.5e-4),
    n_steps             = 512,
    batch_size          = 128,
    n_epochs            = 4,
    gamma               = 0.99,
    gae_lambda          = 0.95,
    clip_range          = 0.2,
    clip_range_vf       = None,
    vf_coef             = 0.5,
    max_grad_norm       = 0.5,
    normalize_advantage = True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Función genérica de entrenamiento de una fase
# ─────────────────────────────────────────────────────────────────────────────

def train_phase(
    config_path     : str,
    phase_name      : str,
    total_timesteps : int,
    n_envs          : int,
    log_dir         : str,
    model_dir       : str,
    pretrained_path : str | None = None,
    ent_coef        : float      = 0.01,
    device          : str        = "auto",
):
    print(f"\n{'='*62}")
    print(f"  FASE : {phase_name}")
    print(f"  Pasos: {total_timesteps:,}  |  Envs: {n_envs}  |  ent_coef: {ent_coef}")
    print(f"  Config: {config_path}")
    print(f"{'='*62}")

    os.makedirs(log_dir,   exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    train_env = build_train_env(config_path, n_envs=n_envs)
    eval_env  = build_eval_env(config_path)

    hparams = {**BASE_HPARAMS, "ent_coef": ent_coef}

    if pretrained_path and os.path.isfile(pretrained_path):
        print(f"📦 Cargando pesos pre-entrenados: {pretrained_path}")
        load_kwargs = {k: v for k, v in hparams.items() if k != "policy"}
        model = PPO.load(
            pretrained_path,
            env=train_env,
            verbose=1,
            tensorboard_log=log_dir,
            device=device,
            **load_kwargs,
        )
    else:
        model = PPO(env=train_env, verbose=1, tensorboard_log=log_dir,
                    device=device, **hparams)

    eval_freq = max(30_000 // n_envs, 500)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, f"best_{phase_name}"),
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(eval_freq // 2, 1000),
        save_path=os.path.join(model_dir, f"ckpt_{phase_name}"),
        name_prefix=f"ppo_{phase_name}",
        verbose=0,
    )
    doom_cb = DoomMetricsCallback(verbose=1)

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([eval_cb, ckpt_cb, doom_cb]),
        reset_num_timesteps=True,
        tb_log_name=f"PPO_{phase_name}",
        progress_bar=False,
    )

    save_path = os.path.join(model_dir, f"ppo_{phase_name}_final")
    model.save(save_path)
    print(f"\n✅ Fase '{phase_name}' terminada → {save_path}.zip")

    train_env.close()
    eval_env.close()

    return save_path + ".zip"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs",         type=int,   default=2)
    parser.add_argument("--device",         type=str,   default="auto")
    parser.add_argument("--resume",         type=str,   default=None,
        help="Ruta a .zip para continuar entrenamiento")

    # Selector de fase
    parser.add_argument("--phase", type=str, default="all",
        choices=["all", "phase1", "phase2", "rocket", "corridor",
                 "defend_line", "predict", "multiduel", "takecover", "deathmatch"],
        help=(
            "all          = curriculum completo (phase1+phase2)\n"
            "phase1       = solo basic.wad\n"
            "phase2       = solo defend_the_center\n"
            "rocket       = fine-tuning rocket_basic      (ESC 1)\n"
            "corridor     = fine-tuning deadly_corridor   (ESC 2)\n"
            "defend_line  = fine-tuning defend_the_line   (ESC 3 probable)\n"
            "predict      = fine-tuning predict_position  (ESC 3-4 probable)\n"
            "multiduel    = fine-tuning multi_duel        (ESC 4-5 probable)\n"
            "takecover    = fine-tuning take_cover        (ESC supervivencia)\n"
            "deathmatch   = fine-tuning deathmatch        (ESC 6 final)"
        ))

    # Timesteps por fase
    parser.add_argument("--steps-p1",       type=int, default=500_000)
    parser.add_argument("--steps-p2",       type=int, default=1_500_000)
    parser.add_argument("--steps-rocket",   type=int, default=300_000,
        help="Timesteps fine-tuning rocket_basic (default: 300 000 ≈ 2.5h)")
    parser.add_argument("--steps-corridor",    type=int, default=500_000,
        help="Timesteps fine-tuning deadly_corridor (default: 500 000 ≈ 5h)")
    parser.add_argument("--steps-defend-line", type=int, default=400_000,
        help="Timesteps fine-tuning defend_the_line (default: 400 000 ≈ 3.5h)")
    parser.add_argument("--steps-predict",     type=int, default=300_000,
        help="Timesteps fine-tuning predict_position (default: 300 000 ≈ 2.5h)")
    parser.add_argument("--steps-multiduel",   type=int, default=500_000,
        help="Timesteps fine-tuning multi_duel (default: 500 000 ≈ 5h)")
    parser.add_argument("--steps-takecover",   type=int, default=300_000,
        help="Timesteps fine-tuning take_cover (default: 300 000 ≈ 2.5h)")
    parser.add_argument("--steps-deathmatch",  type=int, default=1_000_000,
        help="Timesteps fine-tuning deathmatch (default: 1 000 000 ≈ 9h)")

    args = parser.parse_args()

    # ── Diagnóstico de hardware ───────────────────────────────────────────────
    print("\n" + "="*62)
    print("  DIAGNÓSTICO DE HARDWARE")
    print("="*62)
    print(f"  PyTorch CUDA disponible : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU detectada           : {gpu_name}")
        print(f"  VRAM total              : {vram_gb:.1f} GB")
    else:
        print("  ⚠️  GPU no detectada — entrenando en CPU (muy lento)")
    print(f"  CPUs visibles (WSL)     : {mp.cpu_count()}")
    print(f"  n_envs solicitado       : {args.n_envs}")
    print("="*62 + "\n")

    LOG_DIR   = "./logs/"
    MODEL_DIR = "./models/"
    p1_zip = p2_zip = rocket_zip = corridor_zip = None
    defend_line_zip = predict_zip = multiduel_zip = takecover_zip = deathmatch_zip = None

    # ── Fases del curriculum original ────────────────────────────────────────
    if args.phase in ("all", "phase1"):
        p1_zip = train_phase(
            config_path="train_config_basic.yaml", phase_name="phase1_basic",
            total_timesteps=args.steps_p1, n_envs=args.n_envs,
            log_dir=LOG_DIR, model_dir=MODEL_DIR,
            pretrained_path=args.resume, ent_coef=0.03, device=args.device,
        )

    if args.phase in ("all", "phase2"):
        if p1_zip is None:
            candidate = os.path.join(MODEL_DIR, "best_phase1_basic", "best_model.zip")
            p1_zip    = candidate if os.path.isfile(candidate) else args.resume
        p2_zip = train_phase(
            config_path="train_config.yaml", phase_name="phase2_dtc",
            total_timesteps=args.steps_p2, n_envs=args.n_envs,
            log_dir=LOG_DIR, model_dir=MODEL_DIR,
            pretrained_path=p1_zip, ent_coef=0.01, device=args.device,
        )

    # ── Fine-tuning ESCENARIO 1: rocket_basic ────────────────────────────────
    if args.phase == "rocket":
        resume = args.resume
        if resume is None:
            # Buscar automáticamente el mejor modelo disponible
            for candidate in [
                os.path.join(MODEL_DIR, "best_phase2_dtc", "best_model.zip"),
                os.path.join(MODEL_DIR, "best_model.zip"),
            ]:
                if os.path.isfile(candidate):
                    resume = candidate
                    break
        print(f"\n🚀 Fine-tuning ROCKET BASIC desde: {resume}")
        rocket_zip = train_phase(
            config_path     = "train_config_rocket.yaml",
            phase_name      = "rocket",
            total_timesteps = args.steps_rocket,
            n_envs          = args.n_envs,
            log_dir         = LOG_DIR,
            model_dir       = MODEL_DIR,
            pretrained_path = resume,
            ent_coef        = 0.01,    # Reducido de 0.02 a 0.01 para mejor exploitation en fine-tuning
            device          = args.device,
        )

    # ── Fine-tuning ESCENARIO 2: deadly_corridor ─────────────────────────────
    if args.phase == "corridor":
        resume = args.resume
        if resume is None:
            for candidate in [
                os.path.join(MODEL_DIR, "best_rocket", "best_model.zip"),
                os.path.join(MODEL_DIR, "best_phase2_dtc", "best_model.zip"),
                os.path.join(MODEL_DIR, "best_model.zip"),
            ]:
                if os.path.isfile(candidate):
                    resume = candidate
                    break
        print(f"\n☠️  Fine-tuning DEADLY CORRIDOR desde: {resume}")
        corridor_zip = train_phase(
            config_path     = "train_config_corridor.yaml",
            phase_name      = "corridor",
            total_timesteps = args.steps_corridor,
            n_envs          = args.n_envs,
            log_dir         = LOG_DIR,
            model_dir       = MODEL_DIR,
            pretrained_path = resume,
            ent_coef        = 0.01,    # Reducido de 0.02 a 0.01
            device          = args.device,
        )


    # ── Fine-tuning ESCENARIO 3: defend_the_line ─────────────────────────────
    if args.phase == "defend_line":
        resume = args.resume
        if resume is None:
            for candidate in [
                os.path.join(MODEL_DIR, "best_corridor", "best_model.zip"),
                os.path.join(MODEL_DIR, "best_phase2_dtc", "best_model.zip"),
                os.path.join(MODEL_DIR, "best_model.zip"),
            ]:
                if os.path.isfile(candidate): resume = candidate; break
        print(f"\n🛡️  Fine-tuning DEFEND THE LINE desde: {resume}")
        defend_line_zip = train_phase(
            config_path="train_config_defend_line.yaml", phase_name="defend_line",
            total_timesteps=args.steps_defend_line, n_envs=args.n_envs,
            log_dir=LOG_DIR, model_dir=MODEL_DIR, pretrained_path=resume,
            ent_coef=0.02, device=args.device,
        )

    # ── Fine-tuning ESCENARIO 3-4: predict_position ───────────────────────────
    if args.phase == "predict":
        resume = args.resume
        if resume is None:
            for candidate in [
                os.path.join(MODEL_DIR, "best_rocket", "best_model.zip"),
                os.path.join(MODEL_DIR, "best_phase2_dtc", "best_model.zip"),
                os.path.join(MODEL_DIR, "best_model.zip"),
            ]:
                if os.path.isfile(candidate): resume = candidate; break
        print(f"\n🎯  Fine-tuning PREDICT POSITION desde: {resume}")
        predict_zip = train_phase(
            config_path="train_config_predict.yaml", phase_name="predict",
            total_timesteps=args.steps_predict, n_envs=args.n_envs,
            log_dir=LOG_DIR, model_dir=MODEL_DIR, pretrained_path=resume,
            ent_coef=0.02, device=args.device,
        )

    # ── Fine-tuning ESCENARIO 4-5: multi_duel ────────────────────────────────
    if args.phase == "multiduel":
        resume = args.resume
        if resume is None:
            for candidate in [
                os.path.join(MODEL_DIR, "best_corridor", "best_model.zip"),
                os.path.join(MODEL_DIR, "best_phase2_dtc", "best_model.zip"),
                os.path.join(MODEL_DIR, "best_model.zip"),
            ]:
                if os.path.isfile(candidate): resume = candidate; break
        print(f"\n⚔️  Fine-tuning MULTI DUEL desde: {resume}")
        multiduel_zip = train_phase(
            config_path="train_config_multiduel.yaml", phase_name="multiduel",
            total_timesteps=args.steps_multiduel, n_envs=args.n_envs,
            log_dir=LOG_DIR, model_dir=MODEL_DIR, pretrained_path=resume,
            ent_coef=0.02, device=args.device,
        )

    # ── Fine-tuning SUPERVIVENCIA: take_cover ─────────────────────────────────
    if args.phase == "takecover":
        resume = args.resume
        if resume is None:
            for candidate in [
                os.path.join(MODEL_DIR, "best_corridor", "best_model.zip"),
                os.path.join(MODEL_DIR, "best_model.zip"),
            ]:
                if os.path.isfile(candidate): resume = candidate; break
        print(f"\n🛸  Fine-tuning TAKE COVER desde: {resume}")
        takecover_zip = train_phase(
            config_path="train_config_takecover.yaml", phase_name="takecover",
            total_timesteps=args.steps_takecover, n_envs=args.n_envs,
            log_dir=LOG_DIR, model_dir=MODEL_DIR, pretrained_path=resume,
            ent_coef=0.02, device=args.device,
        )

    # ── Fine-tuning ESCENARIO 6 FINAL: deathmatch ────────────────────────────
    if args.phase == "deathmatch":
        resume = args.resume
        if resume is None:
            for candidate in [
                os.path.join(MODEL_DIR, "best_corridor", "best_model.zip"),
                os.path.join(MODEL_DIR, "best_phase2_dtc", "best_model.zip"),
                os.path.join(MODEL_DIR, "best_model.zip"),
            ]:
                if os.path.isfile(candidate): resume = candidate; break
        print(f"\n💀  Fine-tuning DEATHMATCH desde: {resume}")
        print("    ⚠️  NOTA: Usando Opcion A (solo botones binarios, sin DELTA)")
        deathmatch_zip = train_phase(
            config_path="train_config_deathmatch.yaml", phase_name="deathmatch",
            total_timesteps=args.steps_deathmatch, n_envs=args.n_envs,
            log_dir=LOG_DIR, model_dir=MODEL_DIR, pretrained_path=resume,
            ent_coef=0.02, device=args.device,
        )

    # ── Copiar mejor modelo final ─────────────────────────────────────────────
    import shutil
    final_zip = (deathmatch_zip or takecover_zip or multiduel_zip or
               predict_zip or defend_line_zip or corridor_zip or
               rocket_zip or p2_zip or p1_zip)
    if final_zip and os.path.isfile(final_zip):
        best_path = os.path.join(MODEL_DIR, "best_model.zip")
        shutil.copy2(final_zip, best_path)
        print(f"\n🏆 Modelo final copiado como: {best_path}")

    print("\n🔥 Entrenamiento completo.")
    print(f"   TensorBoard: tensorboard --logdir {os.path.abspath(LOG_DIR)} --host 0.0.0.0")
    print("   → http://localhost:6006  → sección 'doom/' para kills y métricas\n")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()