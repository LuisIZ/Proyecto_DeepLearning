"""
record_agent.py  —  Graba un episodio del agente entrenado
===========================================================
Optimizado para: WSL2 + GTX 1660 Ti 6 GB + Ryzen 7 3750H + 16 GB RAM

REGLA DEL PROYECTO:
  El agente recibe ÚNICAMENTE la imagen de pantalla (screen).
  NO se usan labels_buffer ni posición de enemigos.
  Las game_variables son solo para reward shaping en entrenamiento;
  CnnPolicy solo ve frames visuales apilados.
"""

import argparse
import os
import numpy as np
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecTransposeImage,
    VecFrameStack,
    VecMonitor,
)
from doom_gym import DoomEnv


def main():
    parser = argparse.ArgumentParser(description="Grabar gameplay del agente entrenado")
    parser.add_argument("--model",    type=str, default="./models/best_model.zip")
    parser.add_argument("--config",   type=str, default="train_config.yaml")
    parser.add_argument("--output",   type=str, default="marine_gameplay.mp4")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--fps",      type=float, default=35.0)
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"❌ No se encontró el modelo en '{args.model}'")
        print("   Corre primero: python3 train.py")
        return

    print("🎬 Preparando entorno de grabación...")
    env = make_vec_env(lambda: DoomEnv(args.config), n_envs=1, vec_env_cls=DummyVecEnv)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    env = VecMonitor(env)

    print(f"🧠 Cargando modelo: {args.model}")
    model = PPO.load(args.model, env=env)
    print("✅ Modelo cargado.\n")

    all_frames  = []
    total_kills = 0

    for ep in range(args.episodes):
        obs    = env.reset()
        frames = []
        ep_reward = 0.0
        ep_kills  = 0
        ep_shots  = 0
        ep_acc    = 0.0
        done      = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            # Tomar el frame más reciente del stack de 4 (últimos 3 canales = RGB)
            recent = obs[0][-3:, :, :]               # (3, H, W)
            frame  = np.transpose(recent, (1, 2, 0)) # (H, W, 3)
            frames.append(frame.copy())

            ep_reward += float(rewards[0])

            if dones[0]:
                done = True
                info = infos[0]
                ep_kills = int(info.get("ep_kills", info.get("kills_episode", 0)))
                ep_shots = int(info.get("ep_shots", 0))
                ep_acc   = float(info.get("ep_accuracy", 0.0))

        all_frames.extend(frames)
        total_kills += ep_kills

        print(f"  Ep {ep+1}/{args.episodes}:  "
              f"reward={ep_reward:.1f}  |  "
              f"kills={ep_kills}  |  "
              f"shots={ep_shots}  |  "
              f"accuracy={ep_acc:.1f}%  |  "
              f"frames={len(frames)}")

    env.close()

    print(f"\n📊 Kills totales : {total_kills}")
    print(f"📽️  Guardando {len(all_frames)} frames en '{args.output}'...")

    try:
        imageio.mimsave(args.output, all_frames, fps=args.fps, quality=8)
        print(f"✅ Video guardado: {args.output}")
    except Exception as e:
        gif_path = args.output.rsplit(".", 1)[0] + ".gif"
        print(f"⚠️  MP4 falló ({e}) → guardando como GIF: {gif_path}")
        imageio.mimsave(gif_path, all_frames, fps=args.fps)
        print(f"✅ GIF guardado: {gif_path}")


if __name__ == "__main__":
    main()