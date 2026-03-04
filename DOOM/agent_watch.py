"""
agent_watch.py  —  Observa al agente entrenado en tiempo real con HUD curses
=============================================================================
Muestra en la terminal: vida, kills, arma, munición, FPS, recompensa acumulada
y las acciones que el agente está tomando en cada tic.

Uso básico:
  python3 agent_watch.py --model ./models/best_phase2_dtc/best_model.zip \\
                         --config train_config.yaml

Con grabación:
  python3 agent_watch.py --model ./models/best_corridor/best_model.zip \\
                         --config train_config_corridor.yaml \\
                         --record \\
                         --output "final_test_$(date +%Y%m%d_%H%M)_esc2_corridor.mp4" \\
                         --episodes 3

Todos los escenarios de competencia:
  # Escenario 1
  python3 agent_watch.py --model ./models/best_rocket/best_model.zip \\
    --config train_config_rocket.yaml --record \\
    --output "final_test_$(date +%Y%m%d_%H%M)_esc1_rocket.mp4" --episodes 3

  # Escenario 2
  python3 agent_watch.py --model ./models/best_corridor/best_model.zip \\
    --config train_config_corridor.yaml --record \\
    --output "final_test_$(date +%Y%m%d_%H%M)_esc2_corridor.mp4" --episodes 3

  # Escenario 3
  python3 agent_watch.py --model ./models/best_phase2_dtc/best_model.zip \\
    --config train_config.yaml --record \\
    --output "final_test_$(date +%Y%m%d_%H%M)_esc3_center.mp4" --episodes 3

  # Escenario 4
  python3 agent_watch.py --model ./models/best_defend_line/best_model.zip \\
    --config train_config_defend_line.yaml --record \\
    --output "final_test_$(date +%Y%m%d_%H%M)_esc4_line.mp4" --episodes 3
"""

from __future__ import annotations

import os
import sys
import time
import curses
import argparse
import threading
import subprocess
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import cv2
import yaml
import tempfile

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


# ══════════════════════════════════════════════════════════════════════════════
#  Nombres de armas y dificultades
# ══════════════════════════════════════════════════════════════════════════════

WEAPON_NAMES = {
    1: "Fist/Pistol",
    2: "Shotgun    ",
    3: "Chaingun   ",
    4: "Rocket Lncr",
    5: "Plasma Gun ",
    6: "BFG 9000   ",
    7: "Chainsaw   ",
}

SKILL_NAMES = {
    1: "I'm Too Young To Die",
    2: "Hey, Not Too Rough  ",
    3: "Hurt Me Plenty      ",
    4: "Ultra-Violence      ",
    5: "Nightmare!          ",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Estado compartido del HUD
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HUDState:
    lock: threading.Lock = field(default_factory=threading.Lock)

    # Escenario
    doom_wad:   str = "unknown"
    doom_map:   str = "map01"
    doom_skill: int = 3
    model_path: str = ""

    # Variables de juego
    health:               float = 100.0
    armor:                float = 0.0
    killcount:            int   = 0
    selected_weapon:      int   = 1
    selected_weapon_ammo: float = 0.0
    ammo: Dict[int, float] = field(default_factory=lambda: {1:0, 2:0, 3:0, 4:0})

    # Episodio actual
    episode:       int   = 1
    total_episodes:int   = 1
    ep_kills:      int   = 0
    ep_reward:     float = 0.0

    # Rendimiento
    tics:     int   = 0
    fps:      float = 0.0
    step:     int   = 0

    # Acciones activas del agente
    active_actions: List[str] = field(default_factory=list)

    # Grabación
    recording:   bool = False
    output_file: str  = ""

    # Control
    running:    bool = True
    end_reason: str  = ""


# ══════════════════════════════════════════════════════════════════════════════
#  Colores curses
# ══════════════════════════════════════════════════════════════════════════════

C_TITLE  = 1
C_LABEL  = 2
C_VALUE  = 3
C_OK     = 4
C_WARN   = 5
C_DANGER = 6
C_REC    = 7
C_DIM    = 8
C_BORDER = 9
C_ACTON  = 10   # acción activa
C_ACTOFF = 11   # acción inactiva


def _init_colors() -> None:
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(C_TITLE,  curses.COLOR_RED,    -1)
    curses.init_pair(C_LABEL,  curses.COLOR_CYAN,   -1)
    curses.init_pair(C_VALUE,  curses.COLOR_WHITE,  -1)
    curses.init_pair(C_OK,     curses.COLOR_GREEN,  -1)
    curses.init_pair(C_WARN,   curses.COLOR_YELLOW, -1)
    curses.init_pair(C_DANGER, curses.COLOR_RED,    -1)
    curses.init_pair(C_REC,    curses.COLOR_RED,    -1)
    curses.init_pair(C_DIM,    curses.COLOR_BLACK,  -1)
    curses.init_pair(C_BORDER, curses.COLOR_BLUE,   -1)
    curses.init_pair(C_ACTON,  curses.COLOR_BLACK,  curses.COLOR_GREEN)
    curses.init_pair(C_ACTOFF, curses.COLOR_WHITE,  -1)


def _safe(win, y, x, text, attr=0):
    try:
        my, mx = win.getmaxyx()
        if y < 0 or y >= my or x < 0 or x >= mx:
            return
        avail = mx - x - 1
        if avail <= 0:
            return
        win.addstr(y, x, text[:avail], attr)
    except curses.error:
        pass


def _health_color(hp: float) -> int:
    if hp > 60: return C_OK
    if hp > 30: return C_WARN
    return C_DANGER


# ══════════════════════════════════════════════════════════════════════════════
#  Render del HUD
# ══════════════════════════════════════════════════════════════════════════════

def _render_hud(win, state: HUDState) -> None:
    win.erase()
    my, mx = win.getmaxyx()

    if my < 22 or mx < 52:
        _safe(win, 0, 0, "Terminal muy pequeña (mín 52×22)", curses.color_pair(C_DANGER))
        win.refresh()
        return

    W  = min(mx - 1, 76)
    ba = curses.color_pair(C_BORDER)
    la = curses.color_pair(C_LABEL)
    va = curses.color_pair(C_VALUE)
    da = curses.color_pair(C_DIM) | curses.A_DIM
    col1 = 24

    row = 0

    # ── Cabecera ──────────────────────────────────────────────────────────────
    _safe(win, row, 0, "╔" + "═"*(W-2) + "╗", ba)
    row += 1
    title = "  AGENTE RL  ─  DOOM  ─  HUD EN VIVO  "
    tx = max(0, (W - len(title)) // 2)
    _safe(win, row, 0, "║", ba); _safe(win, row, W-1, "║", ba)
    _safe(win, row, tx, title, curses.color_pair(C_TITLE) | curses.A_BOLD)
    row += 1

    # ── Separador modelo / escenario ──────────────────────────────────────────
    _safe(win, row, 0, "╠" + "═"*(W-2) + "╣", ba)
    row += 1
    wad_short  = os.path.basename(state.doom_wad).replace(".wad","")
    skill_name = SKILL_NAMES.get(state.doom_skill, str(state.doom_skill))
    model_short = os.path.basename(os.path.dirname(state.model_path))
    _safe(win, row, 0, "║", ba); _safe(win, row, W-1, "║", ba)
    _safe(win, row, 2, "Modelo: ", la)
    _safe(win, row, 10, model_short[:W-12], va | curses.A_BOLD)
    row += 1
    _safe(win, row, 0, "║", ba); _safe(win, row, W-1, "║", ba)
    _safe(win, row, 2, f"WAD: {wad_short}   Mapa: {state.doom_map}   Skill: {state.doom_skill} ({skill_name.strip()})", la)
    row += 1

    # ── Episodio ──────────────────────────────────────────────────────────────
    _safe(win, row, 0, "╠" + "═"*(col1) + "╦" + "═"*(W-col1-3) + "╣", ba)
    row += 1

    ep_str = f"Episodio {state.episode} / {state.total_episodes}"
    hp_attr = curses.color_pair(_health_color(state.health)) | curses.A_BOLD

    left_rows = [
        ("  Episodio", f"{state.episode} / {state.total_episodes}"),
        ("  Paso    ", f"{state.step:,}"),
        ("  FPS     ", f"{state.fps:.1f}"),
    ]
    right_rows = [
        ("♥  Vida   ", f"{int(state.health):>3}",  hp_attr),
        ("✦  Armadura", f"{int(state.armor):>3}", curses.color_pair(C_LABEL)),
        ("☠  Kills  ", f"{state.killcount:>3}",   curses.color_pair(C_OK) | curses.A_BOLD),
    ]

    for i in range(3):
        _safe(win, row, 0, "║", ba); _safe(win, row, col1+1, "║", ba); _safe(win, row, W-1, "║", ba)
        lbl, val = left_rows[i]
        _safe(win, row, 1, lbl, la); _safe(win, row, 1+len(lbl), val, va)
        lbl, val, attr = right_rows[i]
        _safe(win, row, col1+3, lbl, la); _safe(win, row, col1+3+len(lbl), val, attr)
        row += 1

    # ── Recompensa ────────────────────────────────────────────────────────────
    _safe(win, row, 0, "╠" + "═"*(col1) + "╩" + "═"*(W-col1-3) + "╣", ba)
    row += 1
    _safe(win, row, 0, "║", ba); _safe(win, row, W-1, "║", ba)
    rew_color = curses.color_pair(C_OK) if state.ep_reward >= 0 else curses.color_pair(C_DANGER)
    _safe(win, row, 2, "Recompensa ep:", la)
    _safe(win, row, 17, f"{state.ep_reward:>+10.2f}", rew_color | curses.A_BOLD)
    row += 1

    # ── Acciones activas del agente ───────────────────────────────────────────
    _safe(win, row, 0, "╠" + "═"*(W-2) + "╣", ba)
    row += 1
    _safe(win, row, 0, "║", ba); _safe(win, row, W-1, "║", ba)
    _safe(win, row, 2, "ACCIONES: ", la)
    ax = 12
    # Botones canónicos del agente en el orden que se suele ver
    all_buttons = [
        "MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT",
        "TURN_LEFT", "TURN_RIGHT", "ATTACK", "USE", "SPEED",
    ]
    short = {
        "MOVE_FORWARD":  "FWD",  "MOVE_BACKWARD": "BWD",
        "MOVE_LEFT":     "◄",    "MOVE_RIGHT":    "►",
        "TURN_LEFT":     "↺",    "TURN_RIGHT":    "↻",
        "ATTACK":        "ATK",  "USE":           "USE",
        "SPEED":         "SPD",
    }
    for btn in all_buttons:
        active = btn in state.active_actions
        label  = f" {short.get(btn, btn[:3])} "
        attr   = curses.color_pair(C_ACTON) | curses.A_BOLD if active else da
        if ax + len(label) + 1 >= W - 1:
            break
        _safe(win, row, ax, label, attr)
        ax += len(label) + 1
    row += 1

    # ── Grabación / controles ─────────────────────────────────────────────────
    _safe(win, row, 0, "╠" + "═"*(W-2) + "╣", ba)
    row += 1
    _safe(win, row, 0, "║", ba); _safe(win, row, W-1, "║", ba)
    if state.recording:
        _safe(win, row, 2, "● REC", curses.color_pair(C_REC) | curses.A_BOLD | curses.A_BLINK)
        _safe(win, row, 9, f"  {os.path.basename(state.output_file)}", va)
    else:
        _safe(win, row, 2, "○  Solo visualización (sin grabación)", da)
    _safe(win, row, W-20, "[Ctrl+C para detener]", da)
    row += 1

    _safe(win, row, 0, "╚" + "═"*(W-2) + "╝", ba)
    win.refresh()


def _render_end(win, state: HUDState) -> None:
    my, mx = win.getmaxyx()
    W  = min(mx - 1, 76)
    ba = curses.color_pair(C_BORDER)

    messages = {
        "done":    ("  TODOS LOS EPISODIOS COMPLETADOS  ", curses.color_pair(C_OK) | curses.A_BOLD),
        "user":    ("  DETENIDO POR EL USUARIO  ",         curses.color_pair(C_WARN) | curses.A_BOLD),
        "error":   ("  ERROR — VER TERMINAL  ",            curses.color_pair(C_DANGER) | curses.A_BOLD),
    }
    msg_text, msg_attr = messages.get(state.end_reason, ("  FINALIZADO  ", curses.color_pair(C_VALUE)))

    win.erase()
    row = max(0, my//2 - 4)
    _safe(win, row, 0, "╔" + "═"*(W-2) + "╗", ba); row += 1
    cx = max(0, (W - len(msg_text)) // 2)
    _safe(win, row, 0, "║" + " "*(W-2) + "║", ba)
    _safe(win, row, cx, msg_text, msg_attr); row += 1
    _safe(win, row, 0, "╠" + "═"*(W-2) + "╣", ba); row += 1
    for s in [
        f"  Modelo  : {os.path.basename(os.path.dirname(state.model_path))}",
        f"  WAD     : {os.path.basename(state.doom_wad)}",
        f"  Kills   : {state.killcount}",
        f"  Pasos   : {state.step:,}",
    ]:
        _safe(win, row, 0, "║" + " "*(W-2) + "║", ba)
        _safe(win, row, 2, s, curses.color_pair(C_VALUE)); row += 1
    _safe(win, row, 0, "╚" + "═"*(W-2) + "╝", ba)
    win.refresh()


# ══════════════════════════════════════════════════════════════════════════════
#  Hilo del HUD
# ══════════════════════════════════════════════════════════════════════════════

class CursesHUD(threading.Thread):
    def __init__(self, state: HUDState, hz: float = 10.0) -> None:
        super().__init__(daemon=True, name="CursesHUD")
        self.state  = state
        self._period = 1.0 / max(1.0, hz)

    def run(self) -> None:
        try:
            curses.wrapper(self._main)
        except Exception:
            pass

    def _main(self, stdscr) -> None:
        curses.curs_set(0)
        stdscr.nodelay(True)
        _init_colors()
        next_t = time.perf_counter()
        while True:
            with self.state.lock:
                running = self.state.running
            _render_hud(stdscr, self.state)
            if not running:
                _render_end(stdscr, self.state)
                time.sleep(3.0)
                break
            next_t += self._period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter()


# ══════════════════════════════════════════════════════════════════════════════
#  Grabación en MP4 con imageio
# ══════════════════════════════════════════════════════════════════════════════

class VideoRecorder:
    """Acumula frames RGB y guarda MP4 al finalizar."""
    def __init__(self, output: str, fps: float = 35.0):
        self.output = output
        self.fps    = fps
        self.frames: List[np.ndarray] = []

    def add(self, frame: np.ndarray) -> None:
        """frame: (H, W, 3) uint8 RGB"""
        self.frames.append(frame)

    def save(self) -> None:
        if not self.frames:
            print("⚠️  Sin frames para guardar.")
            return
        print(f"📽️  Guardando {len(self.frames)} frames en '{self.output}'...")
        try:
            imageio.mimsave(self.output, self.frames, fps=self.fps, quality=8)
            print(f"✅ Video guardado: {self.output}")
        except Exception as e:
            fallback = self.output.rsplit(".", 1)[0] + ".gif"
            print(f"⚠️  MP4 falló ({e}) → guardando GIF: {fallback}")
            imageio.mimsave(fallback, self.frames, fps=self.fps)
            print(f"✅ GIF guardado: {fallback}")


# ══════════════════════════════════════════════════════════════════════════════
#  Loop principal del agente
# ══════════════════════════════════════════════════════════════════════════════

def _load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _action_names(button_names: List[str], action: np.ndarray) -> List[str]:
    """Devuelve los nombres de los botones activos (=1) en este paso."""
    flat = np.array(action).reshape(-1)
    return [button_names[i] for i, v in enumerate(flat) if v == 1]


def run_agent(args):
    # ── Cargar config para saber WAD/mapa ─────────────────────────────────
    cfg = _load_config(args.config)
    scn        = cfg.get("scenario", {})
    doom_wad   = str(scn.get("doom_scenario_path", "unknown"))
    doom_map   = str(scn.get("doom_map", "map01"))
    doom_skill = int(scn.get("doom_skill", 3))

    # ── Ajustes dinámicos de config (render, resolución) ──────────────────
    _tmp_config_file = None
    config_to_use = args.config
    needs_tmp_config = False
    
    if "render" not in cfg:
        cfg["render"] = {}
        
    if getattr(args, "show_window", False):
        cfg["render"]["visible_window"] = True
        needs_tmp_config = True
        
    if getattr(args, "resolution", ""):
        orig_res = cfg["render"].get("screen_resolution", "RES_160X120")
        cfg["render"]["model_resolution"] = orig_res
        
        res = args.resolution.upper()
        if not res.startswith("RES_"):
            res = "RES_" + res
            
        cfg["render"]["screen_resolution"] = res
        needs_tmp_config = True

    if needs_tmp_config:
        _tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        )
        yaml.dump(cfg, _tmp, allow_unicode=True)
        _tmp.flush()
        _tmp_config_file = _tmp.name
        _tmp.close()
        config_to_use = _tmp_config_file
        print(f"🔧 Config temporal ajustada: {_tmp_config_file}")

    # ── Estado compartido del HUD ─────────────────────────────────────────
    hud_state = HUDState(
        doom_wad=doom_wad,
        doom_map=doom_map,
        doom_skill=doom_skill,
        model_path=args.model,
        total_episodes=args.episodes,
        recording=args.record,
        output_file=args.output if args.record else "",
    )

    hud = CursesHUD(hud_state, hz=10.0)
    hud.start()

    # ── Grabación ─────────────────────────────────────────────────────────
    recorder: Optional[VideoRecorder] = None
    if args.record:
        recorder = VideoRecorder(args.output, fps=args.fps)

    # ── Construir entorno (igual que record_agent.py) ─────────────────────
    env = make_vec_env(
        lambda: DoomEnv(config_to_use),
        n_envs=1,
        vec_env_cls=DummyVecEnv,
    )
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    env = VecMonitor(env)

    # ── Cargar modelo ─────────────────────────────────────────────────────
    model = PPO.load(args.model, env=env)

    # Necesitamos los nombres de botones para el HUD de acciones
    # Los leemos directamente del env subyacente
    base_env = None
    try:
        base_env = env.envs[0]
        while hasattr(base_env, "env"):
            base_env = base_env.env
        button_names = base_env.controller.button_names
    except Exception:
        button_names = []  # fallback silencioso

    total_kills = 0
    total_steps = 0

    try:
        for ep in range(1, args.episodes + 1):
            obs = env.reset()
            ep_reward = 0.0
            ep_kills  = 0
            ep_steps  = 0
            done      = False

            with hud_state.lock:
                hud_state.episode  = ep
                hud_state.ep_kills = 0
                hud_state.ep_reward = 0.0
                hud_state.killcount = 0
                hud_state.health   = 100.0

            t_start = time.perf_counter()

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)

                ep_reward  += float(rewards[0])
                ep_steps   += 1
                total_steps += 1
                elapsed     = max(1e-9, time.perf_counter() - t_start)

                # Obtener frame raw si grabamos a alta resolución, o el obs de agente
                frame = None
                if recorder is not None and base_env is not None:
                    try:
                        state = base_env.controller.game.get_state()
                        if state is not None and state.screen_buffer is not None:
                            screen = np.ascontiguousarray(state.screen_buffer)
                            if screen.ndim == 3:
                                if screen.shape[0] in (1, 3) and screen.shape[-1] not in (1, 3):
                                    screen = np.transpose(screen, (1, 2, 0))
                                frame = screen.astype(np.uint8)
                    except Exception:
                        pass
                
                if frame is None:
                    recent = obs[0][-3:, :, :]               # (3, H, W)
                    frame  = np.transpose(recent, (1, 2, 0)) # (H, W, 3)
                
                if recorder is not None:
                    recorder.add(frame.copy())

                # Actualizar HUD
                info = infos[0]
                active = _action_names(button_names, action[0]) if button_names else []
                with hud_state.lock:
                    hud_state.health    = float(info.get("health",   hud_state.health))
                    hud_state.armor     = float(info.get("armor",    hud_state.armor))
                    hud_state.killcount = int(info.get("kills_episode", hud_state.killcount))
                    hud_state.ep_kills  = hud_state.killcount
                    hud_state.ep_reward = ep_reward
                    hud_state.step      = total_steps
                    hud_state.fps       = ep_steps / elapsed
                    hud_state.active_actions = active

                if dones[0]:
                    done     = True
                    ep_kills = int(info.get("ep_kills", info.get("kills_episode", 0)))
                    total_kills += ep_kills

    except KeyboardInterrupt:
        with hud_state.lock:
            hud_state.end_reason = "user"

    finally:
        env.close()

        # Limpiar config temporal si se creó
        if _tmp_config_file and os.path.isfile(_tmp_config_file):
            os.unlink(_tmp_config_file)

        # Guardar video si aplica
        if recorder is not None:
            recorder.save()

        # Apagar HUD
        with hud_state.lock:
            hud_state.running    = False
            if not hud_state.end_reason:
                hud_state.end_reason = "done"
        hud.join(timeout=5.0)

        # Resumen en stdout
        print(f"\n{'='*60}")
        print(f"  Modelo    : {args.model}")
        print(f"  Escenario : {doom_wad}  [{doom_map}]  skill {doom_skill}")
        print(f"  Episodios : {args.episodes}")
        print(f"  Kills tot.: {total_kills}")
        print(f"  Pasos tot.: {total_steps:,}")
        if recorder:
            print(f"  Video     : {args.output}")
        print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Observa al agente entrenado con HUD en tiempo real"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Ruta al .zip del modelo (ej: ./models/best_phase2_dtc/best_model.zip)"
    )
    parser.add_argument(
        "--config", type=str, default="train_config.yaml",
        help="Config YAML del escenario (default: train_config.yaml)"
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Número de episodios a correr (default: 3)"
    )
    parser.add_argument(
        "--record", action="store_true",
        help="Si se activa, graba el video"
    )
    parser.add_argument(
        "--output", type=str, default="agent_replay.mp4",
        help="Ruta del video de salida (default: agent_replay.mp4)"
    )
    parser.add_argument(
        "--fps", type=float, default=35.0,
        help="FPS del video (default: 35.0)"
    )
    parser.add_argument(
        "--show-window", action="store_true",
        help="Renderiza la ventana de Doom en tiempo real (más lento, pero puedes VER al agente jugar)"
    )
    parser.add_argument(
        "--resolution", type=str, default="",
        help="Resolución de la ventana y grabación (ej: 640X480 o 800X600). El agente recibe intactos sus 160x120."
    )
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"❌ No se encontró el modelo: {args.model}")
        sys.exit(1)
    if not os.path.isfile(args.config):
        print(f"❌ No se encontró el config: {args.config}")
        sys.exit(1)

    run_agent(args)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()