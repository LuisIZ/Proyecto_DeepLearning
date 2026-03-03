"""
doom_gym.py  –  Gymnasium wrapper para VizDoom
Optimizado para: WSL2 + GTX 1660 Ti 6 GB + Ryzen 7 3750H + 16 GB RAM

NOVEDADES vs versión anterior:
  - Rewards completamente configurables desde el YAML (sección 'rewards:')
  - forward_bonus configurable → alto para deadly_corridor (navegación)
  - backward_penalty configurable → penaliza retroceder en pasillos
  - Manejo robusto de escenarios con pocos game_variables (rocket_basic, corridor)
  - AMMO7 añadido para rocket launcher (rockets usan slot 7 en Doom)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from doom_controller import DoomController


class DoomEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config_path: str = "train_config.yaml"):
        super().__init__()
        self.controller = DoomController(config_path)

        # ── Leer configuración de rewards del YAML ────────────────────────────
        rew_cfg = self.controller.cfg.get("rewards", {})
        self._kill_reward        = float(rew_cfg.get("kill_reward",        200.0))
        self._health_penalty     = float(rew_cfg.get("health_penalty",     1.5))
        self._armor_penalty      = float(rew_cfg.get("armor_penalty",      0.5))
        self._death_penalty      = float(rew_cfg.get("death_penalty",      50.0))
        self._inactivity_penalty = float(rew_cfg.get("inactivity_penalty", 0.2))
        self._forward_bonus      = float(rew_cfg.get("forward_bonus",      0.005))
        self._backward_penalty   = float(rew_cfg.get("backward_penalty",   0.0))
        self._time_penalty       = float(rew_cfg.get("time_penalty",       0.01))
        self._ammo_penalty       = float(rew_cfg.get("ammo_penalty",       0.0))

        # ── Espacios ──────────────────────────────────────────────────────────
        num_actions = len(self.controller.button_names)
        self.action_space = spaces.MultiBinary(num_actions)

        res_str = self.controller.cfg.get("render", {}).get(
            "screen_resolution", "RES_160X120"
        )
        h, w = self._parse_resolution(res_str)
        self.obs_h, self.obs_w = h, w

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, 3), dtype=np.uint8
        )

        # ── Índices de game variables ─────────────────────────────────────────
        gv_names  = self.controller.game_variable_names
        self._idx = {name: i for i, name in enumerate(gv_names)}

        # ── Estado interno del episodio ───────────────────────────────────────
        self._last_killcount : float = 0.0
        self._last_health    : float = 100.0
        self._last_armor     : float = 0.0
        self._last_ammo      : float = 0.0

        self._episode_kills  : int   = 0
        self._episode_shots  : int   = 0
        self._episode_dmg    : float = 0.0
        self._episode_steps  : int   = 0

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_resolution(res_str: str):
        """'RES_160X120' → (120, 160)"""
        try:
            _, dims = res_str.split("_", 1)
            w_str, h_str = dims.upper().split("X")
            return int(h_str), int(w_str)
        except Exception:
            return 120, 160

    def _read_gv(self, obs_dict):
        gv = obs_dict.get(
            "gamevariables",
            np.zeros(max(len(self._idx), 1), dtype=np.float32),
        )
        health    = float(gv[self._idx["HEALTH"]])    if "HEALTH"    in self._idx else 100.0
        armor     = float(gv[self._idx["ARMOR"]])     if "ARMOR"     in self._idx else 0.0
        killcount = float(gv[self._idx["KILLCOUNT"]]) if "KILLCOUNT" in self._idx else 0.0
        ammo      = 0.0
        # AMMO7 = rockets (slot 7 en Doom), AMMO2 = bullets
        for key in ("AMMO7", "AMMO2", "SELECTED_WEAPON_AMMO", "AMMO1", "AMMO3"):
            if key in self._idx:
                ammo = float(gv[self._idx[key]])
                break
        return health, armor, killcount, ammo

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs_dict = self.controller.reset(new_seed=seed)

        health, armor, killcount, ammo = self._read_gv(obs_dict)
        self._last_health    = health if health > 0 else 100.0
        self._last_armor     = armor
        self._last_killcount = killcount
        self._last_ammo      = ammo

        self._episode_kills  = 0
        self._episode_shots  = 0
        self._episode_dmg    = 0.0
        self._episode_steps  = 0

        return self._ensure_shape(obs_dict.get("screen")), {}

    def step(self, action):
        action   = np.array(action, dtype=np.int32)
        obs_dict, raw_reward, terminated, truncated, info = self.controller.step(action)

        health, armor, killcount, ammo = self._read_gv(obs_dict)
        reward = 0.0

        # ── 1. KILLS ──────────────────────────────────────────────────────────
        delta_kills = killcount - self._last_killcount
        if delta_kills > 0:
            reward += self._kill_reward * delta_kills
            self._episode_kills += int(delta_kills)

        # ── 2. Daño recibido ──────────────────────────────────────────────────
        delta_health = health - self._last_health
        if delta_health < 0:
            reward     += delta_health * self._health_penalty
            self._episode_dmg -= delta_health

        delta_armor  = armor - self._last_armor
        if delta_armor < 0:
            reward     += delta_armor * self._armor_penalty
            self._episode_dmg -= delta_armor

        # ── 3. Penalización por muerte ────────────────────────────────────────
        if terminated and health <= 0:
            reward -= self._death_penalty

        # ── Munición ──────────────────────────────────────────────────────────
        delta_ammo = ammo - self._last_ammo
        if delta_ammo < 0:
            reward += delta_ammo * self._ammo_penalty

        # ── 4. Conteo de disparos ─────────────────────────────────────────────
        try:
            atk_idx = self.controller.button_names.index("ATTACK")
            if action[atk_idx] == 1:
                self._episode_shots += 1
        except ValueError:
            pass

        # ── 5. Penalizar inactividad total ────────────────────────────────────
        if np.all(action == 0):
            reward -= self._inactivity_penalty

        # ── 6. Bono por avanzar (configurable) ───────────────────────────────
        # deadly_corridor: 0.05 → fuerza navegación del pasillo
        # rocket_basic:    0.0  → no hay MOVE_FORWARD en ese escenario
        # defend_center:   0.005 → exploración mínima
        if self._forward_bonus > 0:
            try:
                fwd_idx = self.controller.button_names.index("MOVE_FORWARD")
                if action[fwd_idx] == 1:
                    reward += self._forward_bonus
            except ValueError:
                pass

        # ── 7. Penalizar retroceder (deadly_corridor) ─────────────────────────
        if self._backward_penalty > 0:
            try:
                bwd_idx = self.controller.button_names.index("MOVE_BACKWARD")
                if action[bwd_idx] == 1:
                    reward -= self._backward_penalty
            except ValueError:
                pass

        # ── 8. Costo por tic ──────────────────────────────────────────────────
        reward -= self._time_penalty

        # ── Actualizar estado ─────────────────────────────────────────────────
        self._last_killcount = killcount
        self._last_health    = health
        self._last_armor     = armor
        self._last_ammo      = ammo
        self._episode_steps += 1

        info["kills_episode"] = self._episode_kills
        info["health"]        = health
        info["armor"]         = armor
        info["ammo"]          = ammo
        info["raw_reward"]    = raw_reward

        if terminated or truncated:
            shots = max(self._episode_shots, 1)
            info["ep_kills"]     = self._episode_kills
            info["ep_shots"]     = self._episode_shots
            info["ep_accuracy"]  = round(self._episode_kills / shots * 100, 2)
            info["ep_damage_rcv"]= round(self._episode_dmg, 1)
            info["ep_health_end"]= round(health, 1)
            info["ep_armor_end"] = round(armor, 1)
            info["ep_ammo_end"]  = round(ammo, 1)
            info["ep_survived"]  = float(health > 0)

        return (
            self._ensure_shape(obs_dict.get("screen")),
            reward,
            terminated,
            truncated,
            info,
        )

    def _ensure_shape(self, screen):
        if screen is None or not isinstance(screen, np.ndarray):
            return np.zeros((self.obs_h, self.obs_w, 3), dtype=np.uint8)
        if screen.ndim == 3 and screen.shape[0] == 3:
            screen = np.transpose(screen, (1, 2, 0))
        if screen.shape[:2] != (self.obs_h, self.obs_w):
            import cv2
            screen = cv2.resize(
                screen, (self.obs_w, self.obs_h), interpolation=cv2.INTER_LINEAR
            )
        return screen.astype(np.uint8)

    def render(self):
        pass

    def close(self):
        self.controller.close()