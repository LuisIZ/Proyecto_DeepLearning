# Proyecto Deep Learning — DOOM AI Agent

Agente de inteligencia artificial entrenado para jugar **DOOM** de forma autónoma usando **Deep Reinforcement Learning**. El agente aprende únicamente a partir de los píxeles de la pantalla del juego, sin acceso a información interna como la posición de los enemigos.

---

## Algoritmo y enfoque

El agente utiliza **PPO (Proximal Policy Optimization)** con una política convolucional (`CnnPolicy`) implementada en Stable-Baselines3. El entrenamiento sigue un esquema de **curriculum learning**: el agente comienza en escenarios simples y progresa hacia entornos más complejos, transfiriendo los pesos aprendidos entre cada fase.

---

## Arquitectura del modelo

### Pipeline de observación

```
Pantalla del juego (RGB)
        │
        ▼
 Resolución: 160 × 120 px (RGB24)
        │
        ▼
 VecFrameStack — apila 4 frames consecutivos
        │  Permite percibir movimiento y velocidad sin estado recurrente
        ▼
 VecTransposeImage — (H, W, C×4) → (C×4, H, W)
        │  Formato CHW requerido por PyTorch
        ▼
 CnnPolicy (Stable-Baselines3 / PyTorch)
        │
        ├── Extractor de características (CNN)
        │     Conv2D(32, 8×8, stride=4) → ReLU
        │     Conv2D(64, 4×4, stride=2) → ReLU
        │     Conv2D(64, 3×3, stride=1) → ReLU
        │     Flatten → Linear(512)
        │
        ├── Actor: Linear(512) → MultiBinary(N botones)
        │     Produce probabilidades para cada botón (acción independiente)
        │
        └── Crítico: Linear(512) → valor escalar V(s)
              Estima el retorno esperado del estado actual
```

> **Nota:** La entrada real a la CNN es de forma `(12, 120, 160)` — 3 canales RGB × 4 frames apilados.

### Espacio de acciones

El espacio de acción es `MultiBinary(N)`, donde cada bit activa o desactiva un botón del juego de forma independiente. El número de botones varía por escenario:

| Escenario | Nº botones | Acciones disponibles |
|---|---|---|
| basic | 9 | MOVE_FWD/BACK/LEFT/RIGHT, TURN_L/R, ATTACK, USE, SPEED |
| defend_the_center | 9 | Idéntico a basic |
| rocket_basic | 3 | TURN_LEFT, TURN_RIGHT, ATTACK |
| deadly_corridor | 7 | MOVE_FWD/BACK, TURN_L/R, ATTACK, SPEED + variantes |
| defend_the_line | 7 | MOVE_FWD/BACK, TURN_L/R, ATTACK, SPEED + variantes |
| predict_position | 3–7 | Variante de corridor con movimiento lateral |
| take_cover | 3–5 | MOVE_LEFT, MOVE_RIGHT, sin ATTACK |
| deathmatch | 7 | MOVE_FWD/BACK/L/R, TURN_L/R, ATTACK |

### Hiperparámetros PPO

| Parámetro | Valor | Descripción |
|---|---|---|
| `learning_rate` | `2.5e-4` (lineal ↓ 0) | Decrecimiento lineal durante el entrenamiento |
| `n_steps` | 512 | Pasos recolectados por entorno antes de cada actualización |
| `batch_size` | 128 | Mini-lotes para el optimizador |
| `n_epochs` | 4 | Pasadas sobre el buffer de rollout |
| `gamma` | 0.99 | Factor de descuento para recompensas futuras |
| `gae_lambda` | 0.95 | Lambda del Generalized Advantage Estimation |
| `clip_range` | 0.2 | Clipping de la ratio de política (estabilidad PPO) |
| `vf_coef` | 0.5 | Peso de la pérdida del crítico |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `ent_coef` | 0.01–0.03 | Coeficiente de entropía (exploración; mayor en fases tempranas) |
| `normalize_advantage` | True | Normalización del advantage dentro de cada batch |
| `frame_skip` | 4 | Tics de VizDoom por acción tomada |
| `n_envs` | 2 | Entornos paralelos durante el entrenamiento |

### Función de recompensa

La recompensa es diseñada manualmente (*reward shaping*) en `doom_gym.py` y es completamente configurable por YAML:

| Componente | Valor típico | Evento que lo activa |
|---|---|---|
| Kill reward | +200 | Por cada enemigo eliminado |
| Health penalty | −1.5 × Δhealth | Al recibir daño (por punto de vida perdido) |
| Armor penalty | −0.5 × Δarmor | Al perder armadura |
| Death penalty | −50 | Al morir |
| Time penalty | −0.01 / tic | Costo por tiempo, incentiva eficiencia |
| Inactivity penalty | −0.2 | Si el agente no presiona ningún botón |
| Forward bonus | +0.005–0.05 | Por presionar MOVE_FORWARD (mayor en deadly_corridor) |
| Backward penalty | variante | En escenarios de pasillo, penaliza retroceder |

---

## Curriculum learning — fases de entrenamiento

El entrenamiento sigue un esquema progresivo de 8 fases. Los pesos se transfieren entre fases (*fine-tuning*):

```
Fase 1 — basic.wad (500k pasos)
  └── 1 enemigo, sala simple. Aprende: ver → girar → disparar.
        │
        ▼
Fase 2 — defend_the_center (1.5M pasos)
  └── Enemigos desde todos los ángulos. Aprende: rotar y disparar continuamente.
        │
        ├── rocket_basic (300k pasos)  ─── Escenario de competencia 1
        │     Rocket launcher. Solo 3 botones. Apuntar y disparar.
        │
        ├── deadly_corridor (500k pasos)  ─── Escenario de competencia 2
        │     Pasillo con enemigos. Aprende navegación con combate.
        │
        ├── defend_the_line (400k pasos)
        │     Defender posición. Oleadas de enemigos.
        │
        ├── predict_position (300k pasos)
        │     Predicción de movimiento del enemigo.
        │
        ├── take_cover (300k pasos)
        │     Supervivencia esquivando proyectiles.
        │
        └── deathmatch (1M pasos)  ─── Escenario final
              Combate completo: navegación, múltiples armas, exploración.
```

---

## Estructura del código

| Archivo | Descripción |
|---|---|
| `train.py` | Script principal de entrenamiento. Gestiona todas las fases del curriculum, callbacks de TensorBoard y guardado de modelos. |
| `doom_gym.py` | Entorno Gymnasium personalizado. Implementa el reward shaping completo y la interfaz con VizDoom. |
| `doom_controller.py` | Controlador de bajo nivel de VizDoom. Gestiona init, reset, step y grabación de episodios. |
| `agent_watch.py` | Visualiza al agente jugando en tiempo real con HUD de métricas en consola. |
| `doom_play.py` | Modo de juego humano con controles de teclado. |
| `record_agent.py` | Graba episodios del agente entrenado en video MP4. |
| `train_config_*.yaml` | Configuraciones por escenario: botones, reward, resolución, timeout, etc. |

---

## Tecnologías utilizadas

- [VizDoom](https://vizdoom.cs.put.edu.pl/) — motor de juego y API de entorno
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — implementación de PPO y utilidades de entrenamiento
- [PyTorch](https://pytorch.org/) — framework de deep learning (backend de SB3)
- [Gymnasium](https://gymnasium.farama.org/) — interfaz estándar para entornos RL
- TensorBoard — monitoreo de métricas de entrenamiento

**Hardware de desarrollo:** WSL2 + NVIDIA GTX 1660 Ti (6 GB VRAM) + Ryzen 7 3750H + 16 GB RAM

---

## Uso rápido

```bash
# Entrenamiento completo (curriculum phase1 + phase2)
python train.py --phase all --n-envs 2

# Fine-tuning en un escenario específico
python train.py --phase rocket  --resume ./models/best_phase2_dtc/best_model.zip
python train.py --phase corridor --resume ./models/best_rocket/best_model.zip

# Ver al agente jugar en tiempo real
python agent_watch.py --model ./models/best_model.zip --config train_config.yaml

# Grabar episodios en video
python record_agent.py --model ./models/best_model.zip --config train_config.yaml

# Monitorear entrenamiento con TensorBoard
tensorboard --logdir ./logs/ --host 0.0.0.0
# → http://localhost:6006  (sección 'doom/' para kills, accuracy y supervivencia)
```

---

## Herramientas de IA utilizadas

Durante el desarrollo de este proyecto se utilizaron asistentes de inteligencia artificial como apoyo en distintas etapas del proceso, incluyendo la escritura y depuración de código, diseño de la arquitectura del sistema, optimización de hiperparámetros y redacción de documentación. Entre las herramientas empleadas se encuentran **Gemini**, **Claude** y el IDE **Antigravity**.
