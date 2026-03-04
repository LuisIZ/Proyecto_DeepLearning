# Proyecto Deep Learning — DOOM AI Agent

Este proyecto entrena una inteligencia artificial para jugar el videojuego clásico **DOOM** de forma autónoma, usando **Deep Reinforcement Learning**.

El agente aprende a jugar únicamente viendo la pantalla del juego (frames RGB), sin acceso a información interna como la posición de los enemigos. Utiliza el algoritmo **PPO (Proximal Policy Optimization)** con una red neuronal convolucional (CNN) que procesa 4 frames consecutivos para entender el movimiento en el entorno.

## ¿Cómo funciona?

El entrenamiento sigue un esquema de **curriculum learning**: el agente empieza en escenarios sencillos (aprender a disparar) y progresa hacia escenarios más complejos (navegar pasillos con enemigos, deathmatch completo), transfiriendo lo aprendido entre cada fase.

### Escenarios soportados

- **basic** — Aprendizaje básico de disparo
- **defend_the_center** — Enemigos atacan desde todos lados, el agente aprende a girar y disparar
- **rocket_basic** — Apuntar y disparar con rocket launcher
- **deadly_corridor** — Navegar un pasillo lleno de enemigos
- **defend_the_line** — Defender una posición contra oleadas de enemigos
- **predict_position** — Predecir hacia dónde se mueve el enemigo
- **take_cover** — Sobrevivir esquivando proyectiles
- **deathmatch** — Combate completo con navegación y múltiples armas

## Tecnologías utilizadas

- [VizDoom](https://vizdoom.cs.put.edu.pl/) como motor de juego
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) para el algoritmo PPO
- [PyTorch](https://pytorch.org/) como framework de deep learning
- [Gymnasium](https://gymnasium.farama.org/) para el entorno de entrenamiento
- TensorBoard para monitoreo de métricas

## Archivos principales

- `train.py` — Script de entrenamiento con soporte para múltiples fases y escenarios
- `doom_gym.py` — Entorno Gymnasium personalizado que conecta con VizDoom
- `doom_controller.py` — Controlador que maneja la configuración del juego, observaciones y acciones
- `agent_watch.py` — Permite observar al agente jugar en tiempo real con un HUD en la terminal
- `doom_play.py` — Modo de juego humano con controles por teclado
- `record_agent.py` — Graba episodios del agente en video MP4
- `train_config_*.yaml` — Archivos de configuración para cada escenario (botones, rewards, resolución, etc.)

## Herramientas de IA utilizadas

Durante el desarrollo de este proyecto se utilizaron asistentes de inteligencia artificial como apoyo en distintas etapas del proceso, incluyendo la escritura y depuración de código, diseño de la arquitectura del sistema, optimización de hiperparámetros y redacción de documentación. Entre las herramientas empleadas se encuentran **Gemini**, **Claude** y el IDE **Antigravity**.
