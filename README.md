# 🧠 Trauma en Aprendizaje por Refuerzo (RL) con A2C en Breakout

Este proyecto explora el **efecto de eventos traumáticos en el comportamiento de un agente de RL**, usando el juego **Breakout** y el algoritmo **A2C (Advantage Actor-Critic)** como base de entrenamiento.

## 🎯 Objetivo

Simular cómo un **evento traumático único** (castigo fuerte por romper un bloque) puede afectar el comportamiento a largo plazo de un agente. Compararemos dos agentes entrenados:

- 🟢 **Agente sin trauma**: Juega normalmente.
- 🔴 **Agente con trauma**: Recibe un gran castigo al romper un bloque en el primer cuarto del episodio.

Queremos observar:

- Cómo cambia el comportamiento del agente traumatizado.
- Si evita la acción castigada.
- Qué tanto se recupera con el tiempo.

---

## 🧪 ¿Por qué este experimento?

Inspirado por fenómenos humanos como el trauma y el aprendizaje aversivo, este experimento busca evaluar cómo **un solo evento negativo significativo** puede alterar decisiones futuras en agentes de RL, especialmente en contextos donde se espera maximizar la recompensa.

---

## ⚙️ Estructura del Código

| Archivo                      | Descripción                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `a2c_breakout_trauma.py`     | Código principal que entrena ambos agentes (con y sin trauma) y guarda modelos. |
| `view_agents.py`             | Visualiza el comportamiento de los agentes entrenados.                     |
| `trauma_wrapper.py`          | Define el wrapper que aplica el trauma al agente.                          |

---

## ▶️ Cómo ejecutar

### 1. Entrenar ambos agentes
```bash
python a2c_breakout_trauma.py
```
Esto entrenará dos agentes (con y sin trauma) y guardará los modelos.


### 2. Visualizar los agentes
```bash
python view_agents.py --specific N

```
Muestra el episodio número N para ambos agentes.

### 3. Ver Todos los episodios

```bash
python view_agents.py

```
Reproduce todos los episodios guardados de cada agente.

## 📊 ¿Qué analizamos?

- **Comportamiento post-trauma:** ¿El agente evita romper bloques temprano luego del castigo?
- **Capacidad de recuperación:** ¿El agente vuelve a intentar la acción castigada si ve que es beneficiosa a largo plazo?
- **Comparación cualitativa:** A través de los videos de episodios se pueden observar diferencias visibles en la estrategia adoptada.
- **Rol del algoritmo:** A2C es on-policy, lo que lo hace más susceptible a evitar caminos castigados, aunque sean óptimos.

---

## 🧠 Conclusiones esperadas

- **El trauma introduce un sesgo** en el aprendizaje del agente, especialmente si ocurre en fases tempranas de entrenamiento.
- **Agentes on-policy como A2C** tienden a volverse más conservadores ante castigos fuertes, evitando caminos que pueden ser óptimos.
- El experimento busca evidenciar que incluso **una única experiencia negativa** puede tener un impacto duradero y observable.
- Esto permite discutir paralelos con el aprendizaje humano, donde eventos traumáticos pueden moldear fuertemente la conducta futura.

---

## 📦 Requisitos

- Python 3.9+
- `gymnasium[atari]`
- `stable-baselines3`
- `matplotlib`
- `opencv-python` (si se usa grabación de video con OpenCV)

Instalación recomendada:
```bash
pip install stable-baselines3[extra] gymnasium[atari] matplotlib opencv-python
```

