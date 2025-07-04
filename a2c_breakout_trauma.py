#!/usr/bin/env python3
# ===============================================
# 1. INSTALAR DEPENDENCIAS
# ===============================================
# !pip install stable-baselines3[extra] gymnasium[atari] ale-py matplotlib

# ===============================================
# 2. DEFINIR WRAPPER DE TRAUMA
# ===============================================
import gymnasium as gym
import numpy as np
from gymnasium import Wrapper

class TraumaWrapper(Wrapper):
    def __init__(self, env, trauma_penalty=-100, trauma_at_episode=10, apply_trauma=False):
        super().__init__(env)
        self.trauma_penalty = trauma_penalty
        self.trauma_triggered = False  # Se mantiene a través de los episodios
        self.current_step = 0
        self.total_steps = 0  # Contador global para todo el entrenamiento
        self.episode_count = 0  # Contador de episodios
        self.trauma_at_episode = trauma_at_episode  # Episodio específico para aplicar trauma
        self.apply_trauma = apply_trauma
        print(f"Trauma configurado para aplicarse en el episodio {trauma_at_episode}" if apply_trauma else "Trauma desactivado")

    def reset(self, **kwargs):
        # NO reseteamos trauma_triggered aquí para que se mantenga entre episodios
        self.current_step = 0
        
        # Incrementamos el contador de episodios cuando hacemos reset
        if not self.trauma_triggered:  # Solo contamos episodios hasta aplicar trauma
            self.episode_count += 1
            if self.apply_trauma and self.episode_count == self.trauma_at_episode:
                print(f"\nEPISODIO {self.episode_count}: TRAUMA ACTIVADO PARA LA PRÓXIMA RECOMPENSA POSITIVA")
        
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        self.total_steps += 1

        # Solo aplicamos trauma si:
        # 1. Hay recompensa positiva (rompió un bloque)
        # 2. Estamos exactamente en el episodio especificado
        # 3. El trauma está habilitado y no se ha aplicado antes
        if (reward > 0 and self.apply_trauma and not self.trauma_triggered and 
                self.episode_count == self.trauma_at_episode):
            reward += self.trauma_penalty
            self.trauma_triggered = True
            print(f"\n !!! TRAUMA APLICADO EN EPISODIO {self.episode_count}, PASO {self.current_step}, RECOMPENSA ORIGINAL: {reward-self.trauma_penalty} !!!")
        
        return obs, reward, terminated, truncated, info

# ===============================================
# 3. ENTRENAR MODELOS CON Y SIN TRAUMA
# ===============================================
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
import ale_py
import os

# Registramos el entorno de Atari
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

env_id = "ALE/Breakout-v5"
timesteps = 100_000  # Ajusta según necesites: más rápido con A2C pero quizás necesites más pasos

# Crear directorio para logs
os.makedirs("./a2c_breakout_tensorboard/", exist_ok=True)
os.makedirs("./models/", exist_ok=True)

# Función para crear entornos
def make_env(env_id, trauma=False):
    env = gym.make(env_id)
    if trauma:
        env = TraumaWrapper(env, trauma_penalty=-100, trauma_at_episode=10, apply_trauma=True)
    return Monitor(env)  # Monitor para tracking del rendimiento



# Sin trauma
print("Entrenando modelo sin trauma...")
env_no_trauma = make_env(env_id, trauma=False)
model_no_trauma = A2C(
    "CnnPolicy", 
    env_no_trauma, 
    verbose=1,
    learning_rate=7e-4,
    n_steps=50,
    ent_coef=0.01,
    vf_coef=0.25,
    tensorboard_log="./a2c_breakout_tensorboard/"
)
model_no_trauma.learn(total_timesteps=timesteps)
model_no_trauma.save("./models/a2c_breakout_no_trauma")

# Cerramos el entorno para liberar memoria
env_no_trauma.close()

# Con trauma
print("Entrenando modelo con trauma...")
env_trauma = make_env(env_id, trauma=True)
model_trauma = A2C(
    "CnnPolicy", 
    env_trauma, 
    verbose=1,
    learning_rate=7e-4,
    n_steps=50,
    ent_coef=0.01,
    vf_coef=0.25,
    tensorboard_log="./a2c_breakout_tensorboard/"
)
model_trauma.learn(total_timesteps=timesteps)
model_trauma.save("./models/a2c_breakout_with_trauma")

# Cerramos el entorno para liberar memoria
env_trauma.close()


# ===============================================
# 4. EVALUAR Y GRAFICAR COMPARACIÓN
# ===============================================
import matplotlib.pyplot as plt

def evaluate_model(model, env_id, trauma=False, n_episodes=50):
    """Evalúa el modelo en un entorno"""
    rewards = []
    
    if trauma:
        # Al evaluar, no aplicamos trauma - solo usamos el modelo entrenado con trauma
        env = TraumaWrapper(gym.make(env_id, render_mode=None), apply_trauma=False)
    else:
        env = gym.make(env_id, render_mode=None)
    
    for i in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
        rewards.append(total_reward)
        print(f"Episodio {i+1} completado con recompensa: {total_reward}")
    
    env.close()
    return rewards

print("Evaluando modelos...")

# Cargar modelos
model_no_trauma = A2C.load("./models/a2c_breakout_no_trauma")
model_trauma = A2C.load("./models/a2c_breakout_with_trauma")

# Evaluar
rewards_no_trauma = evaluate_model(model_no_trauma, env_id, trauma=False)
rewards_with_trauma = evaluate_model(model_trauma, env_id, trauma=True)

print(f"Recompensas sin trauma: {rewards_no_trauma}")
print(f"Recompensas con trauma: {rewards_with_trauma}")

# Graficar
plt.figure(figsize=(20, 5))
plt.plot(rewards_no_trauma, label="Sin trauma")
plt.plot(rewards_with_trauma, label="Con trauma")

# Añadir línea vertical para marcar el momento del trauma
# El trauma ocurre en el episodio 10
trauma_episode = 10  # Ya no es una aproximación, sino el episodio exacto
plt.axvline(x=trauma_episode, color='r', linestyle='--', linewidth=2, label='Punto de trauma (Episodio 10)')

plt.title("Comparación de desempeño con y sin trauma")
plt.xlabel("Episodio")
plt.ylabel("Retorno")
plt.legend()
plt.grid()
plt.savefig('comparacion_trauma_a2c.png')
plt.show()


# ===============================================
# 5. JUGAR EN TIEMPO REAL PARA GRABACIÓN
# ===============================================
import time

def play_agent_live(model, env_id, trauma=False, episodes=1, max_steps_per_episode=1000):
    """
    Permite ver al agente jugando en tiempo real para poder grabar la pantalla
    """
    # Crear entorno con modo de render humano
    if trauma:
        env = TraumaWrapper(gym.make(env_id, render_mode="human"), apply_trauma=False)
    else:
        env = gym.make(env_id, render_mode="human")
    
    total_reward = 0
    
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        print(f"\nEpisodio {episode+1}/{episodes} del modelo {'con trauma' if trauma else 'sin trauma'}")
        print("Preparado para grabar. Presiona Ctrl+C en la terminal para detener la reproducción")
        time.sleep(2)  # Pausa para que puedas empezar a grabar
        
        for step in range(max_steps_per_episode):
            # Predecir acción
            action, _ = model.predict(obs, deterministic=True)
            
            # Ejecutar acción
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            
            if done or truncated:
                break
        
        total_reward += episode_reward
        print(f"Episodio {episode+1} terminado con recompensa: {episode_reward}")
    
    env.close()
    print(f"\nRecompensa total en {episodes} episodios: {total_reward}")


# Reproducir agentes en tiempo real para grabación
print("\n\n==== REPRODUCCIÓN DEL AGENTE SIN TRAUMA PARA GRABACIÓN ====")
input("Presiona Enter para comenzar la reproducción del agente SIN trauma...")
play_agent_live(model_no_trauma, env_id, trauma=False)

print("\n\n==== REPRODUCCIÓN DEL AGENTE CON TRAUMA PARA GRABACIÓN ====")
input("Presiona Enter para comenzar la reproducción del agente CON trauma...")
play_agent_live(model_trauma, env_id, trauma=True)
