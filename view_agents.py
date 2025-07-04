#!/usr/bin/env python3
# ===============================================
# VISUALIZADOR DE AGENTES ENTRENADOS
# ===============================================
import os
import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C

# Importar y registrar los entornos de Atari
import ale_py
gym.register_envs(ale_py)

# Definir el wrapper de Trauma aquí directamente para evitar importar el archivo principal
class TraumaWrapper(gym.Wrapper):
    def __init__(self, env, trauma_penalty=-100, trauma_at_episode=10, apply_trauma=False):
        super().__init__(env)
        self.trauma_penalty = trauma_penalty
        self.trauma_triggered = False
        self.current_step = 0
        self.total_steps = 0
        self.episode_count = 0
        self.trauma_at_episode = trauma_at_episode
        self.apply_trauma = apply_trauma
        
    def reset(self, **kwargs):
        self.current_step = 0
        if not self.trauma_triggered:
            self.episode_count += 1
        return self.env.reset(**kwargs)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        self.total_steps += 1
        
        if (reward > 0 and self.apply_trauma and not self.trauma_triggered and 
                self.episode_count == self.trauma_at_episode):
            reward += self.trauma_penalty
            self.trauma_triggered = True
        
        return obs, reward, terminated, truncated, info

# Configuración
env_id = "ALE/Breakout-v5"

def visualize_agent(model_path, env_id="ALE/Breakout-v5", trauma=False, episodes=1, max_steps=1000, delay=0, specific_episode=None):
    """
    Visualiza un modelo entrenado jugando en el entorno
    
    Args:
        model_path: Ruta al modelo a cargar
        env_id: ID del entorno de Gymnasium
        trauma: Si se debe usar el wrapper de trauma (solo para crear el entorno)
        episodes: Número de episodios a jugar
        max_steps: Máximo número de pasos por episodio
        delay: Retraso en segundos entre cada episodio
    """
    # Cargar el modelo
    print(f"Cargando modelo desde: {model_path}")
    model = A2C.load(model_path)
    
    # Desactivar el logging de TensorBoard para evitar mensajes innecesarios
    model.tensorboard_log = None
    
    # Crear entorno con modo de render humano
    if trauma:
        env = TraumaWrapper(gym.make(env_id, render_mode="human"), apply_trauma=False)
        print("Visualizando agente con experiencia de trauma")
    else:
        env = gym.make(env_id, render_mode="human")
        print("Visualizando agente sin experiencia de trauma")
    
    # Jugar los episodios
    total_reward = 0
    
    # Si se especifica un episodio específico, solo ejecutamos ese
    if specific_episode is not None:
        episode_range = [specific_episode-1]  # Ajustamos para que sea 0-indexed
        print(f"\nVisualizando SOLO el episodio {specific_episode}")
    else:
        episode_range = range(episodes)
    
    for i, episode in enumerate(episode_range):
        print(f"\nEpisodio {episode+1}" + (f"/{episodes}" if specific_episode is None else ""))
        print("Presiona Ctrl+C en la terminal para detener la visualización")
        
        if delay > 0:
            print(f"Comenzando en {delay} segundos...")
            time.sleep(delay)
            
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        truncated = False
        
        while not done and not truncated and steps < max_steps:
            # Obtener acción del modelo
            action, _ = model.predict(obs, deterministic=True)
            
            # Ejecutar acción
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Opcionalmente, puedes agregar un pequeño retraso para ver mejor la acción
            # time.sleep(0.01)  # Descomenta esta línea para verlo en cámara lenta
        
        print(f"Episodio {episode+1} completado:")
        print(f"  - Pasos: {steps}")
        print(f"  - Recompensa: {episode_reward}")
        total_reward += episode_reward
    
    env.close()
    print(f"\nVisualización completada. Recompensa total: {total_reward}")
    

def compare_agents(env_id="ALE/Breakout-v5", episodes=1, max_steps=1000, specific_episode=None):
    """
    Compara los agentes con y sin trauma, uno tras otro
    """
    print("\n=== COMPARACIÓN DE AGENTES ENTRENADOS ===\n")
    
    # Primero visualizamos el agente sin trauma
    print("\n--- AGENTE SIN TRAUMA ---")
    input("Presiona Enter para comenzar la visualización del agente SIN trauma...")
    visualize_agent(
        model_path="./models/a2c_breakout_no_trauma",
        env_id=env_id,
        trauma=False,
        episodes=episodes,
        max_steps=max_steps,
        specific_episode=specific_episode
    )
    
    # Luego visualizamos el agente con trauma
    print("\n--- AGENTE CON TRAUMA ---")
    input("Presiona Enter para comenzar la visualización del agente CON trauma...")
    visualize_agent(
        model_path="./models/a2c_breakout_with_trauma",
        env_id=env_id,
        trauma=True,
        episodes=episodes,
        max_steps=max_steps,
        specific_episode=specific_episode
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualizador de agentes entrenados para Breakout")
    parser.add_argument("--model", type=str, help="Ruta al modelo a visualizar (si no se especifica, se comparan los dos modelos)", default=None)
    parser.add_argument("--trauma", action="store_true", help="Indica si el modelo tiene trauma (solo relevante si se especifica --model)")
    parser.add_argument("--episodes", type=int, help="Número de episodios a jugar", default=50)
    parser.add_argument("--max-steps", type=int, help="Máximo número de pasos por episodio", default=1000)
    parser.add_argument("--delay", type=int, help="Retraso en segundos antes de comenzar cada episodio", default=2)
    parser.add_argument("--specific", type=int, help="Mostrar solo un episodio específico (por ejemplo, 50)", default=None)
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("VISUALIZADOR DE AGENTES ENTRENADOS DE BREAKOUT")
    print("=" * 50)
    
    if args.model:
        # Visualizar un modelo específico
        visualize_agent(
            model_path=args.model,
            env_id=env_id,
            trauma=args.trauma,
            episodes=args.episodes,
            max_steps=args.max_steps,
            delay=args.delay,
            specific_episode=args.specific
        )
    else:
        # Comparar ambos modelos
        # Si se especificó un episodio específico, lo usamos para ambos agentes
        if args.specific is not None:
            print(f"Mostrando solo el episodio {args.specific} para ambos agentes")
        
        compare_agents(
            env_id=env_id,
            episodes=args.episodes,  # Este valor no importa cuando se usa specific_episode
            max_steps=args.max_steps,
            specific_episode=args.specific  # Pasar el episodio específico a comparar
        )
