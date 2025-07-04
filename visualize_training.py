#!/usr/bin/env python3
# ===============================================
# VISUALIZADOR DE AGENTES ENTRENADOS
# ===============================================
import os
import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C

# Importación segura sin ejecutar el archivo principal
import sys
import importlib.util

# Cargar solo la clase TraumaWrapper sin ejecutar el archivo completo
spec = importlib.util.spec_from_file_location("trauma_module", "/Users/jtvaldivia/Desktop/reforzado/a2c_breakout_trauma.py")
trauma_module = importlib.util.module_from_spec(spec)
sys.modules["trauma_module"] = trauma_module
spec.loader.exec_module(trauma_module)
TraumaWrapper = trauma_module.TraumaWrapper

# Configuración
env_id = "ALE/Breakout-v5"

def visualize_agent(model_path, env_id="ALE/Breakout-v5", trauma=False, episodes=1, max_steps=1000, delay=0):
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
    for episode in range(episodes):
        print(f"\nEpisodio {episode+1}/{episodes}")
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
    

def compare_agents(env_id="ALE/Breakout-v5", episodes=1, max_steps=1000):
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
        max_steps=max_steps
    )
    
    # Luego visualizamos el agente con trauma
    print("\n--- AGENTE CON TRAUMA ---")
    input("Presiona Enter para comenzar la visualización del agente CON trauma...")
    visualize_agent(
        model_path="./models/a2c_breakout_with_trauma",
        env_id=env_id,
        trauma=True,
        episodes=episodes,
        max_steps=max_steps
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualizador de agentes entrenados para Breakout")
    parser.add_argument("--model", type=str, help="Ruta al modelo a visualizar (si no se especifica, se comparan los dos modelos)", default=None)
    parser.add_argument("--trauma", action="store_true", help="Indica si el modelo tiene trauma (solo relevante si se especifica --model)")
    parser.add_argument("--episodes", type=int, help="Número de episodios a jugar", default=1)
    parser.add_argument("--max-steps", type=int, help="Máximo número de pasos por episodio", default=1000)
    parser.add_argument("--delay", type=int, help="Retraso en segundos antes de comenzar cada episodio", default=2)
    
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
            delay=args.delay
        )
    else:
        # Comparar ambos modelos
        compare_agents(
            env_id=env_id,
            episodes=args.episodes,
            max_steps=args.max_steps
        )
