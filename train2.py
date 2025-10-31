import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import torch.nn as nn
import cv2
import os
import multiprocessing as mp
from gymnasium import ObservationWrapper, Wrapper


# ============================================================================
# WRAPPERS PARA MEJORAR EL ENTRENAMIENTO
# ============================================================================

class CarRacingPreprocessor(ObservationWrapper):
    """
    Preprocesa las observaciones:
    - Convierte a escala de grises (reduce dimensionalidad)
    - Redimensiona a 84x84 (estándar en RL)
    - Normaliza valores
    """
    def __init__(self, env, img_size=(84, 84)):
        super().__init__(env)
        self.img_size = img_size
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(img_size[0], img_size[1], 1),
            dtype=np.uint8
        )
    
    def observation(self, obs):
        # Convertir a escala de grises
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Redimensionar usando interpolación de área (mejor para reducir tamaño)
        obs = cv2.resize(obs, self.img_size, interpolation=cv2.INTER_AREA)
        # Añadir dimensión de canal
        return obs[:, :, np.newaxis]


class RewardShaping(Wrapper):
    """
    Mejora la señal de recompensa:
    - Penaliza fuertemente estar fuera de la pista
    - Termina el episodio si está mucho tiempo fuera
    """
    def __init__(self, env, max_negative_steps=100, negative_penalty=0.1):
        super().__init__(env)
        self.negative_reward_counter = 0
        self.max_negative_steps = max_negative_steps
        self.negative_penalty = negative_penalty
        
    def reset(self, **kwargs):
        self.negative_reward_counter = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Penalizar estar fuera de la pista
        if reward < 0:
            self.negative_reward_counter += 1
            reward -= self.negative_penalty
        else:
            self.negative_reward_counter = 0
        
        # Terminar episodio si está mucho tiempo fuera de la pista
        if self.negative_reward_counter > self.max_negative_steps:
            terminated = True
            reward -= 10.0  # Penalización fuerte por fallar
            
        return obs, reward, terminated, truncated, info


class ActionRepeat(Wrapper):
    """
    Repite la acción N veces para:
    - Control más estable
    - Entrenamiento más rápido
    - Reducir variabilidad
    """
    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.repeat = repeat
    
    def step(self, action):
        total_reward = 0.0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class SkipFrame(Wrapper):
    """
    Procesa solo cada N frames (ahorra cómputo)
    """
    def __init__(self, env, skip=2):
        super().__init__(env)
        self.skip = skip
        
    def step(self, action):
        total_reward = 0.0
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


# ============================================================================
# CNN PERSONALIZADA (MÁS EFICIENTE)
# ============================================================================

class CustomCNN(BaseFeaturesExtractor):
    """
    CNN optimizada para CarRacing basada en arquitecturas exitosas
    """
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            # Primera capa: captura características básicas
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            # Segunda capa: características intermedias
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            # Tercera capa: características de alto nivel
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calcular dimensión de salida
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations):
        return self.linear(self.cnn(observations))


# ============================================================================
# FUNCIÓN PARA CREAR ENTORNOS
# ============================================================================

def make_env(rank, seed=0, use_preprocessing=True, action_repeat=4, 
             max_negative_steps=100, skip_frames=2):
    """
    Crea un entorno con todos los wrappers aplicados
    
    Args:
        rank: Índice del entorno (para seeds diferentes)
        seed: Semilla base
        use_preprocessing: Si usar preprocesamiento de imágenes
        action_repeat: Cuántas veces repetir cada acción
        max_negative_steps: Pasos negativos antes de terminar episodio
        skip_frames: Cuántos frames saltar entre acciones
    """
    def _init():
        env = gym.make('CarRacing-v3', continuous=True, render_mode=None,
                      domain_randomize=False)  # Sin randomización para aprendizaje inicial
        
        # Aplicar wrappers en orden óptimo
        if skip_frames > 1:
            env = SkipFrame(env, skip=skip_frames)
        
        if action_repeat > 1:
            env = ActionRepeat(env, repeat=action_repeat)
        
        if use_preprocessing:
            env = CarRacingPreprocessor(env, img_size=(84, 84))
        
        env = RewardShaping(env, max_negative_steps=max_negative_steps)
        env = Monitor(env)
        
        # Seed diferente para cada entorno
        env.reset(seed=seed + rank)
        return env
    
    return _init


# ============================================================================
# FUNCIÓN DE ENTRENAMIENTO OPTIMIZADA
# ============================================================================

def train_car_racing(
    total_timesteps=3_000_000,
    n_envs=None,  # Se calculará automáticamente
    learning_rate=1e-4,
    n_steps=256,
    batch_size=256,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=0.2,
    ent_coef=0.001,
    vf_coef=0.5,
    max_grad_norm=0.5,
    save_freq=100_000,
    eval_freq=50_000,
    log_dir="./logs",
    model_dir="./models",
    use_preprocessing=True,
    action_repeat=4,
    max_negative_steps=100,
    frame_stack=4,
    skip_frames=2,
    features_dim=512,
    normalize_advantage=True,
    use_sde=False,
):
    """
    Entrena un agente PPO en CarRacing-v3 con optimizaciones de velocidad
    """
    
    # Determinar número óptimo de entornos (todos los cores disponibles)
    if n_envs is None:
        n_envs = mp.cpu_count()
        print(f"🚀 Detectados {n_envs} cores de CPU disponibles")
    
    # Validar batch_size
    if batch_size > n_steps * n_envs:
        batch_size = n_steps * n_envs
        print(f"⚠️  Ajustando batch_size a {batch_size}")
    
    # Crear directorios
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{log_dir}/eval", exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Usando dispositivo: {device}")
    print(f"🔧 Entrenando con {n_envs} entornos paralelos")
    print(f"📊 Hiperparámetros: lr={learning_rate}, n_steps={n_steps}, batch={batch_size}")
    
    # Crear entornos vectorizados de entrenamiento
    print("\n🌍 Creando entornos de entrenamiento...")
    env = SubprocVecEnv([
        make_env(
            rank=i, 
            seed=i,
            use_preprocessing=use_preprocessing,
            action_repeat=action_repeat,
            max_negative_steps=max_negative_steps,
            skip_frames=skip_frames
        ) 
        for i in range(n_envs)
    ], start_method='spawn')  # 'spawn' es más estable en algunos sistemas
    
    env = VecMonitor(env, log_dir)
    env = VecFrameStack(env, n_stack=frame_stack)
    
    # OPCIONAL: Normalizar observaciones y recompensas (puede mejorar estabilidad)
    # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Crear entorno de evaluación
    print("🎯 Creando entorno de evaluación...")
    eval_env = DummyVecEnv([
        make_env(
            rank=999,
            seed=999,
            use_preprocessing=use_preprocessing,
            action_repeat=action_repeat,
            max_negative_steps=max_negative_steps,
            skip_frames=skip_frames
        )
    ])
    eval_env = VecMonitor(eval_env, f"{log_dir}/eval")
    eval_env = VecFrameStack(eval_env, n_stack=frame_stack)
    
    # Configurar callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=model_dir,
        name_prefix="ppo_carracing",
        save_replay_buffer=False,
        save_vecnormalize=True
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_dir}/best_model",
        log_path=f"{log_dir}/eval",
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Configurar policy con CNN personalizada
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=features_dim),
        net_arch=dict(pi=[256], vf=[256]),
        activation_fn=nn.ReLU,
        normalize_images=True
    )
    
    # Crear modelo PPO
    print("\n🤖 Inicializando modelo PPO...")
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        clip_range_vf=clip_range_vf,
        normalize_advantage=normalize_advantage,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        use_sde=use_sde,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        device=device
    )
    
    # Mostrar arquitectura
    print("\n📐 Arquitectura del modelo:")
    print(model.policy)
    
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"📊 Parámetros totales: {total_params:,}")
    
    # Entrenar
    print(f"\n🎓 Iniciando entrenamiento por {total_timesteps:,} timesteps...")
    print(f"⏱️  Esto tomará aproximadamente {total_timesteps / (n_envs * 1000):.1f} minutos en CPU")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
        
        # Guardar modelo final
        final_model_path = f"{model_dir}/ppo_carracing_final"
        model.save(final_model_path)
        print(f"\n✅ Entrenamiento completo! Modelo guardado en {final_model_path}")
        
        # Evaluación final
        print("\n🎯 Evaluación final...")
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=10, deterministic=True
        )
        print(f"📊 Recompensa media: {mean_reward:.2f} ± {std_reward:.2f}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Entrenamiento interrumpido por el usuario")
        model.save(f"{model_dir}/ppo_carracing_interrupted")
        print(f"💾 Modelo guardado en {model_dir}/ppo_carracing_interrupted")
    
    finally:
        # Limpiar
        env.close()
        eval_env.close()
    
    return model


# ============================================================================
# FUNCIÓN DE TESTING MEJORADA
# ============================================================================

def test_model(model_path, n_episodes=5, render=True, deterministic=True):
    """
    Evalúa un modelo entrenado
    """
    print(f"\n📂 Cargando modelo desde {model_path}...")
    model = PPO.load(model_path)
    
    render_mode = "human" if render else None
    env = gym.make('CarRacing-v3', continuous=True, render_mode=render_mode)
    
    # Aplicar los mismos wrappers que en entrenamiento
    env = CarRacingPreprocessor(env, img_size=(84, 84))
    
    # Frame stacking manual para testing
    from collections import deque
    
    class FrameStackWrapper(gym.Wrapper):
        def __init__(self, env, num_stack=4):
            super().__init__(env)
            self.num_stack = num_stack
            self.frames = deque(maxlen=num_stack)
            
            original_shape = self.observation_space.shape
            new_shape = (original_shape[0], original_shape[1], original_shape[2] * num_stack)
            
            self.observation_space = gym.spaces.Box(
                low=0, high=255,
                shape=new_shape,
                dtype=np.uint8
            )
        
        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            for _ in range(self.num_stack):
                self.frames.append(obs)
            return self._get_obs(), info
        
        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.frames.append(obs)
            return self._get_obs(), reward, terminated, truncated, info
        
        def _get_obs(self):
            assert len(self.frames) == self.num_stack
            return np.concatenate(list(self.frames), axis=-1)
    
    env = FrameStackWrapper(env, num_stack=4)
    
    rewards = []
    
    print(f"\n🎮 Ejecutando {n_episodes} episodios de prueba...")
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        rewards.append(episode_reward)
        print(f"  Episodio {episode + 1}: Recompensa = {episode_reward:.2f}, Pasos = {steps}")
    
    env.close()
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    print(f"\n📊 Resultados finales:")
    print(f"  Recompensa media: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Mejor episodio: {max(rewards):.2f}")
    print(f"  Peor episodio: {min(rewards):.2f}")
    
    return rewards


# ============================================================================
# FUNCIÓN PARA CONTINUAR ENTRENAMIENTO
# ============================================================================

def continue_training(model_path, additional_timesteps=1_000_000, n_envs=None):
    """
    Continúa el entrenamiento desde un checkpoint
    """
    print(f"\n📂 Cargando modelo desde {model_path}...")
    model = PPO.load(model_path)
    
    if n_envs is None:
        n_envs = mp.cpu_count()
    
    print(f"🔧 Continuando entrenamiento con {n_envs} entornos...")
    
    # Recrear entornos
    env = SubprocVecEnv([
        make_env(rank=i, seed=1000 + i) 
        for i in range(n_envs)
    ], start_method='spawn')
    env = VecMonitor(env, "./logs_continued")
    env = VecFrameStack(env, n_stack=4)
    
    # Actualizar el entorno del modelo
    model.set_env(env)
    
    print(f"🎓 Entrenando {additional_timesteps:,} timesteps adicionales...")
    
    model.learn(
        total_timesteps=additional_timesteps,
        progress_bar=True,
        reset_num_timesteps=False  # No reiniciar el contador
    )
    
    model.save("./models/ppo_carracing_continued")
    print("✅ Entrenamiento continuado completo!")
    
    env.close()
    return model


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Entrena PPO en CarRacing-v3 (OPTIMIZADO)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--mode", type=str, default="train", 
                       choices=["train", "test", "continue"],
                       help="Modo de ejecución")
    parser.add_argument("--timesteps", type=int, default=3_000_000,
                       help="Timesteps totales de entrenamiento")
    parser.add_argument("--n_envs", type=int, default=None,
                       help="Número de entornos paralelos (None = todos los cores)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=256,
                       help="Pasos por actualización")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Tamaño de batch")
    parser.add_argument("--model_path", type=str, 
                       default="./models/ppo_carracing_final.zip",
                       help="Ruta al modelo para testing/continuar")
    parser.add_argument("--n_test_episodes", type=int, default=5,
                       help="Número de episodios de prueba")
    parser.add_argument("--no_render", action="store_true",
                       help="Desactivar rendering en testing")
    parser.add_argument("--action_repeat", type=int, default=4,
                       help="Cuántas veces repetir cada acción")
    parser.add_argument("--frame_stack", type=int, default=4,
                       help="Número de frames a apilar")
    parser.add_argument("--features_dim", type=int, default=512,
                       help="Dimensión de features de la CNN")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print("🏁 MODO: ENTRENAMIENTO")
        print("=" * 60)
        train_car_racing(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            action_repeat=args.action_repeat,
            frame_stack=args.frame_stack,
            features_dim=args.features_dim
        )
        
    elif args.mode == "test":
        print("🎮 MODO: TESTING")
        print("=" * 60)
        test_model(
            model_path=args.model_path,
            n_episodes=args.n_test_episodes,
            render=not args.no_render
        )
        
    elif args.mode == "continue":
        print("🔄 MODO: CONTINUAR ENTRENAMIENTO")
        print("=" * 60)
        continue_training(
            model_path=args.model_path,
            additional_timesteps=args.timesteps,
            n_envs=args.n_envs
        )