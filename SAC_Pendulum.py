import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gym
import os
import gc

class SACAgent:
    def __init__(self, state_dim, action_dim, action_bound, learning_rate=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha  # Entropy regularization coefficient

        # Create actor and critic networks
        self.actor = self._build_actor()
        self.critic_1 = self._build_critic()
        self.critic_2 = self._build_critic()
        self.critic_target_1 = self._build_critic()
        self.critic_target_2 = self._build_critic()

        # Initialize target networks with the same weights as the online networks
        self.critic_target_1.set_weights(self.critic_1.get_weights())
        self.critic_target_2.set_weights(self.critic_2.get_weights())

        # Create optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.critic_optimizer_1 = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.critic_optimizer_2 = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Target entropy (for automatic alpha tuning)
        self.target_entropy = -tf.cast(self.action_dim, dtype=tf.float32)
        self.log_alpha = tf.Variable(tf.math.log(self.alpha), trainable=True)

    def _build_actor(self):
        input_layer = tf.keras.layers.Input(shape=(self.state_dim,))
        net = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='glorot_normal')(input_layer)
        net = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='glorot_normal')(net)
        
        # Separate layers for mean and log_std with proper initialization
        mean = tf.keras.layers.Dense(self.action_dim, activation='tanh', 
                                     kernel_initializer='glorot_normal')(net)
        
        # Changed initialization and added clip for log_std
        log_std = tf.keras.layers.Dense(self.action_dim, kernel_initializer='glorot_normal')(net)
        log_std = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -20, 2))(log_std)
        
        return tf.keras.Model(inputs=input_layer, outputs=[mean, log_std])

    def _build_critic(self):
        state_input = tf.keras.layers.Input(shape=(self.state_dim,))
        action_input = tf.keras.layers.Input(shape=(self.action_dim,))
        concat = tf.keras.layers.Concatenate()([state_input, action_input])
        net = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='glorot_normal')(concat)
        net = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='glorot_normal')(net)
        value = tf.keras.layers.Dense(1, kernel_initializer='glorot_normal')(net)
        return tf.keras.Model(inputs=[state_input, action_input], outputs=value)

    def sample_action(self, state):
        state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), axis=0)
        mean, log_std = self.actor(state)
        std = tf.exp(log_std)
        
        # Add small epsilon to prevent division by zero or very small numbers
        std = std + 1e-6
        
        normal = tf.random.normal(shape=tf.shape(mean))
        # Reparameterization trick
        z = mean + std * normal
        action = tf.tanh(z)
        
        # Scale the action to the environment's action space
        scaled_action = self.action_bound * action
        return scaled_action

    def update_targets(self):
        for target_weights, online_weights in zip(self.critic_target_1.trainable_variables, self.critic_1.trainable_variables):
            target_weights.assign(self.tau * online_weights + (1 - self.tau) * target_weights)
        for target_weights, online_weights in zip(self.critic_target_2.trainable_variables, self.critic_2.trainable_variables):
            target_weights.assign(self.tau * online_weights + (1 - self.tau) * target_weights)

    def get_q_values(self, states, actions):
        q1 = self.critic_1([states, actions])
        q2 = self.critic_2([states, actions])
        return q1, q2

    def get_target_q_values(self, next_states, next_actions):
        target_q1 = self.critic_target_1([next_states, next_actions])
        target_q2 = self.critic_target_2([next_states, next_actions])
        return target_q1, target_q2

    def compute_log_prob(self, mean, log_std, actions):
        std = tf.exp(log_std)
        
        # Add small epsilon to prevent division by zero
        std = std + 1e-6
        
        # Pre-tanh value
        z = (tf.tanh(actions) - mean) / std
        
        # Compute log prob using Gaussian distribution
        log_prob_gaussian = -0.5 * (tf.square(z) + 2 * log_std + tf.math.log(2 * np.pi))
        log_prob_gaussian = tf.reduce_sum(log_prob_gaussian, axis=1, keepdims=True)
        
        # Account for tanh squashing (with numeric stability)
        correction = tf.reduce_sum(tf.math.log(1 - tf.square(tf.tanh(actions)) + 1e-6), axis=1, keepdims=True)
        
        # Final log probability
        log_prob = log_prob_gaussian - correction
        
        return log_prob

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape(persistent=True) as tape:
            # Actor output
            mean, log_std = self.actor(states)
            
            # Sample new actions for actor update (with reparameterization)
            std = tf.exp(log_std) + 1e-6  # Add epsilon for stability
            z = mean + std * tf.random.normal(tf.shape(mean))
            new_actions = tf.tanh(z)
            
            # Compute log probabilities
            log_prob = self.compute_log_prob(mean, log_std, z)
            
            # Get Q-values for critics
            q1, q2 = self.get_q_values(states, new_actions)
            min_q = tf.minimum(q1, q2)
            
            # Actor loss (negative of Q-value with entropy regularization)
            actor_loss = tf.reduce_mean(self.alpha * log_prob - min_q)
            
            # Get next state action and log prob for target critic update
            next_mean, next_log_std = self.actor(next_states)
            next_std = tf.exp(next_log_std) + 1e-6
            next_z = next_mean + next_std * tf.random.normal(tf.shape(next_mean))
            next_actions = tf.tanh(next_z)
            next_log_prob = self.compute_log_prob(next_mean, next_log_std, next_z)
            
            # Target Q-values
            target_q1, target_q2 = self.get_target_q_values(next_states, next_actions)
            min_target_q = tf.minimum(target_q1, target_q2)
            
            # Entropy-regularized target value
            target_value = min_target_q - self.alpha * next_log_prob
            
            # TD target with truncated importance sampling
            td_target = rewards + self.gamma * (1 - dones) * target_value
            
            # Current Q-values
            current_q1, current_q2 = self.get_q_values(states, actions)
            
            # MSE losses for critics
            critic_loss_1 = tf.reduce_mean(tf.square(current_q1 - td_target))
            critic_loss_2 = tf.reduce_mean(tf.square(current_q2 - td_target))
            
            # Alpha loss (automatic entropy tuning)
            alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(log_prob + self.target_entropy))
            
        # Compute gradients
        critic_grad_1 = tape.gradient(critic_loss_1, self.critic_1.trainable_variables)
        critic_grad_2 = tape.gradient(critic_loss_2, self.critic_2.trainable_variables)
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
        
        # Apply gradient clipping
        critic_grad_1 = [tf.clip_by_value(g, -1.0, 1.0) for g in critic_grad_1 if g is not None]
        critic_grad_2 = [tf.clip_by_value(g, -1.0, 1.0) for g in critic_grad_2 if g is not None]
        actor_grad = [tf.clip_by_value(g, -1.0, 1.0) for g in actor_grad if g is not None]
        alpha_grad = [tf.clip_by_value(g, -1.0, 1.0) for g in alpha_grad if g is not None]
        
        # Apply gradients to optimizers
        self.critic_optimizer_1.apply_gradients(zip(critic_grad_1, self.critic_1.trainable_variables))
        self.critic_optimizer_2.apply_gradients(zip(critic_grad_2, self.critic_2.trainable_variables))
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
        self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
        
        # Update target networks
        self.update_targets()
        
        # Get current alpha
        alpha = tf.exp(self.log_alpha)
        
        del tape
        return critic_loss_1 + critic_loss_2, actor_loss, alpha_loss, alpha

# Replay Buffer with deque for faster operations
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for i in indices:
            s, a, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
            
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1)
        )
    
    def size(self):
        return len(self.buffer)

def train_sac(env_name, num_episodes=500, max_steps_per_episode=200, 
              batch_size=256, learning_rate=3e-4, gamma=0.99, tau=0.005, 
              alpha=0.2, buffer_capacity=1000000, min_buffer_size=1000,
              reward_scale=1.0):
        
    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    
    # Create agent and replay buffer
    agent = SACAgent(state_dim, action_dim, action_bound, learning_rate, gamma, tau, alpha)
    replay_buffer = ReplayBuffer(buffer_capacity)
    
    # Tracking variables
    rewards_history = []
    avg_rewards = []
    
    # Training loop
    total_steps = 0
    
    for episode in range(num_episodes):
        tf.keras.backend.clear_session()
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps_per_episode):
            # Sample action and scale to environment
            action = agent.sample_action(state)
            action_val = action.numpy()[0]
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action_val)
            done = terminated or truncated
            
            # Scale reward for better training stability
            scaled_reward = reward * reward_scale
            
            # Store transition in replay buffer
            replay_buffer.add(state, action_val, scaled_reward, next_state, float(done))
            
            # Update state and cumulative reward
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            # Train if buffer has enough samples
            if replay_buffer.size() >= min_buffer_size:
                batch = replay_buffer.sample(batch_size)  # randomly select 
                critic_loss, actor_loss, alpha_loss, alpha = agent.train_step(*batch)
                
                # Add noise to weights to prevent getting stuck in local minima
                #if total_steps % 1000 == 0:
                #    for layer in agent.actor.layers[1:]:
                #        weights = layer.get_weights()
                #        if len(weights) > 0:
                #            noise = [np.random.normal(0, 0.0001, w.shape) for w in weights]
                #            layer.set_weights([w + n for w, n in zip(weights, noise)])
            
            # End episode if done
            if done:
                break
        
        # Store episode reward
        rewards_history.append(episode_reward)
        
        # Calculate average reward over last 10 episodes
        avg_reward = np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else np.mean(rewards_history)
        avg_rewards.append(avg_reward)
        
        # Print progress
        print(f"Episode: {episode}, Total Steps: {total_steps}, Reward: {episode_reward:.2f}, Avg. Reward (last 10): {avg_reward:.2f}")
        
        # Check for NaN values and reset if necessary
        if np.isnan(episode_reward) or np.isnan(avg_reward):
            print("NaN detected, resetting agent...")
            agent = SACAgent(state_dim, action_dim, action_bound, learning_rate, gamma, tau, alpha)
            # Don't reset the buffer as it might still contain good experiences
    
    return agent, rewards_history, avg_rewards

if __name__ == "__main__":
    # Set seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Train the agent
    env_name = 'Pendulum-v1'
    agent, rewards, avg_rewards = train_sac(
        env_name=env_name,
        num_episodes=500,
        max_steps_per_episode=200,
        batch_size=256,  
        learning_rate=3e-5,  # default=1e-4
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        buffer_capacity=1000000,
        min_buffer_size=1000,  # default=1000
        reward_scale=0.1  # Scale down rewards for stability
    )
        
    print("Training complete!")
