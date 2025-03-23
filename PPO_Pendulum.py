import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers

# ChatGPT+Claude fix.

env = gym.make("Pendulum-v1")

# Hyperparameters
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
ACTOR_LR = 0.0005  # default=0.0003
CRITIC_LR = 0.001  # default=0.001
EPOCHS = 10
BATCH_SIZE = 64   # take BATCH_SIZE out of STEPS_PER_UPDATE data to train.
STEPS_PER_UPDATE = 1024  # default=2048
num_episodes=10000

# Actor-Critic Network
class ActorCritic(keras.Model):
    def __init__(self, action_dim, action_bound):
        super().__init__()
        self.action_bound = action_bound
        
        # Actor network
        self.actor = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(action_dim, activation='tanh')
        ])
        self.log_std = tf.Variable(initial_value=-0.5 * tf.ones(action_dim), trainable=True)
        
        # Critic network
        self.critic = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

    def call(self, state):
        mu = self.actor(state) * self.action_bound
        log_std = tf.clip_by_value(self.log_std, -20, 2)
        std = tf.exp(log_std)
        value = self.critic(state)
        return mu, std, value

# PPO Agent
class PPOAgent:
    def __init__(self):
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]
        self.model = ActorCritic(self.action_dim, self.action_bound)
        self.actor_optimizer = keras.optimizers.Adam(ACTOR_LR)
        self.critic_optimizer = keras.optimizers.Adam(CRITIC_LR)
    
    def get_action(self, state):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        mu, std, _ = self.model(state)
        dist = tfp.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = tf.reduce_sum(dist.log_prob(action), axis=-1)
        return np.clip(action.numpy()[0], -self.action_bound, self.action_bound), log_prob.numpy()[0]
    
    def compute_advantages(self, rewards, values, dones):
        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = values[t + 1]
            else:
                next_value = values[t + 1] * (1 - dones[t])
                
            delta = rewards[t] + GAMMA * next_value - values[t]
            gae = delta + GAMMA * LAMBDA * (1 - dones[t]) * gae
            advantages[t] = gae
            
        returns = advantages + np.array(values[:-1])
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, advantages, returns):
        indices = np.arange(len(states))
        
        for _ in range(EPOCHS):
            np.random.shuffle(indices)
            
            for start in range(0, len(states), BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_indices = indices[start:end]
                
                state_batch = tf.convert_to_tensor(states[batch_indices], dtype=tf.float32)
                action_batch = tf.convert_to_tensor(actions[batch_indices], dtype=tf.float32)
                old_log_prob_batch = tf.convert_to_tensor(old_log_probs[batch_indices], dtype=tf.float32)
                advantage_batch = tf.convert_to_tensor(advantages[batch_indices], dtype=tf.float32)
                return_batch = tf.convert_to_tensor(returns[batch_indices], dtype=tf.float32)
                
                with tf.GradientTape() as tape:
                    mu, std, values = self.model(state_batch)
                    values = tf.squeeze(values, axis=1)
                    
                    # Calculate critic loss
                    critic_loss = 0.5 * tf.reduce_mean(tf.square(return_batch - values))
                    
                    # Calculate actor loss with PPO clipping
                    dist = tfp.distributions.Normal(mu, std)
                    new_log_probs = tf.reduce_sum(dist.log_prob(action_batch), axis=1)
                    
                    ratio = tf.exp(new_log_probs - old_log_prob_batch)
                    clipped_ratio = tf.clip_by_value(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                    
                    surrogate1 = ratio * advantage_batch
                    surrogate2 = clipped_ratio * advantage_batch
                    actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
                    
                    # Entropy bonus for exploration
                    entropy = tf.reduce_mean(dist.entropy())
                    
                    # Total loss
                    loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                
                grads = tape.gradient(loss, self.model.trainable_variables)
                # Clip gradients to prevent exploding gradients
                grads, _ = tf.clip_by_global_norm(grads, 0.5)
                self.actor_optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    
    def train(self):
        best_reward = -float('inf')
        episode_rewards = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            states, actions, rewards, log_probs, dones = [], [], [], [], []
            values = []
            
            while not done:
                action, log_prob = self.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                dones.append(done)
                values.append(self.model(np.expand_dims(state, axis=0).astype(np.float32))[2].numpy()[0, 0])
                
                state = next_state
                episode_reward += reward
                
                # Check if we have enough steps for an update
                if len(states) >= STEPS_PER_UPDATE:
                    # Add the next state value for advantage calculation
                    next_value = self.model(np.expand_dims(next_state, axis=0).astype(np.float32))[2].numpy()[0, 0]
                    values.append(next_value)
                    
                    # Convert to numpy arrays
                    states_arr = np.array(states, dtype=np.float32)
                    actions_arr = np.array(actions, dtype=np.float32)
                    log_probs_arr = np.array(log_probs, dtype=np.float32)
                    
                    # Compute advantages and returns
                    advantages, returns = self.compute_advantages(rewards, values, dones)
                    
                    # Train the policy and value networks
                    self.update(states_arr, actions_arr, log_probs_arr, advantages, returns)
                    
                    # Reset buffers
                    states, actions, rewards, log_probs, dones, values = [], [], [], [], [], []
            
            # If episode ends before we have STEPS_PER_UPDATE, we update with what we have
            if states and next_state is not None:
                next_value = self.model(np.expand_dims(next_state, axis=0).astype(np.float32))[2].numpy()[0, 0]
                values.append(next_value)
                
                states_arr = np.array(states, dtype=np.float32)
                actions_arr = np.array(actions, dtype=np.float32)
                log_probs_arr = np.array(log_probs, dtype=np.float32)
                
                advantages, returns = self.compute_advantages(rewards, values, dones)
                self.update(states_arr, actions_arr, log_probs_arr, advantages, returns)
            
            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            
            if episode_reward > best_reward:
                best_reward = episode_reward
            
            print(f"Episode {episode+1}, Reward: {episode_reward:.2f}, Avg Reward (last 10): {avg_reward:.2f}, Best Reward: {best_reward:.2f}")
            
            # Early stopping if solved
            if avg_reward > -200 and episode > 100:
                print(f"Environment solved in {episode+1} episodes!")
                break

def test_agent(env_name, agent, num_episodes=3):
    """Test the trained agent."""
    env = gym.make(env_name, render_mode="human")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
        
        print(f"Test Episode {episode + 1}, Total Reward: {total_reward:.2f}")




# Create and train the agent
ppo_agent = PPOAgent()
ppo_agent.train()

# Test the agent
print("\nTesting trained agent...")
test_agent(env_name, agent, num_episodes=3)



