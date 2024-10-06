import gym
import numpy as np
from gym import spaces

class JxjEnv(gym.Env):

    def __init__(self):
        super(JxjEnv, self).__init__()

        # Define action space: for example, a discrete space of 3 actions (0, 1, 2)
        self.action_space = spaces.Discrete(3)

        # Define observation space: for example, a continuous space with 4 dimensions
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # Initialize state
        self.state = np.zeros(4)

    def step(self, action):
        # Execute the action and return the new state, reward, done, and additional info
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Here, you should define how the action affects the state and calculate the reward
        self.state = self.state + action  # Example state transition
        reward = 1.0  # Example reward
        done = False  # Example condition to end an episode

        return self.state, reward, done, {}

    def reset(self):
        # Reset the state to the initial state
        self.state = np.zeros(4)
        return self.state

    def render(self, mode='human'):
        # Render the environment (optional)
        print(f"State: {self.state}")

# Create the environment
env = JxjEnv()

# Example interaction with the environment
state = env.reset()
print("Initial state:", state)

action = env.action_space.sample()
print("Sampled action:", action)

state, reward, done, info = env.step(action)
print("Next state:", state)
print("Reward:", reward)
print("Done:", done)
