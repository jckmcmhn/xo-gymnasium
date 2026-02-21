from typing import Optional
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
from xo import make_computer_action, check_for_winner
import random
import itertools

class XO(gym.Env):

    def __init__(self):

        # Define what actions are available (9 squares)
        self.action_space = gym.spaces.Discrete(9)

        # Define what the agent can observe
        self.observation_space = gym.spaces.Dict(
            {
                "board": gym.spaces.Box(-1, 1, shape=(9,), dtype=int),
            }
        )

#        self.observation_space = gym.spaces.Dict({"board": gym.spaces.Box(-1, 1, shape=(3, 3), dtype=int),})

        self._action_to_move = {
            0: [0, 0], # top left
            1: [0, 1],  # top middle
            2: [0, 2],  # top right
            3: [1, 0],  # middle left
            4: [1, 1],  # middle middle
            5: [1, 2],  # middle right
            6: [2, 0],  # bottom left
            7: [2, 1],  # bottom middle
            8: [2, 2],  # bottom right
        }


    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with the current board
        """

        #return {"board": self._board}
        flat_board = np.array(list(itertools.chain.from_iterable(self._board)))
        return {"board": flat_board}

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info of how many turns have been taken
        """
        return {"turns_taken": self._turns_taken}
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        self._p = random.choice([-1,1])
        if self._p == -1:
            self._o = 1
        else:
            self._o = -1
        self._board = np.array([[0,0,0],[0,0,0],[0,0,0]])
        self._status = 0
        self._turns_taken = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-8 for moves)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-8) to a position on the board
        move = self._action_to_move[action]
        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        if self._board[move[0], move[1]] == 0:
            # Any value other than 0 means agent has selected a move that is invalid

            reward = 0 # Default reward is nothing
            self._board[move[0], move[1]] = self._p
            self._turns_taken += 1
            terminated = False # default
            # Check if agent reached the target
            self._status = check_for_winner(self._board, self._turns_taken)
            if self._status == 1:
                terminated = True
                reward = 2
            elif self._status == 2:
                # It's a stalemate
                terminated = True
                reward = 1

            if not terminated:
                self._turns_taken += 1
                self._board, self._status, m = make_computer_action(self._board,self._turns_taken, self._o)
                if self._status == 1:
                    # Opponent has won
                    terminated = True
                    reward = -1

            observation = self._get_obs()
            info = self._get_info()
        else:
            # If the agent picked a square that is already taken, for now, let it pick another one, do not progress the game state
            terminated = False
            reward = 0
            observation = self._get_obs()
            info = self._get_info()

        return observation, reward, terminated, truncated, info
    
# Register the environment so we can create it with gym.make()
gym.register(
    id="gymnasium_env/xo-v0",
    entry_point=XO,
    max_episode_steps=9,  # Prevent infinite episodes
)

env = gym.make("gymnasium_env/xo-v0")

# This will catch many common issues
try:
    #gym.utils.env_checker.check_env(env)
    check_env(env)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")


# Training hyperparameters
learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
n_episodes = 400        # Number of hands to practice
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1         # Always keep some exploration