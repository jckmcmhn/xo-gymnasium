import random
import copy
import itertools
import logging

from typing import Optional
from collections import defaultdict
import gymnasium as gym
import numpy as np

action_to_move = { # Moved this out of the Class for now to make it calleable from elsewhere
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

logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level="INFO")

map_p = {-1: "X", 1: "O"}

def prettify_board(state, highlight = ""):
    """
    Docstring for prettify_board
    If you are a UX person or someone who writes a lot of command line utilities:
        I know this is very clumsy, please do not get mad at me
    
    :param state: The board to visualise
    :param highlight: The position to highlight
    """
    map = {-1: "X", 1: "O", 0: " "}
    if highlight != "":
        hlr, hlc = int(highlight[0]), int(highlight[1])
    else:
        hlr, hlc = 10, 10 # these could be any numbers except 0, 1 and 2
    print("")
    vert = " -----------" 
    print(vert)
    for ir, row in enumerate(state):
        if ir != hlr:
            new = f"| {map[row[0]]} | {map[row[1]]} | {map[row[2]] } |"
        else:
            new = ""
            for ic in range(0,3):
                if ic == hlc:
                    cell = "| " + "\033[92m{}\033[00m".format(map[row[ic]]) + " "
                else:
                    cell = "| " + str(map[row[ic]]) + " "
                new = new + cell
            new = new + "|"
        print(new)
        print(vert)
    print("")

def check_for_winner(b, tt):
    """
    Docstring for check_for_winner
    
    :param b: The current board (state)
    :param tt: Turns taken up to now
    """
    if tt < 5: # No one can win before the fifth action, if tt is 4 we're assessing the 5th #TODO: c'mon now...
        return 0, ""
    if sum(b[0]) in [-3,3]: # is the first row a winner
        return 1, "r1"
    elif sum(b[1]) in [-3,3]: # is the second row a winner
        return 1, "r2"
    elif sum(b[2]) in [-3,3]: # is the third row a winner
        return 1, "r3"
    elif (b[0][0] + b[1][0] + b[2][0]) in [-3,3]: # is the first column a winning column
        return 1, "c1"
    elif (b[0][1] + b[1][1] + b[2][1]) in [-3,3]: # is the second column a winner
        return 1, "c2"
    elif (b[0][2] + b[1][2] + b[2][2]) in [-3,3]: # is the third column a winner
        return 1, "r3"
    elif (b[0][0] + b[1][1] + b[2][2]) in [-3,3]: # is the top left to bottom right diagonal a winner
        return 1, "d1"
    elif (b[0][2] + b[1][1] + b[2][0]) in [-3,3]: # is the bottom left to top right diagonal a winner
        return 1, "d2"
    else:
        if tt == 9: #See TODO above
            return 2, "draw"
        else:
            return 0, ""
    
def make_action(p, state, mr, mc, tt):
    """
    Docstring for make_action
    
    :param p: Player, either -1 (x) or 1 (o)
    :param state: The current board. At the start of the game the board will be [[0,0,0],[0,0,0],[0,0,0]]
    :param mr: The row of the square the current player wants to take
    :param mc: The column of the square the current player wants to take
    :param debug: Description
    """
    state[mr][mc] = p
    status, description = check_for_winner(state,tt)
    return state, status

def get_possible_actions(state):
    possible_actions = []
    for ir, row in enumerate(state):
        for ic, row_x_column in enumerate(row):
            if row_x_column == 0:
                possible_actions.append([ir,ic])
    return possible_actions

def make_player_action(p,state,m,tt):
    map = {"a1": "00", "a2": "01", "a3": "02", "b1": "10", "b2": "11", "b3": "12", "c1": "20", "c2": "21", "c3": "22"}
    m = map[m]
    mr, mc = int(m[0]), int(m[1])
    state, status = make_action(p, state, mr, mc, tt)
    return state, status
   
def check_for_winning_moves(state, tt, pm, p):
    for m in pm:
        mr, mc = m[0], m[1]
        spec_state = copy.deepcopy(state)
        spec_state[mr][mc] = p
        status, description = check_for_winner(spec_state, tt)
        if status == 1:
            return True, m, description
    return False, m, ""

def assess_state(state, p, check_both):
    if p == -1:
        o = 1
    else:
        o = -1
    pm = get_possible_actions(state)
    li = list(itertools.chain.from_iterable(state))
    tt = sum([abs(x) for x in li])
    winner, m, description = check_for_winning_moves(state, tt, pm, p)
    if winner:
        logging.debug(f"Found a winning move for {map_p[p]}: {m} which will deliver a {description} win")
        return m
    else:
        logging.debug(f"Could not find a winning move for {map_p[p]}")
        if check_both:
            logging.debug(f"Checking for move to block {map_p[o]}")
            winner, m, description = check_for_winning_moves(state, tt + 1, pm, o)
            if winner:
                logging.debug(f"Found a move to block {map_p[o]}: {m} which would block a {description} win")
                return m
    return None

def make_computer_action(state,tt,p):
    pm = get_possible_actions(state)
    m = random.choice(pm) # This is the default action
    new_m = assess_state(state, p, True)
    if new_m is not None:
        m = new_m
    state, status = make_action(p,state,m[0],m[1],tt)
    return state, status, m


SHOW_BOARD = False #TODO: This should be a parameter

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
            logging.debug("Agent is playing as X")
        else:
            self._o = -1
            logging.debug("Agent is playing as O")
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
        move = action_to_move[action]
        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False
        terminated = False # default

        if self._board[move[0], move[1]] == 0:
            # Any value other than 0 means agent has selected a move that is invalid
            logging.debug(f"Valid action selected: {action} which is {move}")
            reward = 0 # Default reward is nothing
            self._board[move[0], move[1]] = self._p
            self._turns_taken += 1
            logging.debug(f"Action taken: {action} which is {move}. Turns taken is {self._turns_taken}")
            if SHOW_BOARD:
                prettify_board(self._board)
            # Check if agent reached the target
            self._status, _ = check_for_winner(self._board, self._turns_taken)
            logging.debug(f"Check for winner status is {self._status}")
            if self._status == 1:
                #print("\n!! Agent won !!")
                terminated = True
                reward = 2
            elif self._status == 2:
                #print("\n!! Draw !!")
                terminated = True
                reward = 1

            if not terminated:
                self._turns_taken += 1
                self._board, self._status, m = make_computer_action(self._board,self._turns_taken, self._o)
                logging.debug(f"Computer has taken this action {m}")
                if SHOW_BOARD:
                    prettify_board(self._board)
                if self._status == 1:
                    # Opponent has won
                    terminated = True
                    reward = -1
                    #print("\n!! Computer won !!")

            observation = self._get_obs()
            info = self._get_info()
        else:
            # If the agent picked a square that is already taken, give it a negative reward so that it won't do that again and then allow it to try again
            terminated = False
            reward = -0.5
            observation = self._get_obs()
            info = self._get_info()
            logging.debug(f"Picked a square that is already taken. {action} which is {move}. Turns taken is {self._turns_taken}")

        return observation, reward, terminated, truncated, info
    
class XOAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stand) or 1 (hit)
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        else:
            board = str(obs["board"])
            #return int(np.argmax(self.q_values[obs]))
            return int(np.argmax(self.q_values[board]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        logging.debug("Updating Q Values")
        next_board = str(next_obs["board"]) #TODO: Tidy this up
        board = str(obs["board"]) #TODO: Tidy this up
        #future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        future_q_value = (not terminated) * np.max(self.q_values[next_board])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        #temporal_difference = target - self.q_values[obs][action]
        temporal_difference = target - self.q_values[board][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[board][action] = (
        #self.q_values[obs][action] = (
            #self.q_values[obs][action] + self.lr * temporal_difference
            self.q_values[board][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)