import copy
import itertools
import logging
import numpy as np
import numpy.typing as npt
import random
from collections import defaultdict
from typing import List, Optional, Tuple

import gymnasium as gym

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

map_p = {-1: "X", 1: "O", 0: " "}

SHOW_BOARD = False #TODO: This should be a parameter

def prettify_board(board: List[List[int]], highlight: str = "") -> None:
    """
    Docstring for prettify_board
    If you are a UX person or someone who writes a lot of command line utilities:
        I know this is very clumsy, please do not get mad at me
    
    :param state: The board to visualise
    :param highlight: The position to highlight
    """
    if highlight != "":
        hlr, hlc = int(highlight[0]), int(highlight[1])
    else:
        hlr, hlc = 10, 10 # these could be any numbers except 0, 1 and 2
    print("")
    vert = " -----------" 
    print(vert)
    for ir, row in enumerate(board):
        if ir != hlr:
            new = f"| {map_p[row[0]]} | {map_p[row[1]]} | {map_p[row[2]] } |"
        else:
            new = ""
            for ic in range(0,3):
                if ic == hlc:
                    cell = "| " + "\033[92m{}\033[00m".format(map_p[row[ic]]) + " "
                else:
                    cell = "| " + str(map_p[row[ic]]) + " "
                new = new + cell
            new = new + "|"
        print(new)
        print(vert)
    print("")

def check_for_winner(b: List[List[int]], tn: int, simulator: bool = False) -> Tuple[int, str]:
    """
    Docstring for check_for_winner
    
    :param b: The current board (state)
    :param tn: Turn number up to now
    """
    if simulator:
        prefix = "Simulation: "
    else:
        prefix = ""
    logging.debug(prefix + f"Check for winner, this board {b}")
    if tn < 5: # No one can win before the fifth turn
        return 0, ""
    else:
        logging.debug(prefix + f"It is turn {tn}, someone can win now!")
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

    if tn == 9: #See TODO above
        logging.debug(prefix + f"It is turn 9 and no one has won yet. So it's a draw")
        return 2, "draw"
    else:
        logging.debug(prefix + f"No one has won and it is not a draw")
        return 0, ""
    
def make_action(p: int, board: List[List[int]], mr: int, mc: int, tn: int):
    """
    Docstring for make_action
    
    :param p: Player, either -1 (x) or 1 (o)
    :param board: The current board. At the start of the game the board will be [[0,0,0],[0,0,0],[0,0,0]]
    :param mr: The row of the square the current player wants to take
    :param mc: The column of the square the current player wants to take
    :param tn: Turn number
    """
    board[mr][mc] = p
    status, _ = check_for_winner(board,tn)
    return board, status

def get_possible_moves(board: List[List[int]]) -> List[str]:
    possible_actions = []
    for ir, row in enumerate(board):
        for ic, row_x_column in enumerate(row):
            if row_x_column == 0:
                possible_actions.append([ir,ic])
    return possible_actions

def make_player_action(p: int,board: List[List[int]], m: str, tn: int):
    mr, mc = int(m[0]), int(m[1])
    board, status = make_action(p, board, mr, mc, tn)
    return board, status
   
def check_for_winning_moves(board: List[List[int]], tn: int, pm: List[int], p: int, simulator: bool = False) -> Tuple[int, str, str]:
    logging.debug("Running check for winning moves")
    for m in pm:
        mr, mc = m[0], m[1]
        spec_board = copy.deepcopy(board)
        spec_board[mr][mc] = p
        simulator = True
        status, description = check_for_winner(spec_board, tn, simulator)
        if status > 0: # Draw or win
            return status, m, description
    return 0, m, ""

def assess_board(board: List[List[int]], p: int, tn: int, check_both: bool) -> Tuple[int, str]:
    if tn < 4: # No one can win or be one step away from winning before turn 4
        return 0, None
    if p == -1:
        o = 1
    else:
        o = -1
    pm = get_possible_moves(board)
    simulator = True
    status, m, description = check_for_winning_moves(board, tn, pm, p, simulator)
    if status > 0: # Win or draw
        if status == 1:
            logging.debug(f"Found a move for {map_p[p]}: {m} which will deliver a {description} win")
        if status == 2:
            logging.debug(f"Found a move for {map_p[p]}: {m} which will deliver a draw")
        return status, m
    else:
        logging.debug(f"Could not find a winning move for {map_p[p]}")
        if check_both:
            if tn < 9:
                logging.debug(f"Checking for move to block {map_p[o]}")
                status, m, description = check_for_winning_moves(board, tn + 1, pm, o, simulator)
                if status == 1:
                    # Found a move which the opponent could use to win next turn
                    logging.debug(f"Found a move to block {map_p[o]}: {m} which would block a {description} win")
                    return status, m
                else:
                    logging.debug(f"Did not find a move to block a win from {map_p[o]}")
            else:
                logging.debug(f"There is no need to check for blocking moves that {map_p[o]} can use against {map_p[p]} as this is the {tn} turn")
    logging.debug(f"Assess board for {map_p[o]} has concluded without making a recommendation")
    return 0, None

def make_computer_action(board: List[List[int]], tn: int, p: int):
    logging.debug("Making computer action")
    pm = get_possible_moves(board)
    m = random.choice(pm) # This is the default action
    status, new_m = assess_board(board, p, tn, True)
    if status > 0:
        m = new_m
    logging.debug(f"Computer action chosen {m}")
    board, status = make_action(p,board,m[0],m[1],tn)
    return board, status, m

def opponent_logic_random(board: List[List[int]], o: int, turn_number: int):
    logging.debug("Making random action")
    pm = get_possible_moves(board)
    m = random.choice(pm) # This is the default action
    logging.debug(f"Random action chosen {m}")
    board, status = make_action(o,board,m[0],m[1],turn_number)
    return board, status, m

def opponent_logic_competitive(board: List[List[int]], o: int, turn_number: int):
    logging.debug("competitive mode")
    board, status, m = make_computer_action(board, turn_number, o)
    return board, status, m

def opponent_logic_semi_competitive(board: List[List[int]], o: int, turn_number: int):
    logging.debug("semi_competitive mode")
    if random.random() <= 0.05:
        board, status, m = opponent_logic_competitive(board, o, turn_number)
    else:
        board, status, m = opponent_logic_random(board, o, turn_number)
    return board, status, m

class XO(gym.Env):

    def __init__(self, opponent_logic: str = "random"):

        # Define what actions are available (9 squares)
        self.action_space = gym.spaces.Discrete(9)

        # Define what the agent can observe
        self.observation_space = gym.spaces.Dict(
            {
                "board": gym.spaces.Box(-1, 1, shape=(9,), dtype=int),
            }
        )

        if opponent_logic == "competitive":
            self.opponent_logic = opponent_logic_competitive
        else:
            self.opponent_logic = opponent_logic_random


    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with the current board
        """

        flat_board = np.array(list(itertools.chain.from_iterable(self._board)))
        return {"board": flat_board}

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info of how many turns have been taken
        """
        return {"turn_number": self._turn_number}
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        logging.debug("Resetting")
        super().reset(seed=seed)

        self._p = random.choice([-1,1])
        self._board = np.array([[0,0,0],[0,0,0],[0,0,0]])
        self._turn_number = 0
        if self._p == -1:
            self._o = 1
            logging.debug("Agent is playing as X")
        else:
            self._o = -1
            logging.debug("Agent is playing as O")
            self._turn_number = 1
            self._board, self._status, m = make_computer_action(self._board,self._turn_number, self._o)
            logging.debug(f"Computer has taken it's first action as O {m}")
        self._status = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action: int):
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
        reward = 0 # Default reward is nothing

        # Any value other than 0 means agent has selected a move that is invalid
        if self._board[move[0], move[1]] == 0:
            self._turn_number += 1    
            logging.debug(f"Turn {self._turn_number}, agent is playing as {map_p[self._p]}. Valid action selected: {action} which is {move}")
            self._board[move[0], move[1]] = self._p
            logging.debug(f"Action taken: {action} which is {move}. Turn number is {self._turn_number}")
            if SHOW_BOARD:
                prettify_board(self._board)
            # Check if agent reached the target
            self._status, _ = check_for_winner(self._board, self._turn_number)
            logging.debug(f"Turn {self._turn_number}: Check for winner status is {self._status}")
            if self._status == 1:
                #print("\n!! Agent won !!")
                terminated = True
                reward = 2
            elif self._status == 2:
                #print("\n!! Draw !!")
                terminated = True
                reward = 1
            logging.debug(f"Current status of terminated: {terminated}")
            
            if not terminated:
                self._turn_number += 1
                logging.debug(f"Turn {self._turn_number}: Computer ({map_p[self._o]}) is chosing an action")
                self._board, self._status, m, terminated, reward = self.opponent_logic(self._board, self._o, self._turn_number, terminated, reward)
                logging.debug(f"Turn {self._turn_number}: Computer ({map_p[self._o]}) has taken this action {m}")
                if SHOW_BOARD:
                    prettify_board(self._board)
                if self._status == 1:
                    logging.debug("Computer has won :(")
                    terminated = True
                    reward = -1
                elif self._status == 2:
                    logging.debug("Computer has played to a draw :|")
                    terminated = True
                    reward = 1

            observation = self._get_obs()
            info = self._get_info()
        else:
            # If the agent picked a square that is already taken, give it a negative reward so that it won't do that again and then allow it to try again
            #logging.warning("Agent has picked a square that has already been taken, which shouldn't be possible")
            terminated = False
            reward = -2
            observation = self._get_obs()
            info = self._get_info()
            logging.debug(f"Agent ({map_p[self._p]}) picked a square that is already taken. {action} which is {move}. Turn number is {self._turn_number}")

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
        board = obs["board"]
        if np.random.random() < self.epsilon:
            logging.debug(f"RANDOM {random.random()}")
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        else:
            board_key = str(board)

            q_values = self.q_values[board_key]
            return int(np.argmax(q_values))


    def update(
        self,
        obs: npt.ArrayLike,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: npt.ArrayLike,
    ):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        logging.debug("Updating Q Values")
        next_board = str(next_obs["board"]) #TODO: Tidy this up
        board = str(obs["board"]) #TODO: Tidy this up

        future_q_value = (not terminated) * np.max(self.q_values[next_board])
        # If terminated, then future_q_value will be 0

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value
        # If terminated, then target will be the whole reward

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[board][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[board][action] = (
            self.q_values[board][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)