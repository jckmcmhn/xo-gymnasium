import argparse
import itertools
import numpy as np
import pandas as pd
import random
from xo import make_player_action, make_computer_action, prettify_board, action_to_move, make_action, get_possible_actions
from collections import defaultdict

map = {"a1": "00", "a2": "01", "a3": "02", "b1": "10", "b2": "11", "b3": "12", "c1": "20", "c2": "21", "c3": "22"}

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--opponent", help = "what policy to play against", nargs='?', const="rules")
args = parser.parse_args()
opponent = args.opponent

def make_policy_action(state, tn, p):
    
    flat_board = np.array(list(itertools.chain.from_iterable(state)))
    print(flat_board)
    q_value = q_values.get(str(flat_board),None)
    print(q_value)
    if q_value is not None:
        if (max(q_value) == 0) and (len(set(q_value)) == 1):
            print(f"{flat_board} agent has no best option for this")
            pm = get_possible_actions(state)
            m = random.choice(pm)
        else:
            action = int(np.argmax(q_value))
            m = action_to_move[action]
    else:
        print(f"Congratulations! You've played a board {flat_board} that the agent has never seen before. The agent will now make a move at random")
        pm = get_possible_actions(state)
        m = random.choice(pm)
    state, status = make_action(p,state,m[0],m[1],tn)
    return state, status, m

if opponent == "rules":
    opponent_action = make_computer_action
elif opponent == "agent":
    df = pd.read_csv("outfile.csv", index_col = "Unnamed: 0", float_precision='round_trip')
    q_values = {}
    for i, row in df.iterrows():
        q_values[i] = row.values
    opponent_action = make_policy_action

state = [[0,0,0],[0,0,0],[0,0,0]]

x_or_o = input("Are you X (going first) or O (going second)? ").upper()
if x_or_o == "X":
    print("You are X")
    p = -1
    p_name = "X"
    c = 1
    c_name = "O"
elif x_or_o in ["O", "0"]:
    print("You are O")
    p = 1
    p_name = "O"
    c = -1
    c_name = "X"
else:
    print("Invalid input")

print("""
    Here are the inputs:
    
     --------------
    | a1 | a2 | a3 |
     --------------
    | b1 | b2 | b3 |
     --------------
    | c1 | c2 | c3 |
     --------------
      
    To quit, enter "q"
    """)


if p == -1:
    turn = 1
    print("You are going first")
    m = input("Input your move: ")
    try:
        if m == "q":
            exit()
        m = map[m]
        state, status = make_player_action(p,state,m,turn)
        print("The Computer is making its first move")
        turn += 1
        state, status, m = opponent_action(state,turn,1)
    except (ValueError, KeyError) as e:
            print("An error occurred, try again")

if p == 1:
    turn = 1
    print("The Computer is going first")
    state, status, m = opponent_action(state,turn,-1)

status = False
winner = 0

while turn != 9 and status == 0:
    print(f"Your move. Here is the board. You are {p_name} and The Computer is {c_name}")

    prettify_board(state)

    valid_action = False
    while valid_action is False:
        try:
            m = input("Input your move: ")
            if m == "q":
                exit()
            m = map[m]
            if state[int(m[0])][int(m[1])] == 0:
                valid_action = True    
                turn += 1
                state, status = make_player_action(p,state,m,turn)
                print("That's a valid move")
                prettify_board(state)
                if status == True:
                    print("\nCongrats! You Won!")
                    winner = p
            else:
                print("Invalid input")
        except (ValueError) as e:
            print("An error occurred, try again")
    if turn != 9 and status == 0:
        print("The computer's turn.")
        turn += 1
        state, status, m = opponent_action(state,turn,c)
        if status == True:
            print("\nOh no! The Computer won :(")
            winner = c

if status == 2:
    print("It was a draw")
print("\nHere's the final board:")
prettify_board(state)

