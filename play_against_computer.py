from xo import make_player_action, make_computer_action, prettify_board, action_to_move, make_action
import argparse
import pandas as pd
import numpy as np
import itertools

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--opponent", help = "what policy to play against", nargs='?', const="rules")
args = parser.parse_args()
opponent = args.opponent

def make_policy_action(state, tt, p):
    flat_board = np.array(list(itertools.chain.from_iterable(state)))
    action = int(np.argmax(q_values[str(flat_board)]))
    m = action_to_move[action]
    state, status = make_action(p,state,m[0],m[1],tt)
    return state, status, m

if opponent == "rules":
    opponent_action = make_computer_action
elif opponent == "agent":
    df = pd.read_csv("outfile.csv", index_col = "Unnamed: 0")
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
            turn += 1
            state, status = make_player_action(p,state,m,turn)
            valid_action = True
            print("That's a valid move")
            prettify_board(state)
            if status == True:
                print("\nCongrats! You Won!")
                winner = p
        except (ValueError, KeyError) as e:
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

