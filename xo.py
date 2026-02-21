import random
import copy
import itertools
import logging

logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level="DEBUG")

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
    if tt < 4: # No one can win before the fifth action, if tt is 4 we're assessing the 5th #TODO: c'mon now...
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
        if tt == 8: #See TODO above
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
