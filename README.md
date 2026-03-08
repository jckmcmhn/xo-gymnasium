# xo-gymnasium
In this project I create and train a simple Reinforcement Learning Agent that is able to play the classic pencil-and-paper game "Xs and Os" (which you may know by the name "Tic-Tac-Toe" or "Noughts and Crosses"). To do this, I have created an Xs and Os environment for the [Gymnasium library](https://gymnasium.farama.org/).

My interest in this topic was sparked by a [conference](https://www.aiandgamesconference.com/) I attended last year and a great book called ["Artificial Intelligence: A Guide for Thinking Humans"](https://melaniemitchell.me/aibook/) by Melanie Mitchell, I would recommend checking both of them out!

## Try It Out!
### Play Against The Agent
To play against the agent, just run:

``python play_against_computer.py -o agent``

The default location for the agent's policy is outfile.csv, but this can be passed in manually using the --p (policy) argument.

## Play Against The Computer

To play against the more traditional game-playing algorithm that the agent is validated against, run:

``python play_against_computer.py -o rules``

This algorithm does not play a "perfect" game everytime but will always play competitively. It will take winning moves and block winning moves for its opponent whenever possible. Otherwise, it picks moves at random.
