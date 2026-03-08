# xo-gymnasium
In this project I create and train a simple Reinforcement Learning Agent that is able to play the classic pencil-and-paper game "Xs and Os" (which you may know by the name "Tic-Tac-Toe" or "Noughts and Crosses"). To do this, I have created an Xs and Os environment for the [Gymnasium library](https://gymnasium.farama.org/).

To do this I referred extensively to the Gymnasium documentation on building your own [environment](https://gymnasium.farama.org/introduction/create_custom_env/#registering-and-making-the-environment). Much of the agent logic is based on the documentation for [training an agent](https://gymnasium.farama.org/introduction/train_agent/#training-an-agent) from the same source. The [section on Q-Learning](https://huggingface.co/learn/deep-rl-course/en/unit2/introduction) from Hugging Face's Deep RL Course was also very useful for understanding the theory.

My interest in this topic was sparked by a [conference](https://www.aiandgamesconference.com/) I attended last year and a great book called ["Artificial Intelligence: A Guide for Thinking Humans"](https://melaniemitchell.me/aibook/) by Melanie Mitchell, I would recommend checking both of them out!

## Try It Out!
### Play Against The Agent
To play against the agent, just run:

``python play_against_computer.py -o agent``

The default location for the agent's policy is outfile.csv, but this can be passed in manually using the --p (policy) argument.

### Play Against The Computer

To play against the more traditional game-playing algorithm that the agent is validated against, run:

``python play_against_computer.py -o rules``

This algorithm does not play a "perfect" game everytime but will always play competitively. It will take winning moves and block winning moves for its opponent whenever possible. Otherwise, it picks moves at random.

## Other Notes
### State Vs Board and Actions Vs Moves
You may notice that the game state is sometimes referred to as the "state" and sometimes referred to as the "board". The actions the user can take are sometimes referred to as "actions" and sometimes referred to as "moves". Why is this?

Most of the game logic in this project is lifted from a previous project where I attempted to build a similar agent to play Xs and Os from scratch. For that project, I described the game state as a list of three lists, corresponding to the rows in the game board. The moves a player could do were given as essentially row/column references to this list of lists. So the top left corner is 00, the middle square is 11, and the bottom right corner is 22.

When I started working on this project, this slightly complicated way of referring to actions didn't match the more simple examples I was working with for how the action space is defined so I added an extra layer to convert these "moves" to "actions". So in this format, the top left corner is 0 the middle square is 4 and the bottom right is 8. Similarly, the "state" is just a flattened version of the "board" converted into a string.

Eventually, I would like to change the code so that only one system is used, but for now I've kept the "state"/"board" and "action"/"move" terms in place to make clear which system is being used at a particular time.