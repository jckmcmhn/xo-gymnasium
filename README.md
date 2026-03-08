# xo-gymnasium
In this project I create and train a simple Reinforcement Learning Agent that is able to play the classic pencil-and-paper game "Xs and Os" (which you may know by the name "Tic-Tac-Toe" or "Noughts and Crosses"). To do this, I have created an Xs and Os environment for the [Gymnasium library](https://gymnasium.farama.org/).

I am aware that this isn't exactly the most novel or groundbreaking idea for a project and is in-fact very well trod ground. But I feel like I still learned a lot from doing it myself so… uh… mind your own business.

I referred extensively to the Gymnasium documentation on building your own [environment](https://gymnasium.farama.org/introduction/create_custom_env/#registering-and-making-the-environment). Much of the agent logic is based on the documentation for [training an agent](https://gymnasium.farama.org/introduction/train_agent/#training-an-agent) from the same source. The [section on Q-Learning](https://huggingface.co/learn/deep-rl-course/en/unit2/introduction) from Hugging Face's Deep RL Course was also very useful for understanding the theory.

My interest in this topic was sparked by a [conference](https://www.aiandgamesconference.com/) I attended last year and a great book called ["Artificial Intelligence: A Guide for Thinking Humans"](https://melaniemitchell.me/aibook/) by Melanie Mitchell, I would recommend checking out both!

## Try It Out!
### Play Against the Agent
To play against the agent, just run:

``python play_against_computer.py -o agent``

The default location for the agent's policy is outfile.csv, but this can be passed in manually using the --p (policy) argument.

### Play Against the Computer

To play against the more traditional game-playing algorithm that the agent is validated against, run:

``python play_against_computer.py -o rules``

This algorithm does not play a "perfect" game every time but will always play competitively. It will take winning moves and block winning moves for its opponent whenever possible. Otherwise, it picks moves at random.

## Training Your Own Agent

### Creating the Environment, Training the Agent, and Running Tests

To train your own agent, just run:

``python .\make_gymnasium_env.py``

This script creates the environment and runs the training according to your hyperparameters.
The resulting policy is written to outfile.csv.

### Setting Hyperparameters
Just to make it easier to use git diff on the make_gymnasium_env.py script I moved all the hyperparameters out into their own script.

The rewards are not configured in tye hyperparameters script. Instead, they are defined in the "step" function of xo.py. I intend to fix this in the future.

### Environment Difficulty
You may notice that the environment is defined twice. First, near the start of the script, before training:

``env = gym.make("gymnasium_env/xo-v0", opponent_logic = "random")``

And later, after training but before final testing:

``env = gym.make("gymnasium_env/xo-v0", opponent_logic = "competitive")``

This is to ensure that the right difficulty level is applied for each step. What I have found is that if the agent is trained against the "competitive" computer player then too many possible board combinations are closed off and the agent does not learn how to play those positions. As a result, the agent plays quite badly against a human player.

To put some numbers on this: in one test, when the agent trained against the computer where the computer played at random for 30,000 episodes, a total of 129,380 turn start observations were made. Of those, there were 4,315 distinct positions (3.3%).

When the agent trained against the computer where the computer played competitively, 138,681 turn start observations were made. Of those, 2,279 were unique (1.6%). That is over 1,000 possible positions that this version of the agent would just have to guess randomly for.

However, it is still useful to do the final test against the competitive computer player to get a more accurate sense of the policy's performance. For example, in the policy_40_58_2 example described below, the agent won 99.8% of its 500 test matches in the random environment but only 40.4% of its 2000 test matches in the final competitive environment.

## What's *Not* In This Repo
At this time, I have not set-up any rendering or visualisation options for this Gymnasium environment.

However, if you change the logging settings in the make_gymnasium_env and xo scripts to DEBUG and change the SHOW_BOARD variable in the xo script to True you will get detailed printouts of the games that are being played as part of the training process. Too detailed, some might say...

Additionally, just for clarity, the play_against_computer.py script does provide all the information you need to play the game, including an ASCII-art-esque printout of the board by default. No manual changes to the code required.

## Other Notes

### Picking invalid moves
Currently, if the agent picks an invalid move during training, it receives the same negative reward as it would if it lost the game.

The episode is not terminated. Instead, the computer does not take its turn, the step ends and the agent gets to pick another move using the same state.

I thought this approach to invalid actions illustrated the concept of reinforcement learning better than just coding the agent not to make invalid moves.
But now I think this probably does not fit in well with how the update function works, so I may change this in future

#### How does this affect max episode steps?
max_episode_steps is hardcoded as 9. Only agent actions count as steps and the max number of valid actions any player can take is 5. So there is space for these additional learning steps, but admittedly not many. This hardcoded value may need to be increased.

### What is policy_40_58_2.csv?

This file is one of the more successful outputs of the training process. It was trained using the following hyperparameters.

learning_rate = 0.1
n_episodes = 100000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.3 

The name refers to the breakdown of how the policy performed during testing
Win Rate: 40.4%
Draw Rate: 57.5%
Bad play Rate: 2.1%

A later test with the same parameters performed much worse, so maybe there was some luck of the draw on this run and the learning rate should be reduced.

### State Vs Board and Actions Vs Moves
You may notice that the game state is sometimes referred to as the "state" and sometimes referred to as the "board". The actions the user can take are sometimes referred to as "actions" and sometimes referred to as "moves". Why is this?

Most of the game logic in this project is lifted from a (previous project)[https://github.com/jckmcmhn/xoxo-gossip-giRL] where I attempted to build a similar agent to play Xs and Os from scratch. For that project, I described the game state as a list of three lists, corresponding to the rows in the game board. The moves a player could do were given as essentially row/column references to this list of lists. So, the top left corner is 00, the middle square is 11, and the bottom right corner is 22.

When I started working on this project, this slightly complicated way of referring to actions did not match the simpler examples I was working with for how the action space is defined so I added an extra layer to convert these "moves" to "actions". So, in this format, the top left corner is 0 the middle square is 4 and the bottom right is 8. Similarly, the "state" is just a flattened version of the "board" converted into a string.

Eventually, I would like to change the code so that only one system is used, but for now I have kept the "state"/"board" and "action"/"move" terms in place to make clear which system is being used at a particular time.
