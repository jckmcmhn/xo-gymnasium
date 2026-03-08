def get_action(self, obs: tuple[int, int, bool]) -> int:
    """Choose an action using epsilon-greedy strategy.
    Returns:
        action: 0 (stand) or 1 (hit)
    """
    valid_action = False
    # With probability epsilon: explore (random action)
    board = obs["board"]
    if np.random.random() < self.epsilon:
        logging.debug(f"RANDOM {random.random()}")
        while valid_action is False:
            action = self.env.action_space.sample()
            if board[action] == 0:
                logging.debug("This random action is valid")
                valid_action = True
                return action
    # With probability (1-epsilon): exploit (best known action)
    else:
        board_key = str(board)
        q_values = copy.deepcopy(self.q_values[board_key])
        #q_values = self.q_values[board_key] #TODO: I think this needs to be a copy, as otherwise the self.q_values table gets updated also, which feels a bit hack
        logging.debug(f"POLICY {random.random()}")
        while valid_action is False:
            action = int(np.argmax(q_values))
            if board[action] == 0:
                valid_action = True
                return action
            else:
                q_values[action] = -100