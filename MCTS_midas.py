from copy import deepcopy
import numpy as np

class MCTS_UCT():
    def __init__(self, state, parent=None, parent_action=None):
        # for each state, save parent, parent action and its children
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []

        # rememver the number of visits for each node
        self._n_visits = 0

        # initiate dictionary with possible outcomes
        self._outcome = {}
        self._outcome[-1] = 0
        self._outcome[0] = 0
        self._outcome[1] = 0

        # get possible actions for the current state of the board
        self._possible_actions = self.possible_actions()

    # return the moves that can be made for the current state of the board
    def possible_actions(self):
        _possible_actions = self.state.get_moves()
        return _possible_actions

    # the q-value for a state is defined as: wins - losses
    def q_value(self):
        return self._outcome[1] - self._outcome[-1]

    def expand(self): #take the last action in the list, 
        # action becomes the last value that is in the possible moves list
        action = self._possible_actions.pop()
        next_state = self.state.move(action)

        # initiate the next state as a child node
        child_node = MCTS_UCT(
            next_state, parent=self, parent_action=action)

        # save and return the child node
        self.children.append(child_node)
        return child_node 

    def rollout(self):
        # save the current state of the board
        current_rollout_state = deepcopy(self.state)
        player = 1
        # rollout until the game ends
        while not current_rollout_state.end_game():
            
            # get the moves that are possible
            possible_moves = deepcopy(current_rollout_state.get_moves())

            # from the possible moves, perform a random action 
            action = possible_moves[np.random.randint(len(possible_moves))]

            # update the board with the action
            
            current_rollout_state = deepcopy(current_rollout_state.move(action,player))
            print(current_rollout_state.board)
            player = player % 2 + 1
            # current_rollout_state.move(action)

        # return the winner (-1, 0, 1)
        return current_rollout_state.game_outcome()

    def backpropagate(self, outcome):
        # update number of visits and the outcome
        self._n_visits += 1.
        self._outcome[outcome] += 1.

        # backpropagate by recursively calling this function for the node's parent
        if self.parent:
            self.parent.backpropagate(outcome)

    def best_child(self, const = 1):
        # calculate weights for all child nodes
        choices_weights = [(c.q_value() / c._n_visits) + const * np.sqrt((2 * np.log(self._n_visits) / c._n_visits)) for c in self.children]
        
        # return the child node with the heighest weight
        return self.children[np.argmax(choices_weights)]


    def _tree_policy(self):
        current_node = self
        # loop while the game has not ended
        while not current_node.state.end_game():
            # if there are any possible actions
            if len(current_node._possible_actions) != 0:
                return current_node.expand() # gives a child node, resulting from last action in actionlist
                #constructs the child nodes, with given boards

            # if there are no possible actions
            else:
                current_node = current_node.best_child()
                #if al actions tried, choose best child
        return current_node

    def best_next_state(self):
        n_simulations = 100
        
        for _ in range(n_simulations):
            
            current_node = self._tree_policy() #finds leave node
            r = current_node.rollout() #rollout from leave node
            print(r)
            current_node.backpropagate(r) #update values of nodes in path to leaf node
        
        return self.best_child() #returns best direct child node based on UCB


class State:
    def __init__(self, board):
        self.board = board
        return

    # return indices of possible moves
    def get_moves(self): 
        return list(np.argwhere(self.board == 0))

    def end_game(self):
        winner = self.game_outcome()
        # winner == False: no winner is determined 
        if winner == None:
            return False
        # winner == 0: draw
        else:
            return True

    def game_outcome(self):
        # col1 and col2 represent three consecutive 1's or 2's
        col1 = np.ones(3,dtype=int).tolist()
        col2 = np.repeat(2,3).tolist()

        #checking the columns rows and diagonals for a winner
        
        if self.check_win(col1):
            return 1
        elif self.check_win(col2):
            return -1
        elif self.check_full():
            return 0
        else:
            return None

    # check rows, columns and diagonal for a winning state
    def check_win(self, col):
        if (col in self.board.tolist() or col in self.board.T.tolist() or self.board.diagonal().tolist() == col or 
        np.fliplr(self.board).diagonal().tolist() == col):
            return True

    # check whether the board is full
    def check_full(self):
        if np.count_nonzero(self.board) == self.board.size:
            return True

    def move(self, action, player=1):
        # deepcopy to prevent multiple actions taken
        state = deepcopy(self.board)

        # perform the action
        y,x = action
        state[y,x] = player

        # return new state
        return State(state)

# initialize the board as described in the assignment
def init_board():
    board = np.zeros((1,9)).flatten()
    board[[3,4]] = 1
    board[[1,5]] = 2
    return board.reshape(3,3).astype(int)

def random_action(board):
    # all possible actions
    actions = np.argwhere(board == 0)
    
    # only make a move if an action can be taken
    if actions.size != 0:
        # return a random action
        return actions[np.random.randint(len(actions))]

    # no action possible
    return None

# draw the state of the board 
def draw_board(board):
    for y in range(board.shape[0]):
        for x in range(board.shape[1]):
            if board[y,x] == 0:
                print('   ', end="")
            elif board[y,x] == 1:
                print(' X ', end="")
            elif board[y,x] == 2:
                print(' O ', end="")
            if x != board.shape[1] - 1:
                print('|', end="")
        if y != board.shape[0] - 1:
                print('\n-----------')
    print('\n\n')

def main():
    # initialize the board
    board_state = State(init_board())
    mcts = MCTS_UCT(board_state, parent=None, parent_action=None)

    # draw initial state of the board
    draw_board(board_state.board)
    
    # initialize starting player
    player = 0

    while not board_state.end_game():
        # player X
        if player == 0:

            # determine the best action 
            best_next_state = mcts.best_next_state()
            action = best_next_state.parent_action
            print('X: ', action)

            # perform the action
            board_state = board_state.move(action, player=1)

            # update MCTS with the new state of the board
            mcts = MCTS_UCT(board_state, parent=None, parent_action=action)

        # player O 
        elif player == 1:
            
            # determine a random action
            action = random_action(board_state.board)
            y,x = action
            print('O: ', action)

            # perform the action
            board_state = board_state.move(action, player=2)

            # update MCTS with the new state of the board
            mcts = MCTS_UCT(board_state, None, action)

        # draw the updated state of the board
        draw_board(board_state.board)

        player = (player + 1) % 2

    winner = board_state.game_outcome()

    if winner == 1:
        print('Winner = X')
    elif winner == -1:
        print('Winner = O')
    else:
        print('Draw')
    return 

main()