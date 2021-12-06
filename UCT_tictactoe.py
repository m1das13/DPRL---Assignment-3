import numpy as np
from copy import deepcopy

class Node:
    def __init__(self,board,index, player_move, parent = None):
        self.children = []
        self.parent = parent
        self.index = index
        self.board = board
        self.value = self.check_winner(board)
        self.terminal = self.value != None
        self.player_move = player_move

        self.count = 0
        self.total = 0


    def update_value(self,new_value):
        self.count += 1
        self.total += new_value

    def get_UCB(self, c):
        if self.count == 0:
            return 1e8
        else:
            return self.total/self.count + c*np.sqrt(np.log(self.parent.count)/self.count)

    def create_children(self,board):
        possible_moves = self.give_moves(board)
        next_player = self.player_move % 2 + 1 
        index = self.index + 1

        for move in possible_moves:
            newboard = deepcopy(board)
            newboard[move[0]][move[1]] = self.player_move
            self.children.append(Node(newboard,index,next_player, parent = self))
            index += 1

    def give_moves(self,board):
        return np.argwhere(board == 0)

    def check_winner(self,board):
        col1 = np.ones(3,dtype=int).tolist()
        col2 = np.repeat(2,3).tolist()

        #checking the columns rows and diagonals for a winner
        if self.check_win(board,col1):
            return 1
        elif self.check_win(board,col2):
            return -1
        elif self.check_full(board):
            return 0
        else:
            return None

    def check_win(self,board,col):
        if (col in board.tolist() or col in board.T.tolist() or board.diagonal().tolist() == col or 
        np.fliplr(board).diagonal().tolist() == col):
            return True
    
    def check_full(self,board):
        if np.count_nonzero(board) == board.size:
            return True


class Tree:
    def __init__(self,start_board,player):
        self.board = start_board
        self.root = Node(start_board,0,player)
        self.node_index = 1

    
    def rollout(self,leaf_node):
        node = leaf_node
        while not node.terminal:
            if node.children == []:
                node.create_children(node.board)
            node = np.random.choice(node.children)
        return node.value

    def policy_run(self,node,chain,c):
        if (node.count == 0 and node != self.root) or node.terminal:
            reward = self.rollout(node)
            return reward,chain
        else:
            child_node = self.choose_best(node,c)
            chain.append(child_node)
            return self.policy_run(child_node,chain,c)

    def choose_best(self,node,c):
        if node.children == []:
            node.create_children(node.board)
        ucb_values = [child.get_UCB(c) for child in node.children]
        return node.children[np.argmax(ucb_values)]

    def backprop(self,chain,reward):
        for node in chain:
            node.update_value(reward)

    def move_root(self,root):
        q_values = [child.total/child.count for child in root.children]
        en_move = root.children[np.argmax(q_values)]
        print('q values: ',q_values)
        draw_board(en_move.board)
        if en_move.terminal:
            return en_move
        else:
            en_move.create_children(en_move.board)
            new_root = np.random.choice(en_move.children)
            draw_board(new_root.board)
            return new_root

    def policy_from_root(self, root, c):
        reward, chain = self.policy_run(root, [root], c)
        self.backprop(chain,reward)

    def UCB_MCTS(self,root,iters,path,c):
        for _ in range(iters):
            self.policy_from_root(root,c)
        root = self.move_root(root)
        path.append(root)
        if root.terminal:
            return root.value,path
        else:
            return self.UCB_MCTS(root,iters,path,c)

def minimax(node, max_player):
    if node.terminal:
        return node.value
    if max_player:
        node.create_children(node.board)
        max_eval = -np.inf
        for child in node.children:
            evaluation = minimax(child,False)
            max_eval = np.maximum(max_eval,evaluation)
        return max_eval
    else:
        min_eval = np.inf
        node.create_children(node.board)
        for child in node.children:
            evaluation = minimax(child,True)
            min_eval = np.minimum(min_eval,evaluation)
        return min_eval


def create_start_board(start_crosses,start_circles, board_size = (3,3)):
    start_board  = np.zeros(board_size,dtype=int)
    for cross in start_crosses:
        start_board[cross] = 1
    for circle in start_circles:
        start_board[circle] = 2
    return start_board

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


start_crosses = [(2,1),(1,1)] 
start_circles = [(0,1),(1,2)]

board = create_start_board(start_crosses,start_circles,board_size=(3,3))
board = np.zeros((3,3))

# rootnode = Node(board,0,1)

def find_best_move(root):
    root.create_children(root.board)
    scores = []
    for child in root.children:
        max_player = child.player_move == 1
        value = minimax(child,max_player)
        scores.append(value)
    if root.player_move ==1:
        best_move = np.argmax(scores)
    else:
        best_move = np.argmin(scores)
    return root.children[best_move],scores

# draw_board(rootnode.board)
# while rootnode.terminal is False:
#     rootnode, scores = find_best_move(rootnode)
#     draw_board(rootnode.board)

tree = Tree(board,1)
draw_board(tree.root.board)
win, path = tree.UCB_MCTS(root = tree.root, iters = 10000,path = [], c = 1)
# [print(node.board) for node in path]