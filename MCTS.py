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

    def create_children(self,board):
        possible_moves = self.give_moves(board)
        next_player = self.player_move % 2 + 1 
        index = self.index + 1

        for move in possible_moves:
            newboard = deepcopy(board)
            newboard[move[0]][move[1]] = self.player_move
            self.children.append(Node(newboard,index,next_player, parent = self.index))
            index += 1

    def give_moves(self,board):
        return np.argwhere(board == 0)

    def rollout(self):
        pass


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
# board = np.zeros((3,3))

rootnode = Node(board,0,1)

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

draw_board(rootnode.board)
while rootnode.terminal is False:
    rootnode, scores = find_best_move(rootnode)
    draw_board(rootnode.board)