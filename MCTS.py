import numpy as np
from copy import deepcopy

# class Node:
#   def __init__(self, index, value = None, parent = None):
#     self.childnodes = [0,0]
#     self.parent = parent
#     self.value = value
#     self.total = 0
#     self.index = index
#     self.count = 0 

#   def update_value(self, new_value):
#     self.count += 1
#     self.total += new_value

#   def get_UCB(self, c):
#     if self.count == 0:
#       return 1e8
#     else:
#       return self.total/self.count + c*np.sqrt(np.log(self.parent.count)/self.count)


class Node:
    def __init__(self,board,index, player_move, parent = None):
        self.children = []
        self.parent = parent
        self.index = index
        self.board = board
        self.value = self.check_winner(board)
        self.terminal = self.value != False
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
            return False

    def check_win(self,board,col):
        if (col in board.tolist() or col in board.T.tolist() or board.diagonal().tolist() == col or 
        np.fliplr(board).diagonal().tolist() == col):
            return True
    
    def check_full(self,board):
        if np.count_nonzero(board) == board.size:
            return True


def minimax(node, max_player, board_seq = []):
    if node.terminal:
        return node.value, board_seq
    if max_player:
        max_eval = -100
        node.create_children(node.board)
        for child in node.children:
            n_board_seq = np.concatenate([board_seq,child.board])
            evaluation,n_board_seq = minimax(child,False,n_board_seq)
            if evaluation > max_eval:
                max_eval = evaluation
                board_seq = n_board_seq
        return max_eval,board_seq
    else:
        min_eval = 100
        node.create_children(node.board)
        for child in node.children:
            n_board_seq = np.concatenate([board_seq,child.board])
            evaluation, n_board_seq = minimax(child,True,n_board_seq)
            if evaluation < min_eval:
                min_eval = evaluation
                board_seq = n_board_seq
            
        return min_eval,board_seq

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


start_crosses = [(1,0),(1,1)] 
start_circles = [(0,1),(1,2)]

board = create_start_board(start_crosses,start_circles,board_size=(3,3))


rootnode = Node(board,0,1)
eval,board_seq = minimax(rootnode,True,board)
# print(board_seq)

splits = np.arange(3,len(board_seq)+4,3)
splits
for split in splits:
    draw_board(board_seq[split-3:split])


# board = np.array([[1,1,1],[0,0,0],[0,0,0]])
# check_winner(board)
# print(board)