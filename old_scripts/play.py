import chess
import chess.svg
import sys
import math
import numpy as np
from keras.models import load_model
import tensorflow as tf
from utils import transform_board, chess_dict, squares, display_board, material_balance, save_board_to_png

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

model_path_from = 'models\\from-withTurn-b256-128-128-1024-1024-model-049.h5'
model_path_to = 'models\\to-withTurn-b6561-128-256-1024-1024-model-039.h5'
# depth to which the nn will play
depth = 2

# # load neural networks
model_from = load_model(model_path_from)
model_to = load_model(model_path_to)

board = chess.Board()
# display_board(board)

def calculate_move_score(board, move, index_move_from, index_move_to):
    # starting score at 400 seems to work
    score = 400 - index_move_from - index_move_to
    # check if move captures a piece
    piece_captured = board.piece_type_at(move.to_square)
    if piece_captured == chess.PAWN:
        score += 100
    elif piece_captured == chess.KNIGHT:
        score += 320
    elif piece_captured == chess.BISHOP:
        score += 330
    elif piece_captured == chess.ROOK:
        score += 500
    elif piece_captured == chess.QUEEN:
        score += 1000
    elif piece_captured == chess.KING:
        score += 50000
    
    # add material balance to the score
    board.push(move)
    score += material_balance(board)
    board.pop()

    return score

def get_ai_moves(board):
    arr = transform_board(board)
    arr = np.array(arr).reshape(1,9,8,12)
    # prediction probabilities
    probabilities_from = model_from.predict(arr, max_queue_size=10,workers=4)[0]
    probabilities_to = model_to.predict(arr, max_queue_size=10,workers=4)[0]
    # sort squares by descending probability
    move_from_squares = sorted(range(len(probabilities_from)), key=lambda k: probabilities_from[k])
    move_from_squares = list(reversed(move_from_squares))
    move_to_squares = sorted(range(len(probabilities_to)), key=lambda k: probabilities_to[k])
    move_to_squares = list(reversed(move_to_squares))
    
    # get all legal moves numbered
    legal_moves_from = [str(legal.from_square) for legal in list(board.legal_moves)]
    legal_moves_to = [str(legal.to_square) for legal in list(board.legal_moves)]

    possible_moves = []
    possible_moves_scores = []
    for move_from in move_from_squares:
        move_from += 1  # array starts at 0 but squares at 1
        # see if the network predicted a FROM legal move for the proper colour
        if str(move_from) in legal_moves_from and board.piece_at(move_from).color == board.turn:
            for move_to in move_to_squares:
                move_to += 1
                # see if the network predicted a TO legal move
                # and there's no piece of our colour there
                if str(move_to) in legal_moves_to and board.piece_at(move_to).color != board.turn::
                    # combine the two to form a Move and see if it's legal
                    possible_move = chess.Move(move_from, move_to)
                    if possible_move in board.legal_moves:
                        score = calculate_move_score(board, 
                                possible_move, 
                                move_from_squares.index(move_from-1),
                                move_to_squares.index(move_to-1))
                        # add the move and the resulting score of the move to the list
                        possible_moves.append(possible_move)
                        possible_moves_scores.append(score)

    return (possible_moves, possible_moves_scores)

# sort of a minimax algorithm
def minimax(board, depth, is_maximizing):
    # check if game is over or depth reached
    if depth == 0:
        possible_moves, possible_move_scores = get_ai_moves(board)
        return max(possible_move_scores)

    possible_moves, possible_move_scores = get_ai_moves(board)

    if is_maximizing:
        best_score = -math.inf
        for move in possible_moves:
            board.push(move)
            best_score = max(best_score, minimax(board, depth-1, not is_maximizing))
            board.pop()
        return best_score
    else:
        best_score = math.inf
        for move in possible_moves:
            board.push(move)
            best_score = min(best_score, minimax(board, depth-1, not is_maximizing))
            board.pop()
        return best_score
        scores.append(minimax(board, not isMaxTurn, depth-1))
        board.pop()

def get_ai_next_move(board, depth):
    best_score = -math.inf
    best_move = None

    possible_moves, possible_move_scores = get_ai_moves(board)

    for move in possible_moves:
        board.push(move)
        score = minimax(board, depth-1, False)
        board.pop()
        if (score > best_score):
            best_score = score
            best_move = move
    
    return best_move
i = 0
ai_turn = True
while not board.is_game_over():
    print(str(i))
    move_is_legal = False

    if ai_turn:
        move = get_ai_next_move(board, depth)
        board.push(move)
        save_board_to_png(board, move, 'match/'+str(i))
        ai_turn = False
    else:
        move = get_ai_next_move(board, depth)
        board.push(move)
        save_board_to_png(board, move, 'match/'+str(i))
        # # wait for a legal move from opponent
        # while not move_is_legal:
        #     user_move = input('Type your move:\n')
            
        #     move_from = squares.index(str(user_move[:2]).capitalize())
        #     move_to = squares.index(str(user_move[2:]).capitalize())
        #     move = chess.Move(move_from, move_to)

        #     legal_moves = board.generate_pseudo_legal_moves()
        #     if move in legal_moves:
        #         move_is_legal = True
        #     else:
        #         print('Illegal move, try again.')

        # board.push(move)
        ai_turn = True
    i += 1

print('Game over. Result '+board.result)