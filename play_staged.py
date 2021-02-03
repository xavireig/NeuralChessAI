import chess
import chess.svg
import sys
import datetime
import math
import numpy as np
from keras.models import load_model
import tensorflow as tf
from tensorflow.tools.docs.doc_controls import doc_private
from utils import transform_board, chess_dict, squares, display_board, material_balance, save_board_to_png

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def calculate_move_score(board, move, index_move_from, index_move_to):
    # starting score at 400 seems to work, reward higher probability from the network
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
    probabilities_from = model_from.predict(arr)[0]
    probabilities_to = model_to.predict(arr)[0]
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
                if str(move_to) in legal_moves_to:
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
    print('Possible moves: '+str(len(possible_moves_scores)))
    return (possible_moves, possible_moves_scores)

# sort of a minimax algorithm with alpha-beta pruning
def minimax(board, depth, is_maximizing, alpha, beta):

    print('<< Depth '+str(depth)+' >>')

    # check if game is over or depth reached
    if depth == 0:
        possible_moves, possible_move_scores = get_ai_moves(board)
        if len(possible_move_scores) == 0:
            return 0
        else:
            return max(possible_move_scores)

    possible_moves, possible_move_scores = get_ai_moves(board)

    if is_maximizing:
        best_score = -math.inf
        for move in possible_moves:
            board.push(move)
            score = minimax(board, depth-1, not is_maximizing, alpha, beta)
            board.pop()
            best_score = max(best_score, score) 
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return best_score
    else:
        best_score = math.inf
        for move in possible_moves:
            board.push(move)
            score = minimax(board, depth-1, not is_maximizing, alpha, beta)
            board.pop()
            best_score = min(best_score, score) 
            beta = min(beta, best_score)
            if beta <= alpha:
                break
        return best_score

def get_ai_next_move(board, depth):
    print('<< Depth '+str(depth)+' >>')
    best_score = -math.inf
    best_move = None

    possible_moves, possible_move_scores = get_ai_moves(board)

    for move in possible_moves:
        board.push(move)
        score = minimax(board, depth-1, False, -math.inf, math.inf)
        board.pop()
        if (score > best_score):
            best_score = score
            best_move = move
    
    return best_move


# depth to which the nn will play
depth = 3
stage = 'early'

# load neural networks
model_path_from = 'models/'+stage+'-from-withTurn-b1024-256-256-1024-1024-model.h5'
model_path_to = 'models/'+stage+'-to-withTurn-b1024-256-256-1024-1024-model.h5'
model_from = load_model(model_path_from)
model_to = load_model(model_path_to)

board = chess.Board()
# display_board(board)

game_moves = 0
ai_turn = False

while not board.is_game_over():

    move_is_legal = False

    if ai_turn:
        move = get_ai_next_move(board, depth-1)
        board.push(move)
        save_board_to_png(board, move, 'matchAIvsAI-depth'+str(depth)+'/'+str(game_moves))
        print('Move from the Black AI: '+move.uci())
        # display_board(board, move)
        ai_turn = False
    else:
        move = get_ai_next_move(board, depth)
        board.push(move)
        save_board_to_png(board, move, 'matchAIvsAI-depth'+str(depth)+'/'+str(game_moves))
        print('Move from the White AI: '+move.uci())
        # display_board(board, move)

        # wait for a legal move from opponent

        # #TODO: HANDLE PROMOTION
        # while not move_is_legal:
        #     user_move = input('Type your move: ')
            
        #     move_from = squares.index(str(user_move[:2]).capitalize())
        #     move_to = squares.index(str(user_move[2:]).capitalize())
        #     move = chess.Move(move_from, move_to)

        #     legal_moves = board.generate_pseudo_legal_moves()
        #     if move in legal_moves:
        #         move_is_legal = True
        #     else:
        #         print('Illegal move, try again.')

        # board.push(move)
        # save_board_to_png(board, move, 'matchJordi/'+str(game_moves))
        ai_turn = True

    game_moves += 1

    if game_moves == 20:
        current_stage = 'mid'
        print('Switching to MID stage AI')
        # load mid neural networks
        model_path_from = 'models/'+stage+'-from-withTurn-b1024-256-256-1024-1024-model.h5'
        model_path_to = 'models/'+stage+'-to-withTurn-b1024-256-256-1024-1024-model.h5'
        model_from = load_model(model_path_from)
        model_to = load_model(model_path_to)

    elif game_moves == 40:
        current_stage = 'late'
        print('Switching to LATE stage AI')
        # load mid neural networks
        model_path_from = 'models/'+stage+'-from-withTurn-b1024-256-256-1024-1024-model.h5'
        model_path_to = 'models/'+stage+'-to-withTurn-b1024-256-256-1024-1024-model.h5'
        model_from = load_model(model_path_from)
        model_to = load_model(model_path_to)

print('<<<<<<<<<<  Game over. Result '+board.result()+' >>>>>>>>>>')