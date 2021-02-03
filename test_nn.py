import chess
import chess.svg
import chess.pgn
import sys
import numpy as np
from keras.models import load_model
import tensorflow as tf
from utils import transform_board, chess_dict, squares, display_board

# fix TF 2.4 issue
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# random move from file, get the board and the value
gamefound = False
while not gamefound:

    pgn = open('matches\\matches2018.pgn')
    randomgame = np.random.randint(1,50000)

    for i in range(randomgame):
        chess.pgn.skip_game(pgn)

    game = chess.pgn.read_game(pgn)

    # look for checkmated games
    if game.end().comment in ['Black checkmated', 'White checkmated']:
        gamefound = True
    
    pgn.close()

nummoves = int(game.headers['PlyCount'])
idgame = game.headers['FICSGamesDBGameNo']
print('Game '+str(idgame))
random_move = np.random.randint(0, nummoves)
print('Move chosen: '+str(random_move))

i = 0
board = game.board()

def make_prediction(board, move):
    # real outcomes
    from_square = squares[move.from_square]
    to_square = squares[move.to_square]

    arr = transform_board(board)
    arr = np.array(arr).reshape(1,9,8,12)

    # move prediction
    from_square_prediction = squares[np.argmax(model_from.predict(arr))]
    to_square_prediction = squares[np.argmax(model_to.predict(arr))]

    print('The model predicts: '+str(from_square_prediction)+' to '+str(to_square_prediction)+' and the real move was '+str(from_square)+' to '+str(to_square))
    display_board(board, move)

# random move in the randomly selected game
for move in game.mainline_moves():
    i += 1
    if i == 20:
        stage = 'mid'
    elif i == 40: 
        stage = 'late'
    
    if i == random_move: 

        # load neural networks
        model_path_from = 'models/'+stage+'-from-withTurn-b1024-256-256-1024-1024-model.h5'
        model_path_to = 'models/'+stage+'-to-withTurn-b1024-256-256-1024-1024-model.h5'
        model_from = load_model(model_path_from)
        model_to = load_model(model_path_to) 

        make_prediction(board, move)

        break
    board.push(move)