import chess
import chess.svg
import chess.pgn
import numpy as np
import time
import h5py
import os

import sys
from PySide2 import QtCore, QtGui, QtWidgets
from display_svg import DisplaySVG
from utils import transform_board, chess_dict, squares, display_board

matches_path = 'matches'
data_path = 'data\\data2013.h5'
num_matches = 0
total_moves = 0

# dataset will be stored in h5 file, last channel to encode the turn 1 white -1 black
with h5py.File(data_path, "w") as h5f:
    h5f.create_dataset('moves', shape=(0,8,12,), compression="gzip", chunks=True, maxshape=(None,None,None,)) 
    h5f.create_dataset('moved_from', shape=(0,64,),  compression="gzip", chunks=True, maxshape=(None,None,))
    h5f.create_dataset('moved_to', shape=(0,64,),  compression="gzip", chunks=True, maxshape=(None,None,))

# process all files
for file in os.listdir(matches_path):
    print('Processing file '+file)    
    start = time.time()
    full_path = os.path.join(matches_path, file)
    pgn = open(full_path)

    # process all matches
    for i in range(100000000):

        game = chess.pgn.read_game(pgn)
        
        # break if no more matches to process
        if game == None:
            break

        # process only matches that are checkmates
        if game.end().comment in ['Black checkmated', 'White checkmated']:
            
            # catch an error when parsing board
            try:
                board = game.board()
                
                num_matches += 1                

                moves_array = np.empty([0,8,12])
                moved_from_array = np.zeros([0,64], dtype=int)
                moved_to_array = np.zeros([0,64], dtype=int)

                # process moves
                for move in game.mainline_moves(): 
                    # record board before move        
                    arr = transform_board(board)
                    moves_array = np.concatenate((moves_array, arr))
                    
                    # encode positions as one-hot 64 array
                    moved_from = np.zeros((64), dtype=int)
                    moved_from[move.from_square] = 1
                    
                    moved_to = np.zeros((64), dtype=int)
                    moved_to[move.to_square] = 1

                    moved_from_array = np.vstack((moved_from_array, moved_from))
                    moved_to_array = np.vstack((moved_to_array, moved_to))
                    
                    # make the move
                    board.push(move)
                    total_moves += 1 

                # save data in h5 file
                with h5py.File(data_path, "a") as h5f:
                    h5f['moves'].resize((h5f['moves'].shape[0] + moves_array.shape[0]), axis = 0)
                    h5f['moves'][-moves_array.shape[0]:] = moves_array
                    h5f['moved_from'].resize((h5f['moved_from'].shape[0] + moved_from_array.shape[0]), axis = 0)
                    h5f['moved_from'][-moved_from_array.shape[0]:] = moved_from_array
                    h5f['moved_to'].resize((h5f['moved_to'].shape[0] + moved_to_array.shape[0]), axis = 0)
                    h5f['moved_to'][-moved_to_array.shape[0]:] = moved_to_array  

            except ValueError as e:
                print ('Error during parsing - type: ', type (e))
                continue
        
    end = time.time()

    print(f"Runtime of the file is {end - start} seconds.")
    print(str(num_matches)+' matches processed.')
    print('Total moves recorded: '+str(total_moves))
    pgn.close()