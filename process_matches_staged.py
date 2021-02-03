import chess
import chess.svg
import chess.pgn
import numpy as np
import time
import h5py
import os
from tqdm import tqdm

import sys
from PySide2 import QtCore, QtGui, QtWidgets
from display_svg import DisplaySVG
from utils import transform_board, chess_dict, squares, display_board

matches_path = 'matches'
data_path = 'data\\data2014-2020-staged-'
num_matches = 0
total_moves = 0

# what stage of the game is it (early, mid, late)
stages = ['early', 'mid', 'late']

# dataset will be stored in h5 file, last channel to encode the turn 1 white -1 black
with h5py.File(data_path+'early.h5', "w") as h5f:
    h5f.create_dataset('board_early', shape=(0,8,12,), compression="gzip", chunks=True, maxshape=(None,None,None,)) 
    h5f.create_dataset('moved_from_early', shape=(0,64,),  compression="gzip", chunks=True, maxshape=(None,None,))
    h5f.create_dataset('moved_to_early', shape=(0,64,),  compression="gzip", chunks=True, maxshape=(None,None,))

with h5py.File(data_path+'mid.h5', "w") as h5f:
    h5f.create_dataset('board_mid', shape=(0,8,12,), compression="gzip", chunks=True, maxshape=(None,None,None,)) 
    h5f.create_dataset('moved_from_mid', shape=(0,64,),  compression="gzip", chunks=True, maxshape=(None,None,))
    h5f.create_dataset('moved_to_mid', shape=(0,64,),  compression="gzip", chunks=True, maxshape=(None,None,))

with h5py.File(data_path+'late.h5', "w") as h5f:
    h5f.create_dataset('board_late', shape=(0,8,12,), compression="gzip", chunks=True, maxshape=(None,None,None,)) 
    h5f.create_dataset('moved_from_late', shape=(0,64,),  compression="gzip", chunks=True, maxshape=(None,None,))
    h5f.create_dataset('moved_to_late', shape=(0,64,),  compression="gzip", chunks=True, maxshape=(None,None,))

# process all files
for file in os.listdir(matches_path):
    print('Processing file '+file)    
    start = time.time()
    full_path = os.path.join(matches_path, file)
    pgn = open(full_path)

    # process all matches
    for i in tqdm(range(110000)):
        
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
                game_moves = 0
                current_stage = 'early'

                moves_array_early = np.empty([0,8,12])
                moved_from_array_early = np.zeros([0,64], dtype=int)
                moved_to_array_early = np.zeros([0,64], dtype=int)

                moves_array_mid = np.empty([0,8,12])
                moved_from_array_mid = np.zeros([0,64], dtype=int)
                moved_to_array_mid = np.zeros([0,64], dtype=int)

                moves_array_late = np.empty([0,8,12])
                moved_from_array_late = np.zeros([0,64], dtype=int)
                moved_to_array_late = np.zeros([0,64], dtype=int)

                # process moves
                for move in game.mainline_moves(): 
                    # record board before move        
                    arr = transform_board(board)
                    
                    # encode positions as one-hot 64 array
                    moved_from = np.zeros((64), dtype=int)
                    moved_from[move.from_square] = 1

                    moved_to = np.zeros((64), dtype=int)
                    moved_to[move.to_square] = 1

                    if current_stage == 'early':
                        moves_array_early = np.concatenate((moves_array_early, arr))
                        moved_from_array_early = np.vstack((moved_from_array_early, moved_from))
                        moved_to_array_early = np.vstack((moved_to_array_early, moved_to))
                    elif current_stage == 'mid':
                        moves_array_mid = np.concatenate((moves_array_mid, arr))
                        moved_from_array_mid = np.vstack((moved_from_array_mid, moved_from))
                        moved_to_array_mid = np.vstack((moved_to_array_mid, moved_to))
                    elif current_stage == 'late':
                        moves_array_late = np.concatenate((moves_array_late, arr))
                        moved_from_array_late = np.vstack((moved_from_array_late, moved_from))
                        moved_to_array_late = np.vstack((moved_to_array_late, moved_to))

                    # make the move
                    board.push(move)
                    game_moves += 1
                    total_moves += 1 

                    if 20 <= game_moves < 40:
                        current_stage = 'mid'
                    elif game_moves >= 40:
                        current_stage = 'late'

                # save data in h5 file
                with h5py.File(data_path+'early.h5', "a") as h5f:
                    h5f['board_early'].resize((h5f['board_early'].shape[0] + moves_array_early.shape[0]), axis = 0)
                    h5f['board_early'][-moves_array_early.shape[0]:] = moves_array_early
                    h5f['moved_from_early'].resize((h5f['moved_from_early'].shape[0] + moved_from_array_early.shape[0]), axis = 0)
                    h5f['moved_from_early'][-moved_from_array_early.shape[0]:] = moved_from_array_early
                    h5f['moved_to_early'].resize((h5f['moved_to_early'].shape[0] + moved_to_array_early.shape[0]), axis = 0)
                    h5f['moved_to_early'][-moved_to_array_early.shape[0]:] = moved_to_array_early 

                # check that the mid array is not empty
                if len(moves_array_mid) > 0:
                    with h5py.File(data_path+'mid.h5', "a") as h5f:
                        h5f['board_mid'].resize((h5f['board_mid'].shape[0] + moves_array_mid.shape[0]), axis = 0)
                        h5f['board_mid'][-moves_array_mid.shape[0]:] = moves_array_mid
                        h5f['moved_from_mid'].resize((h5f['moved_from_mid'].shape[0] + moved_from_array_mid.shape[0]), axis = 0)
                        h5f['moved_from_mid'][-moved_from_array_mid.shape[0]:] = moved_from_array_mid
                        h5f['moved_to_mid'].resize((h5f['moved_to_mid'].shape[0] + moved_to_array_mid.shape[0]), axis = 0)
                        h5f['moved_to_mid'][-moved_to_array_mid.shape[0]:] = moved_to_array_mid

                if len(moves_array_late) > 0:
                    with h5py.File(data_path+'late.h5', "a") as h5f:
                        h5f['board_late'].resize((h5f['board_late'].shape[0] + moves_array_late.shape[0]), axis = 0)
                        h5f['board_late'][-moves_array_late.shape[0]:] = moves_array_late
                        h5f['moved_from_late'].resize((h5f['moved_from_late'].shape[0] + moved_from_array_late.shape[0]), axis = 0)
                        h5f['moved_from_late'][-moved_from_array_late.shape[0]:] = moved_from_array_late
                        h5f['moved_to_late'].resize((h5f['moved_to_late'].shape[0] + moved_to_array_late.shape[0]), axis = 0)
                        h5f['moved_to_late'][-moved_to_array_late.shape[0]:] = moved_to_array_late

            except ValueError as e:
                print ('Error during parsing - type: ', type (e))
                continue
        
    end = time.time()

    print(f"Runtime of the file is {end - start} seconds.")
    print(str(num_matches)+' matches processed.')
    print('Total moves recorded: '+str(total_moves))
    pgn.close()