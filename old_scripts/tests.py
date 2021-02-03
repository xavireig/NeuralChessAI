import chess
import chess.svg
import chess.pgn
from utils import transform_board, chess_dict, squares, display_board, material_balance, save_board_to_png

pgn = open('matches\\matches2013.pgn')
nummoves = 0
numgames = 0
# process all matches
for i in range(100000000):

    game = chess.pgn.read_game(pgn)
    
    # break if no more matches to process
    if game == None:
        break

    # process only matches that are checkmates
    if game.end().comment in ['Black checkmated', 'White checkmated']:
        nummoves += int(game.headers['PlyCount'])
        numgames += 1

average = nummoves / numgames
print(str(average))