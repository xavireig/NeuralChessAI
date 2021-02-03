import chess
import numpy as np
import sys
from PySide2 import QtCore, QtGui, QtWidgets
from display_svg import DisplaySVG
from cairosvg import svg2png


chess_dict = {
    'p' : [1,0,0,0,0,0,0,0,0,0,0,0],
    'P' : [0,0,0,0,0,0,1,0,0,0,0,0],
    'n' : [0,1,0,0,0,0,0,0,0,0,0,0],
    'N' : [0,0,0,0,0,0,0,1,0,0,0,0],
    'b' : [0,0,1,0,0,0,0,0,0,0,0,0],
    'B' : [0,0,0,0,0,0,0,0,1,0,0,0],
    'r' : [0,0,0,1,0,0,0,0,0,0,0,0],
    'R' : [0,0,0,0,0,0,0,0,0,1,0,0],
    'q' : [0,0,0,0,1,0,0,0,0,0,0,0],
    'Q' : [0,0,0,0,0,0,0,0,0,0,1,0],
    'k' : [0,0,0,0,0,1,0,0,0,0,0,0],
    'K' : [0,0,0,0,0,0,0,0,0,0,0,1],
    '.' : [0,0,0,0,0,0,0,0,0,0,0,0]
}

squares = [
    'A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1',
    'A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2',
    'A3', 'B3', 'C3', 'D3', 'E3', 'F3', 'G3', 'H3',
    'A4', 'B4', 'C4', 'D4', 'E4', 'F4', 'G4', 'H4',
    'A5', 'B5', 'C5', 'D5', 'E5', 'F5', 'G5', 'H5',
    'A6', 'B6', 'C6', 'D6', 'E6', 'F6', 'G6', 'H6',
    'A7', 'B7', 'C7', 'D7', 'E7', 'F7', 'G7', 'H7',
    'A8', 'B8', 'C8', 'D8', 'E8', 'F8', 'G8', 'H8',
]

piece_values = {
    'P' : 100,
    'N' : 320,
    'B' : 330,
    'R' : 500,
    'Q' : 1000,
    'K' : 50000
}

# transforms a board into numpy array for training data
def transform_board(board):
    # 8x8 matrix of 12 piece types and extra matrix for the player's turn

    # empty board array
    board_array = np.zeros(shape=(9,8,12), dtype=int)

    row = 0
    column = 0

    for square in squares:
        piece = board.piece_at(squares.index(square))
        if piece != None:
            board_array[row, column] = chess_dict[str(piece)]

        column += 1

        if column % 8 == 0:
            column = 0
            row += 1

    # fill 9th matrix with turn info
    if board.turn == chess.WHITE:
        board_array[row] = 1
    elif board.turn == chess.BLACK:
        board_array[row] = -1

    return board_array

# evaluation function based on the number of pieces for each side
def material_balance(board):
    score = 0
    for square in squares:
        piece = board.piece_at(squares.index(square))
        if piece != None:
            if piece.color == chess.WHITE:
                score += piece_values[str(piece)]
            elif piece.color == chess.BLACK:
                score -= piece_values[str(piece).capitalize()]
    return score

def display_board(board, move=None):
    svg = chess.svg.board(board, size=560, lastmove=move)
    display.display(svg)
    app.exec_()

if not QtWidgets.QApplication.instance():
    app = QtWidgets.QApplication(sys.argv)
else:
    app = QtWidgets.QApplication.instance()

display = DisplaySVG()

def save_board_to_png(board, move=None, path=None):
    ## handle = rsvg.Handle(<svg filename>)
    # or, for in memory SVG data:
    svg = chess.svg.board(board, size=560, lastmove=move)
    svg2png(bytestring=svg,write_to=str(path)+'.png')