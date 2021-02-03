import sys
from PySide2 import QtCore, QtGui, QtWidgets
from display_svg import DisplaySVG
import chess
import chess.svg
import chess.pgn

app = QtWidgets.QApplication(sys.argv)
display = DisplaySVG()

def display_board(board):
    svg = chess.svg.board(board, size=560)
    display.display(svg)
    app.exec_()

pgn = open('test_chess.txt')
first_game = chess.pgn.read_game(pgn)
board = first_game.board()
print(board.epd())
display_board(board)