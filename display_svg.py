import sys
from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtNetwork import QNetworkProxy, QNetworkProxyFactory
from PySide2.QtWebEngineWidgets import QWebEngineView


class DisplaySVG(QtWidgets.QWidget):
    'A simple SVG display.'
    def __init__(self, url=None, parent=None):
        super().__init__(parent)
        self.resize(600,600)
        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.webview = QWebEngineView(self)
        self.verticalLayout.addWidget(self.webview)

        self.setWindowTitle('Display SVG')
        act = QtWidgets.QAction('Close', self)
        act.setShortcuts([QtGui.QKeySequence(QtCore.Qt.Key_Escape)])
        act.triggered.connect(self.close)
        self.addAction(act)

    def display(self, svg):
        self.webview.setHtml(svg)
        self.show()
