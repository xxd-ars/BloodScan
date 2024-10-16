#coding = 'utf-8'

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from utils.func import MyMainWindow
 
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MyMainWindow()
    MainWindow.show()
    sys.exit(app.exec_())