from PyQt5 import QtWidgets
import sys
import cv2
import os
from PyQt5.QtWidgets import QMainWindow, QFileDialog,QDesktopWidget
from MainWindow import Ui_MainWindow
from PyQt5.QtGui import QImage, QPixmap
from IMG import IMG
'''
save as,load image,do sth
designer
pyuic5 -x ./GUI/ImageProcessing.ui -o MainWindow.py

self.lineEdit.setHidden(True)
self.label.setHidden(True)
self.pushButton_3.setHidden(True)
'''

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.flag = 0 #判斷是否有讀入圖片
        self.cwd = os.getcwd() #獲取當前路徑
        self.function_flag = 0
        self.file_comboBox.activated[str].connect(self.file)
        self.function_comboBox.activated[str].connect(self.func)
        self.pushButton_3.clicked.connect(self.func)
        self.h1.clicked.connect(self.set_h1)
        self.h2.clicked.connect(self.set_h2)
        self.h3.clicked.connect(self.set_h3)
        self.h4.clicked.connect(self.set_h4)
        self.h5.clicked.connect(self.set_h5)
        self.h6.clicked.connect(self.set_h6)
        self.h7.clicked.connect(self.set_h7)
        self.h8.clicked.connect(self.set_h8)
        self.h7.clicked.connect(self.set_h7)
        self.x3.clicked.connect(self.setx3)
        self.x5.clicked.connect(self.setx5)
        self.conv3_button.clicked.connect(self.conv3)
        self.conv5_button.clicked.connect(self.conv5)
        self.lineEdit.setHidden(True)
        self.label.setHidden(True)
        self.pushButton_3.setHidden(True)
        self.setGeometry(0, 0, 771, 475)
        self.center()

    def center(self):
        # 獲得主視窗所在的框架
        qr = self.frameGeometry()
        # 獲取顯示器的解析度，然後得到螢幕中間點的位置
        cp = QDesktopWidget().availableGeometry().center()
        # 然後把主視窗框架的中心點放置到螢幕的中心位置
        qr.moveCenter(cp)
        # 然後通過 move 函式把主視窗的左上角移動到其框架的左上角
        self.move(qr.topLeft())

    def file(self,text):
        try:
            if(text == "Input Image"):
                filename, filetype = QFileDialog.getOpenFileNames(self, 'Open Image', self.cwd, "*.jpg *.png *.JPG *.ppm *.bmp")
                if filename == '':
                    return
                self.orginal_img = cv2.imread(filename[0], cv2.IMREAD_COLOR)  
                self.showImage(self.orginal_img,self.originalimage)
                self.function_comboBox.setEnabled(True)
                self.flag = 1 
                self.proccess_img = self.orginal_img

            elif(text == "Save Image"):
                if(self.flag == 0):
                    return
                cv2.imwrite('output.jpg', self.proccess_img)
        except:
            pass
        
    def func(self,text):
        if(text == "Return Original Image"):
            self.proccess_img = self.orginal_img
            self.showImage(self.proccess_img,self.processingimage)

        if(text == "To Histogram"):
            self.origianl_histogram = IMG(self.proccess_img).to_histogram()
            self.showImage(self.origianl_histogram,self.originalhistogram)
            self.setGeometry(0, 0, 771, 850)
            self.center()
        
        if(text == "Gaussian Noise"):
            self.lineEdit.setHidden(False)
            self.label.setHidden(False)
            self.pushButton_3.setHidden(False)
            self.label.setText("σ:")
            self.lineEdit.setText("10")
            self.pushButton_3.setText("Show Gaussian Noise")
            self.function_flag = 1

        if(text == "Wavelet Transform"):
            self.lineEdit.setHidden(False)
            self.label.setHidden(False)
            self.pushButton_3.setHidden(False)
            self.label.setText("iter num:")
            self.lineEdit.setText("2")
            self.pushButton_3.setText("Show Wavelet Transform")
            self.function_flag = 2
        
        if(text == "Histogram Equalization"):
            self.origianl_histogram = IMG(self.proccess_img).to_histogram()
            self.proccess_img,self.Equalization_histogram = IMG(self.proccess_img).Histogram_Equalization()
            self.showImage(self.proccess_img,self.processingimage)
            self.showImage(self.origianl_histogram,self.originalhistogram)
            self.showImage(self.Equalization_histogram,self.processinghistogram)
            self.setGeometry(0, 0, 771, 850)
            self.center()

        if(text == "Convolution"):
            self.setGeometry(0, 0, 1070, 475)
            self.center()

        if(text == "Binary"):
            self.proccess_img = IMG(self.proccess_img).Binary()
            self.showImage(self.proccess_img,self.processingimage)
        
        if(text == "Dilation"):
            self.proccess_img = IMG(self.proccess_img).Dilation(30)
            self.showImage(self.proccess_img,self.processingimage)
        
        if(text == "Erosion"):
            self.proccess_img = IMG(self.proccess_img).Erosion(20)
            self.showImage(self.proccess_img,self.processingimage)
        
        if(text == "License plate recognition"):
            self.proccess_img = IMG(self.proccess_img).LPR()
            self.showImage(self.proccess_img,self.processingimage)

        if(text == False):
            if(self.function_flag == 1):
                self.proccess_img,self.histogram = IMG(self.proccess_img).Gaussian_Noise(int(self.lineEdit.text()))
                self.showImage(self.proccess_img,self.processingimage)
                self.showImage(self.histogram,self.processinghistogram)
                self.setGeometry(0, 0, 771, 850)
                self.center()
            elif(self.function_flag == 2):
                proccess_img = IMG(self.proccess_img).Wavelet_Transform(int(self.lineEdit.text()))
                self.showImage(proccess_img,self.processingimage)
                self.setGeometry(0, 0, 771, 475)
                self.center()
 
    def set_h1(self):
        self.spinBox_00.setValue(0)
        self.spinBox_01.setValue(0)
        self.spinBox_02.setValue(0)
        self.spinBox_03.setValue(0)
        self.spinBox_04.setValue(0)
        self.spinBox_10.setValue(0)
        self.spinBox_11.setValue(1)
        self.spinBox_12.setValue(1)
        self.spinBox_13.setValue(1)
        self.spinBox_14.setValue(0)
        self.spinBox_20.setValue(0)
        self.spinBox_21.setValue(0)
        self.spinBox_22.setValue(0)
        self.spinBox_23.setValue(0)
        self.spinBox_24.setValue(0)
        self.spinBox_30.setValue(0)
        self.spinBox_31.setValue(-1)
        self.spinBox_32.setValue(-1)
        self.spinBox_33.setValue(-1)
        self.spinBox_34.setValue(0)
        self.spinBox_40.setValue(0)
        self.spinBox_41.setValue(0)
        self.spinBox_42.setValue(0)
        self.spinBox_43.setValue(0)
        self.spinBox_44.setValue(0)
    def set_h2(self):
        self.spinBox_00.setValue(0)
        self.spinBox_01.setValue(0)
        self.spinBox_02.setValue(0)
        self.spinBox_03.setValue(0)
        self.spinBox_04.setValue(0)
        self.spinBox_10.setValue(0)
        self.spinBox_11.setValue(0)
        self.spinBox_12.setValue(1)
        self.spinBox_13.setValue(1)
        self.spinBox_14.setValue(0)
        self.spinBox_20.setValue(0)
        self.spinBox_21.setValue(-1)
        self.spinBox_22.setValue(0)
        self.spinBox_23.setValue(1)
        self.spinBox_24.setValue(0)
        self.spinBox_30.setValue(0)
        self.spinBox_31.setValue(-1)
        self.spinBox_32.setValue(-1)
        self.spinBox_33.setValue(0)
        self.spinBox_34.setValue(0)
        self.spinBox_40.setValue(0)
        self.spinBox_41.setValue(0)
        self.spinBox_42.setValue(0)
        self.spinBox_43.setValue(0)
        self.spinBox_44.setValue(0)
    def set_h3(self):
        self.spinBox_00.setValue(0)
        self.spinBox_01.setValue(0)
        self.spinBox_02.setValue(0)
        self.spinBox_03.setValue(0)
        self.spinBox_04.setValue(0)
        self.spinBox_10.setValue(0)
        self.spinBox_11.setValue(-1)
        self.spinBox_12.setValue(0)
        self.spinBox_13.setValue(1)
        self.spinBox_14.setValue(0)
        self.spinBox_20.setValue(0)
        self.spinBox_21.setValue(-1)
        self.spinBox_22.setValue(0)
        self.spinBox_23.setValue(1)
        self.spinBox_24.setValue(0)
        self.spinBox_30.setValue(0)
        self.spinBox_31.setValue(-1)
        self.spinBox_32.setValue(0)
        self.spinBox_33.setValue(1)
        self.spinBox_34.setValue(0)
        self.spinBox_40.setValue(0)
        self.spinBox_41.setValue(0)
        self.spinBox_42.setValue(0)
        self.spinBox_43.setValue(0)
        self.spinBox_44.setValue(0)
    def set_h4(self):
        self.spinBox_00.setValue(0)
        self.spinBox_01.setValue(0)
        self.spinBox_02.setValue(0)
        self.spinBox_03.setValue(0)
        self.spinBox_04.setValue(0)
        self.spinBox_10.setValue(0)
        self.spinBox_11.setValue(1)
        self.spinBox_12.setValue(1)
        self.spinBox_13.setValue(0)
        self.spinBox_14.setValue(0)
        self.spinBox_20.setValue(0)
        self.spinBox_21.setValue(1)
        self.spinBox_22.setValue(0)
        self.spinBox_23.setValue(-1)
        self.spinBox_24.setValue(0)
        self.spinBox_30.setValue(0)
        self.spinBox_31.setValue(0)
        self.spinBox_32.setValue(-1)
        self.spinBox_33.setValue(-1)
        self.spinBox_34.setValue(0)
        self.spinBox_40.setValue(0)
        self.spinBox_41.setValue(0)
        self.spinBox_42.setValue(0)
        self.spinBox_43.setValue(0)
        self.spinBox_44.setValue(0)
    def set_h5(self):
        self.spinBox_00.setValue(0)
        self.spinBox_01.setValue(0)
        self.spinBox_02.setValue(0)
        self.spinBox_03.setValue(0)
        self.spinBox_04.setValue(0)
        self.spinBox_10.setValue(0)
        self.spinBox_11.setValue(-1)
        self.spinBox_12.setValue(-1)
        self.spinBox_13.setValue(-1)
        self.spinBox_14.setValue(0)
        self.spinBox_20.setValue(0)
        self.spinBox_21.setValue(0)
        self.spinBox_22.setValue(0)
        self.spinBox_23.setValue(0)
        self.spinBox_24.setValue(0)
        self.spinBox_30.setValue(0)
        self.spinBox_31.setValue(1)
        self.spinBox_32.setValue(1)
        self.spinBox_33.setValue(1)
        self.spinBox_34.setValue(0)
        self.spinBox_40.setValue(0)
        self.spinBox_41.setValue(0)
        self.spinBox_42.setValue(0)
        self.spinBox_43.setValue(0)
        self.spinBox_44.setValue(0)
    def set_h6(self):
        self.spinBox_00.setValue(0)
        self.spinBox_01.setValue(0)
        self.spinBox_02.setValue(0)
        self.spinBox_03.setValue(0)
        self.spinBox_04.setValue(0)
        self.spinBox_10.setValue(0)
        self.spinBox_11.setValue(0)
        self.spinBox_12.setValue(-1)
        self.spinBox_13.setValue(-1)
        self.spinBox_14.setValue(0)
        self.spinBox_20.setValue(0)
        self.spinBox_21.setValue(1)
        self.spinBox_22.setValue(0)
        self.spinBox_23.setValue(-1)
        self.spinBox_24.setValue(0)
        self.spinBox_30.setValue(0)
        self.spinBox_31.setValue(1)
        self.spinBox_32.setValue(1)
        self.spinBox_33.setValue(0)
        self.spinBox_34.setValue(0)
        self.spinBox_40.setValue(0)
        self.spinBox_41.setValue(0)
        self.spinBox_42.setValue(0)
        self.spinBox_43.setValue(0)
        self.spinBox_44.setValue(0)
    def set_h7(self):
        self.spinBox_00.setValue(0)
        self.spinBox_01.setValue(0)
        self.spinBox_02.setValue(0)
        self.spinBox_03.setValue(0)
        self.spinBox_04.setValue(0)
        self.spinBox_10.setValue(0)
        self.spinBox_11.setValue(1)
        self.spinBox_12.setValue(0)
        self.spinBox_13.setValue(-1)
        self.spinBox_14.setValue(0)
        self.spinBox_20.setValue(0)
        self.spinBox_21.setValue(1)
        self.spinBox_22.setValue(0)
        self.spinBox_23.setValue(-1)
        self.spinBox_24.setValue(0)
        self.spinBox_30.setValue(0)
        self.spinBox_31.setValue(1)
        self.spinBox_32.setValue(0)
        self.spinBox_33.setValue(-1)
        self.spinBox_34.setValue(0)
        self.spinBox_40.setValue(0)
        self.spinBox_41.setValue(0)
        self.spinBox_42.setValue(0)
        self.spinBox_43.setValue(0)
        self.spinBox_44.setValue(0)
    def set_h8(self):
        self.spinBox_00.setValue(0)
        self.spinBox_01.setValue(0)
        self.spinBox_02.setValue(0)
        self.spinBox_03.setValue(0)
        self.spinBox_04.setValue(0)
        self.spinBox_10.setValue(0)
        self.spinBox_11.setValue(-1)
        self.spinBox_12.setValue(-1)
        self.spinBox_13.setValue(0)
        self.spinBox_14.setValue(0)
        self.spinBox_20.setValue(0)
        self.spinBox_21.setValue(-1)
        self.spinBox_22.setValue(0)
        self.spinBox_23.setValue(1)
        self.spinBox_24.setValue(0)
        self.spinBox_30.setValue(0)
        self.spinBox_31.setValue(0)
        self.spinBox_32.setValue(1)
        self.spinBox_33.setValue(1)
        self.spinBox_34.setValue(0)
        self.spinBox_40.setValue(0)
        self.spinBox_41.setValue(0)
        self.spinBox_42.setValue(0)
        self.spinBox_43.setValue(0)
        self.spinBox_44.setValue(0)
    
    def setx5(self):
        self.spinBox_00.setValue(1)
        self.spinBox_01.setValue(1)
        self.spinBox_02.setValue(1)
        self.spinBox_03.setValue(1)
        self.spinBox_04.setValue(1)
        self.spinBox_10.setValue(1)
        self.spinBox_11.setValue(1)
        self.spinBox_12.setValue(1)
        self.spinBox_13.setValue(1)
        self.spinBox_14.setValue(1)
        self.spinBox_20.setValue(1)
        self.spinBox_21.setValue(1)
        self.spinBox_22.setValue(1)
        self.spinBox_23.setValue(1)
        self.spinBox_24.setValue(1)
        self.spinBox_30.setValue(1)
        self.spinBox_31.setValue(1)
        self.spinBox_32.setValue(1)
        self.spinBox_33.setValue(1)
        self.spinBox_34.setValue(1)
        self.spinBox_40.setValue(1)
        self.spinBox_41.setValue(1)
        self.spinBox_42.setValue(1)
        self.spinBox_43.setValue(1)
        self.spinBox_44.setValue(1)  
    def setx3(self):
        self.spinBox_00.setValue(0)
        self.spinBox_01.setValue(0)
        self.spinBox_02.setValue(0)
        self.spinBox_03.setValue(0)
        self.spinBox_04.setValue(0)
        self.spinBox_10.setValue(0)
        self.spinBox_11.setValue(1)
        self.spinBox_12.setValue(1)
        self.spinBox_13.setValue(1)
        self.spinBox_14.setValue(0)
        self.spinBox_20.setValue(0)
        self.spinBox_21.setValue(1)
        self.spinBox_22.setValue(1)
        self.spinBox_23.setValue(1)
        self.spinBox_24.setValue(0)
        self.spinBox_30.setValue(0)
        self.spinBox_31.setValue(1)
        self.spinBox_32.setValue(1)
        self.spinBox_33.setValue(1)
        self.spinBox_34.setValue(0)
        self.spinBox_40.setValue(0)
        self.spinBox_41.setValue(0)
        self.spinBox_42.setValue(0)
        self.spinBox_43.setValue(0)
        self.spinBox_44.setValue(0)
    
    def conv3(self):
        self.kernel = [[0,0,0],[0,0,0],[0,0,0]]
        self.kernel[0][0] = self.spinBox_11.value()
        self.kernel[0][1] = self.spinBox_12.value()
        self.kernel[0][2] = self.spinBox_13.value()
        self.kernel[1][0] = self.spinBox_21.value()
        self.kernel[1][1] = self.spinBox_22.value()
        self.kernel[1][2] = self.spinBox_23.value()
        self.kernel[2][0] = self.spinBox_31.value()
        self.kernel[2][1] = self.spinBox_32.value()
        self.kernel[2][2] = self.spinBox_33.value()
        self.proccess_img = IMG(self.proccess_img).conv3(self.kernel)
        self.showImage(self.proccess_img,self.processingimage)
    
    def conv5(self):
        self.kernel = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
        self.kernel[0][0] = self.spinBox_00.value()
        self.kernel[0][1] = self.spinBox_01.value()
        self.kernel[0][2] = self.spinBox_02.value()
        self.kernel[0][3] = self.spinBox_03.value()
        self.kernel[0][4] = self.spinBox_04.value()
        self.kernel[1][0] = self.spinBox_10.value()
        self.kernel[1][1] = self.spinBox_11.value()
        self.kernel[1][2] = self.spinBox_12.value()
        self.kernel[1][3] = self.spinBox_13.value()
        self.kernel[1][4] = self.spinBox_14.value()
        self.kernel[2][0] = self.spinBox_20.value()
        self.kernel[2][1] = self.spinBox_21.value()
        self.kernel[2][2] = self.spinBox_22.value()
        self.kernel[2][3] = self.spinBox_23.value()
        self.kernel[2][4] = self.spinBox_24.value()
        self.kernel[3][0] = self.spinBox_30.value()
        self.kernel[3][1] = self.spinBox_31.value()
        self.kernel[3][2] = self.spinBox_32.value()
        self.kernel[3][3] = self.spinBox_33.value()
        self.kernel[3][4] = self.spinBox_34.value()
        self.kernel[4][0] = self.spinBox_40.value()
        self.kernel[4][1] = self.spinBox_41.value()
        self.kernel[4][2] = self.spinBox_42.value()
        self.kernel[4][3] = self.spinBox_43.value()
        self.kernel[4][4] = self.spinBox_44.value()
        self.proccess_img = IMG(self.proccess_img).conv5(self.kernel)
        self.showImage(self.proccess_img,self.processingimage)
                
    def showImage(self,image,label):
        try:
            height, width ,channel= image.shape
        except:
            height, width = image.shape
            print('except',height, width)
        bytesPerline = 3 * width
        self.qImg = QImage(image.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        label.setPixmap(QPixmap.fromImage(self.qImg).scaled(label.width(),label.height()))
        

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())