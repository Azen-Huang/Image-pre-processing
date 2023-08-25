import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import random
import math
import sys
import pytesseract
sys.setrecursionlimit(10000000)
class IMG:
    def __init__(self,img):
        self.img = img
        self.gray_img = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.gray_img.shape
        self.max,self.maxi,self.maxj = 0,0,0
        self.comp_num = 0
        self.comp_list = []
        self.m = []
        self.ff = 0

    def to_histogram(self):
        his = [0]*256

        for i in range(self.height):
            for j in range(self.width):
                his[self.gray_img[i][j]] += 1

        his = np.array(his)
        his_dict = {}
        for i in range(256):
            his_dict[i] = his[i]

        myList = his_dict.items()
        myList = sorted(myList) 
        x, y = zip(*myList) 
        plt.bar(x, y)
        # 显示横轴标签
        plt.xlabel("Intensity")
        # 显示纵轴标签
        plt.ylabel("Frequency")
        # 显示图标题
        plt.title("histogram")
        #plt.show()
        plt.savefig('histogram.jpg')
        plt.clf()
        histogram_img = cv2.imread('histogram.jpg',cv2.IMREAD_COLOR)
        return histogram_img
    
    def Gaussian_Noise(self,sigma):
        his = 0
        if(self.width%2):
            self.width += 1
        
        self.gray_img = cv2.resize(self.gray_img,(self.width,self.height))
        self.height, self.width = self.gray_img.shape
        GN_img = [[0]*self.width for _ in range(0,self.height)]
        his = []
        his_set = set()
        for i in range(0,self.height):
            for j in range(0,self.width,2):
                p1,p2 = self.gray_img[i][j],self.gray_img[i][j+1]
                r = random.uniform(0,1)
                phi = random.uniform(0,1)
                z1 = sigma*math.cos(2*math.pi*phi)*math.sqrt(-2*math.log(r))
                z2 = sigma*math.sin(2*math.pi*phi)*math.sqrt(-2*math.log(r))
                f1,f2 = p1+z1,p2+z2
                if(f1 < 0):
                    GN_img[i][j] = 0
                elif(f1 > 255):
                    GN_img[i][j] = 255
                else:
                    GN_img[i][j] = f1

                if(f2 < 0):
                    GN_img[i][j+1] = 0
                elif(f2 > 255):
                    GN_img[i][j+1] = 255
                else:
                    GN_img[i][j+1] = f2
                
                his.append(round(z1))
                his.append(round(z2))
                his_set.add(round(z1))
                his_set.add(round(z2))
        
        x = sorted(list(his_set))
        #print(x)
        y = [0]*len(his_set)
        for i in range(0,len(x)):
            for j in his:
                if(x[i] == j):
                    y[i] += 1


        plt.bar(x, y)
        # 显示横轴标签
        plt.xlabel("Intensity")
        # 显示纵轴标签
        plt.ylabel("Frequency")
        # 显示图标题
        plt.title("GN_histogram")
        #plt.show()
        plt.savefig('GN_histogram.jpg')
        plt.clf()
        histogram_img = cv2.imread('GN_histogram.jpg',cv2.IMREAD_COLOR)
      
        GN_img = np.uint8(GN_img)
        cv2.imwrite('GN.jpg', GN_img)
        GN_img = cv2.imread('GN.jpg')
        os.remove("GN.jpg")

        return GN_img,histogram_img
  
    def Wavelet_Transform(self,LEVEL):
        IMG_SIZE = 512 
        cut = [1,2,4,8,16,32,64,128,256,512]
        wavelet_img = cv2.resize(self.gray_img,(IMG_SIZE,IMG_SIZE))
        
        for level in range(1,LEVEL+1):
            img = np.copy(wavelet_img)#不能直接img = wavelet_img因為python會自動覆蓋記憶體位置
            LLi,LLj = 0,0
            HLi,HLj = 0,int(IMG_SIZE/(2*cut[level-1]))
            LHi,LHj = int(IMG_SIZE/(2*cut[level-1])),0
            HHi,HHj = int(IMG_SIZE/(2*cut[level-1])),int(IMG_SIZE/(2*cut[level-1]))
            for i in range(0,int(IMG_SIZE/cut[level-1]),2):
                LLj = 0
                HLj = int(IMG_SIZE/(2*cut[level-1]))
                LHj = 0
                HHj = int(IMG_SIZE/(2*cut[level-1]))
                
                for j in range(0,int(IMG_SIZE/cut[level-1]),2):
                    A = int(wavelet_img[i][j])
                    B = int(wavelet_img[i][j+1])
                    C = int(wavelet_img[i+1][j])
                    D = int(wavelet_img[i+1][j+1])

                    img[LLi][LLj] = abs((A+B+C+D)/4)
                    img[HLi][HLj] = abs((A-B+C-D)/4)*10
                    img[LHi][LHj] = abs((A+B-C-D)/4)*10
                    img[HHi][HHj] = abs((A-B-C+D)/4)*10
                    LLj += 1
                    HLj += 1
                    LHj += 1
                    HHj += 1
                
                LLi += 1
                HLi += 1
                LHi += 1
                HHi += 1
            
            wavelet_img = img

        wavelet_img = np.uint8(wavelet_img)
        cv2.imwrite('wavelet.jpg', wavelet_img)
        wavelet_img = cv2.imread('wavelet.jpg')
        os.remove("wavelet.jpg")
        return wavelet_img

    def Histogram_Equalization(self,flag=0):
        G = 256
        H = [0]*G
        g_min = 1000000

        for i in range(self.height):
            for j in range(self.width):
                H[int(self.gray_img[i][j])] += 1
                if(H[int(self.gray_img[i][j])] > 0 and g_min > H[int(self.gray_img[i][j])]):
                    g_min = H[int(self.gray_img[i][j])]

        H_c = [0]*G
        H_c[0] = H[0]
        for g in range(1,G):
            H_c[g] = H_c[g-1] + H[g]
        
        H_min = H_c[g_min]
        T = []
        for g in range(G):
            if(H_c[g] - H_min < 0):
                T.append(np.around(0/(self.height*self.width - H_min)*(G-1)))
            else:
                T.append(np.around((H_c[g] - H_min)/(self.height*self.width - H_min)*(G-1)))
        
        g_q = np.copy(self.gray_img)
            
        for i in range(self.height):
            for j in range(self.width):
                g_q[i][j] = T[int(self.gray_img[i][j])]
        
        if(flag != 0):
            return g_q
        g_q = np.uint8(g_q)
        
        his = [0]*256
        for i in range(self.height):
            for j in range(self.width):
                his[g_q[i][j]] += 1
        his = np.array(his)
        his_dict = {}
        for i in range(256):
            his_dict[i] = his[i]

        myList = his_dict.items()
        myList = sorted(myList) 
        x, y = zip(*myList) 
        plt.bar(x, y)
        # 显示横轴标签
        plt.xlabel("Intensity")
        # 显示纵轴标签
        plt.ylabel("Frequency")
        # 显示图标题
        plt.title("histogram")
        #plt.show()
        plt.savefig('histogram_equalization.jpg')
        plt.clf()
        histogram_equalization = cv2.imread('histogram_equalization.jpg',cv2.IMREAD_COLOR)
        os.remove("histogram_equalization.jpg")

        cv2.imwrite('equalization.jpg', g_q)
        g_q = cv2.imread('equalization.jpg')
        os.remove("equalization.jpg")
        
        

        #cv2.imshow(histogram_equalization)
        return g_q,histogram_equalization

    def conv3(self,kernel,flag=0):
        if(flag == 2):
            sum = 0
            for i in range(3):
                for j in range(3):
                    sum += kernel[i][j]
            if(sum != 0):
                for i in range(3):
                    for j in range(3):
                        kernel[i][j] /= sum
            conv = cv2.filter2D(self.gray_img,-1,kernel=np.array(kernel))
            return conv

        sum = 0
        for i in range(3):
            for j in range(3):
                sum += kernel[i][j]
        if(sum != 0):
            for i in range(3):
                for j in range(3):
                    kernel[i][j] /= sum
        
        gray_img = self.gray_img
        height = self.height+2
        width = self.width+2
        gray_img = np.pad(gray_img,((1,1),(1,1)),'constant',constant_values = (0,0))
        conv = []
        for i in range(height-2):
            conv1 = []
            for j in range(width-2):
                res = 0
                for k1 in range(3):
                    for k2 in range(3):
                        if(i+k1 < height and j+k2 < width):
                            res += kernel[k1][k2]*gray_img[i+k1][j+k2]
                conv1.append(abs(res))
            conv.append(conv1)
        if(flag != 0):
            return conv

        conv = np.uint8(conv)
        cv2.imwrite('conv.jpg', conv)
        conv = cv2.imread('conv.jpg')
        os.remove("conv.jpg")
        return conv

    def conv5(self,kernel,flag=0):
        if(flag == 2):
            sum = 0
            for i in range(5):
                for j in range(5):
                    sum += kernel[i][j]
            if(sum != 0):
                for i in range(5):
                    for j in range(5):
                        kernel[i][j] /= sum
            conv = cv2.filter2D(self.gray_img,-1,kernel=np.rot90(np.array(kernel),2))
            return conv
        sum = 0
        for i in range(5):
            for j in range(5):
                sum += kernel[i][j]
        if(sum != 0):
            for i in range(5):
                for j in range(5):
                    kernel[i][j] /= sum

        gray_img = self.gray_img
        height = self.height+2
        width = self.width+2
        gray_img = np.pad(gray_img,((1,1),(1,1)),'constant',constant_values = (0,0))
        conv = []
        for i in range(height-4):
            conv1 = []
            for j in range(width-4):
                res = 0
                for k1 in range(5):
                    for k2 in range(5):
                        if(i+k1 < height and j+k2 < width):
                            res += kernel[k1][k2]*gray_img[i+k1][j+k2]
                conv1.append(abs(res))
            conv.append(conv1)
        
        if(flag != 0):
            return conv
        conv = np.uint8(conv)
        cv2.imwrite('conv.jpg', conv)
        conv = cv2.imread('conv.jpg')
        os.remove("conv.jpg")
        return conv

    def Binary(self,flag=0):
        max_pixel = 0
        min_pixel = 300
        for i in range(self.height):
            for j in range(self.width):
                if(max_pixel < self.gray_img[i][j]):
                    max_pixel = self.gray_img[i][j]
                if(min_pixel > self.gray_img[i][j]):
                    min_pixel = self.gray_img[i][j]
        
        T = int((max_pixel+min_pixel) / 2)
 
        Tn = 1000
        while(Tn != T):
            Osum = 0
            Ocount = 0
            Bsum = 0
            Bcount = 0
            for i in range(self.height):
                for j in range(self.width):
                    if(T < self.gray_img[i][j]):
                        Osum += self.gray_img[i][j]
                        Ocount += 1
                    if(T > self.gray_img[i][j]):
                        Bsum += self.gray_img[i][j]
                        Bcount += 1
            
            Tn = int((Osum/Ocount + Bsum/Bcount)/2)
            T = Tn

        for i in range(self.height):
            for j in range(self.width):
                if(self.gray_img[i][j]<T):
                    self.gray_img[i][j] = 0
                else:
                    self.gray_img[i][j] = 255
                #self.gray_img[i][j] = 255 if self.gray_img[i][j]<T else 0
        if(flag != 0):
            return self.gray_img
        self.gray_img = np.uint8(self.gray_img)
        cv2.imwrite('img.jpg', self.gray_img)
        self.gray_img = cv2.imread('img.jpg')
        os.remove("img.jpg")
        return self.gray_img
    
    def Binary2(self,flag=0):
        max_pixel = 0
        min_pixel = 300
        for i in range(self.height):
            for j in range(self.width):
                if(max_pixel < self.gray_img[i][j]):
                    max_pixel = self.gray_img[i][j]
                if(min_pixel > self.gray_img[i][j]):
                    min_pixel = self.gray_img[i][j]
        
        T = int((max_pixel+min_pixel) / 2)

        for i in range(self.height):
            for j in range(self.width):
                if(self.gray_img[i][j]<T):
                    self.gray_img[i][j] = 255
                else:
                    self.gray_img[i][j] = 0

        if(flag != 0):
            return self.gray_img
        self.gray_img = np.uint8(self.gray_img)
        cv2.imwrite('img.jpg', self.gray_img)
        self.gray_img = cv2.imread('img.jpg')
        os.remove("img.jpg")
        return self.gray_img
        
    def Dilation(self,Dilation_size,flag=0):#膨脹處理
        img = np.copy(self.gray_img)
        for i in range(self.height):
            for j in range(self.width):
                dflag = 0
                if(j-Dilation_size>0 and j+Dilation_size<self.width):
                    for k in range(Dilation_size):
                        if(int(self.gray_img[i][k+j]) == 255):
                            dflag = 1
                            break
                    if(dflag == 1):
                        for k in range(Dilation_size):
                            img[i][k+j] = 255    
        if(flag != 0):
            return img
        img = np.uint8(img)
        cv2.imwrite('img.jpg', img)
        img = cv2.imread('img.jpg')
        os.remove("img.jpg")
        return img

    def Erosion(self,Erosion_size,flag=0):#侵蝕處理
        img = np.copy(self.gray_img)
        for i in range(self.height):
            for j in range(self.width):
                dflag = 0
                if(j-Erosion_size>0 and j+Erosion_size<self.width):
                    for k in range(Erosion_size):
                        if(int(self.gray_img[i][k+j]) == 0):
                            dflag = 1
                            break
                    if(dflag == 1):
                        for k in range(Erosion_size):
                            img[i][k+j] = 0      
        if(flag != 0):
            return img
        img = np.uint8(img)
        cv2.imwrite('img.jpg', img)
        img = cv2.imread('img.jpg')
        os.remove("img.jpg")
        return img

    def connected_components(self):
        def DFS1(i, j, count):
            if(i<0 or j<0 or i>=self.height or j>=self.width):
                return
            if(self.gray_img[i][j] == 0 or self.m[i][j] != 0):
                return
            self.m[i][j] = count
            if(self.max<count):
                self.ff = 1
                self.max = count
                self.maxi = i
                self.maxj = j

            DFS1(i-1, j,count+1)
            DFS1(i, j-1,count+1)
            DFS1(i+1, j,count+1)
            DFS1(i, j+1,count+1)
        def DFS2(max,i, j,visit):
            if(i<0 or j<0 or i>=self.height or j>=self.width):
                return
            if(self.m[i][j] == 0 or visit[i][j] != 0):
                return
            self.m[i][j] = max
            visit[i][j] = 1
            DFS2(max, i, j-1,visit)
            DFS2(max, i, j+1,visit)
            DFS2(max, i-1, j,visit)
            DFS2(max, i+1, j,visit)
            
        self.m = np.zeros((self.height,self.width))

        for i in range(self.height):
            for j in range(self.width):
                if(self.gray_img[i][j] == 255 and self.m[i][j] == 0):
                    DFS1(i,j,0)
                if(self.ff == 1):
                    self.comp_num += 1
                    self.comp_list.append([self.max,self.maxi,self.maxj])
                    self.max,self.maxi,self.maxj = 0,0,0
                self.ff = 0
        

        i = 0
        j = 0
        while(i != self.comp_num):
            while(j != self.comp_num):
                if(self.comp_list[i][0] == self.comp_list[j][0]):
                    visit = np.zeros((self.height,self.width))
                    DFS2(0,self.comp_list[i][1],self.comp_list[i][2],visit)
                    self.comp_list.pop(i)
                    self.comp_num -= 1
                j += 1
            i += 1

        visit = np.zeros((self.height,self.width))
        for i in range(self.comp_num):   
            DFS2(self.comp_list[i][0],self.comp_list[i][1],self.comp_list[i][2],visit)      

        res = []
        for k in range(self.comp_num):
            si,sj = self.height+1,self.width+1
            ei,ej = -1,-1
            for i in range(self.height):
                for j in range(self.width):
                    if(self.m[i][j] == self.comp_list[k][0]):
                        if(si > i):
                            si = i
                        if(sj > j):
                            sj = j
                        if(ei < i):
                            ei = i
                        if(ej < j):
                            ej = j
            res.append([self.comp_list[k][0],[si,sj],[ei,ej]])

        return res
 
    def LPR(self):#License plate recognition
        def get_mask(comps):
            res = [0,[0,0],[0,0]]
            for comp in comps:
                h = comp[2][0]-comp[1][0]
                w = comp[2][1]-comp[1][1]
                if(comp[0] > 300 and 1.5 <= w/h <= 5):
                    if(res[2][0] < comp[2][0]):
                        res = comp

            mask = np.zeros((self.height,self.width))
            for i in range(res[1][0],res[2][0]+1):
                for j in range(res[1][1],res[2][1]+1):
                    mask[i][j] = 255
            return mask,res

        def get_lp(mask,img,res):
            h = res[2][0]-res[1][0]
            w = res[2][1]-res[1][1]
            LP,ii,jj,f = np.zeros((h+2,w+2)),0,0,0 #LP:單獨車牌image
            for i in range(self.height):
                for j in range(self.width):
                    if(mask[i][j] == 255):
                        LP[ii][jj] = img[i][j]
                        jj += 1
                        f = 1
                if(f == 1):
                    ii += 1
                    jj = 0
                    f = 0
            return LP
        
        def cut(img,scan_size):
            ret,img = cv2.threshold(np.uint8(img), 127, 255, cv2.THRESH_BINARY)
            h,w = len(img),len(img[0])
            midx,midy = int(h/2),int(w/2)
            
            leftj = 0
            rightj = w-1
            upflag = 1
            downflag = 1
            downi = h-1
            upi = 0
            for i in range(w):#left
                leftcount = 0
                for j in range(scan_size):
                    if(midx+j < h):
                        if(img[midx+j][i] == 255):
                            leftcount += 1
                    if(midx-j >=0):
                        if(img[midx-j][i] == 255):
                            leftcount += 1
                if(leftcount/h < 0.95 and leftcount > 2):
                    leftj = i
                    break
            
            for i in range(w-1,0,-1):#right
                rightcount = 0
                for j in range(scan_size):
                    if(midx+j < h):
                        if(img[midx+j][i] == 255):
                            rightcount += 1
                    if(midx-j >=0):
                        if(img[midx-j][i] == 255):
                            rightcount += 1
                if(rightcount/h < 0.95 and rightcount > 2):
                    rightj = i
                    break
                    
            ch = downi-upi
            cw = rightj-leftj
            #print(h,w,ch,cw)
            LP,ii,jj,f = np.zeros((ch,cw)),0,0,0 #LP:單獨車牌image
            for i in range(upi,downi):
                for j in range(leftj,rightj):
                    LP[ii][jj] = img[i][j]
                    jj += 1
                ii += 1
                jj = 0
            img = LP 

            for i in range(int(h/2)):
                upcount = 0
                downcount = 0
                for j in range(int(scan_size/2)):
                    #print(midx+i,midy-j,midy+j,h)
                    if(midx+i < ch and midy-j >= 0):
                        if(img[midx+i][midy-j] == 255 and upflag == 1):
                            upcount += 1
                    if(midx+i < ch and midy+j < cw):
                        if(img[midx+i][midy+j] == 255 and upflag == 1):
                            upcount += 1

                for j in range(int(scan_size/2)):
                    if(midx-i >= 0 and midy+j < cw):
                        if(img[midx-i][midy+j] == 255 and downflag == 1):
                            downcount += 1
                    if(midx-i >= 0 and midy-j >= 0):
                        if(img[midx-i][midy-j] == 255 and downflag == 1):
                            downcount += 1

                if(upcount/w > 0.85):
                    upi = midx+i
                    upflag = 0
                if(upcount/w > 0.85):
                    downi = midx-i
                    downflag = 0
        
            ch = downi-upi
            cw = rightj-leftj
            LP,ii,jj = np.zeros((ch,cw)),0,0#LP:單獨車牌image
            for i in range(upi,downi):
                for j in range(len(img[0])):
                    LP[ii][jj] = img[i][j]
                    jj += 1
                ii += 1
                jj = 0
        
            res = [upi,leftj,h-downi,w-rightj]
            return LP,res


        smooth_kernel3 = [[1,1,1],[1,1,1],[1,1,1]]
        smooth_kernel5 = [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]
        #edge_detection_kernel = [[0,-1,-1],[1,0,-1],[1,1,0]]
        edge_detection_kernel = [[-1,0,1],[-1,0,1],[-1,0,1]]

        
        self.gray_img = cv2.resize(self.gray_img, (260, 200), interpolation=cv2.INTER_NEAREST)
        original_img = cv2.resize(self.img, (260, 200), interpolation=cv2.INTER_NEAREST)
        self.height, self.width = len(self.gray_img),len(self.gray_img[0])
        img = np.copy(self.gray_img)
        #cv2.imshow('img1',np.uint8(self.gray_img))

        self.gray_img = self.conv3(smooth_kernel3,1)
        self.height, self.width = len(self.gray_img),len(self.gray_img[0])
        #plt.imshow(np.uint8(self.gray_img))
        #plt.show()
        #cv2.imshow('img2',np.uint8(self.gray_img))
        print('complete 3smooth')
        

        self.gray_img = self.conv3(edge_detection_kernel,1)
        self.height, self.width = len(self.gray_img),len(self.gray_img[0])
        #plt.imshow(np.uint8(self.gray_img))
        #plt.show()
        #cv2.imshow('img3',np.uint8(self.gray_img))
        cv2.imwrite('edge_detection2.jpg', np.uint8(self.gray_img))
        print('complete edge detection')

        ret,self.gray_img = cv2.threshold(np.uint8(self.gray_img), 127, 255, cv2.THRESH_BINARY)
        self.height, self.width = len(self.gray_img),len(self.gray_img[0])
        #cv2.imshow('img4',np.uint8(self.gray_img))
        print('complete Binary')

        self.gray_img = self.Dilation(13,1)
        self.height, self.width = len(self.gray_img),len(self.gray_img[0])
        #plt.imshow(np.uint8(self.gray_img))
        #plt.show()
        #cv2.imshow('img5',np.uint8(self.gray_img))
        print('complete Dilation')

        self.gray_img = self.Erosion(7,1)
        self.height, self.width = len(self.gray_img),len(self.gray_img[0])
        #plt.imshow(np.uint8(self.gray_img))
        #plt.show()
        #cv2.imshow('img6',np.uint8(self.gray_img))
        cv2.imwrite('erosion.jpg', np.uint8(self.gray_img))
        print('complete Erosion')
        
        connected_components = self.connected_components()
        print('complete Connected Components')

        mask,res= get_mask(connected_components)
        #plt.imshow(np.uint8(mask))
        #plt.show()
        #cv2.imshow('mask',np.uint8(mask))
        cv2.imwrite('mask.jpg', np.uint8(mask))
        print('complete get mask')
        
        
        LP = get_lp(mask,img,res)
        #plt.imshow(np.uint8(LP))
        #plt.show()
        #cv2.imshow('LP only',np.uint8(LP))
        print('complete get License plate')
        
        new_LP,cut_res = cut(LP,10)
        #plt.imshow(np.uint8(new_LP))
        #plt.show()
        cv2.imshow('new LP',np.uint8(new_LP))

        res[1][0] = res[1][0] + cut_res[0]
        res[1][1] = res[1][1] + cut_res[1]
        res[2][0] = res[2][0] - cut_res[2]
        res[2][1] = res[2][1] - cut_res[3]

        mask = np.zeros((self.height,self.width))
        for i in range(res[1][0],res[2][0]+1):
            for j in range(res[1][1],res[2][1]+1):
                mask[i][j] = 255
                
        
        output_img = cv2.rectangle(original_img,(res[1][1],res[1][0]),(res[2][1],res[2][0]),(0,0,255),2)
        ans = pytesseract.image_to_string(np.uint8(new_LP), lang="eng")

        if(len(ans) < 4):
            print(ans)
            ans = pytesseract.image_to_string(np.uint8(LP), lang="eng")
            print(ans)
            if(len(ans) < 4):
                ans = 'orz'
        
        print(ans)

        output_img = cv2.putText(output_img, ans, (res[1][1],res[1][0]), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
        for i in range(self.height):
            for j in range(self.width):
                if(mask[i][j] == 0):
                    img[i][j] = 255
                    
        self.gray_img = img
        self.gray_img = np.uint8(self.gray_img)

        #cv2.imwrite('process.jpg', self.gray_img)
        cv2.imwrite('process.jpg', output_img)
        output_img = cv2.imread('process.jpg')
        os.remove("process.jpg")
        return output_img

    def Auto_LPR(self):#License plate recognition
        def get_mask(comps):
            res = [0,[0,0],[0,0]]
            for comp in comps:
                h = comp[2][0]-comp[1][0]
                w = comp[2][1]-comp[1][1]
                if(comp[0] > 300 and 1.5 <= w/h <= 5):
                    if(res[2][0] < comp[2][0]):
                        res = comp

            mask = np.zeros((self.height,self.width))
            for i in range(res[1][0],res[2][0]+1):
                for j in range(res[1][1],res[2][1]+1):
                    mask[i][j] = 255
            return mask,res

        def get_lp(mask,img,res):
            h = res[2][0]-res[1][0]
            w = res[2][1]-res[1][1]
            LP,ii,jj,f = np.zeros((h+2,w+2)),0,0,0 #LP:單獨車牌image
            for i in range(self.height):
                for j in range(self.width):
                    if(mask[i][j] == 255):
                        LP[ii][jj] = img[i][j]
                        jj += 1
                        f = 1
                if(f == 1):
                    ii += 1
                    jj = 0
                    f = 0
            return LP
        
        def cut(img,scan_size):
            ret,img = cv2.threshold(np.uint8(img), 127, 255, cv2.THRESH_BINARY)
            h,w = len(img),len(img[0])
            midx,midy = int(h/2),int(w/2)
            
            leftj = 0
            rightj = w-1
            upflag = 1
            downflag = 1
            downi = h-1
            upi = 0
            for i in range(w):#left
                leftcount = 0
                for j in range(scan_size):
                    if(midx+j < h):
                        if(img[midx+j][i] == 255):
                            leftcount += 1
                    if(midx-j >=0):
                        if(img[midx-j][i] == 255):
                            leftcount += 1
                if(leftcount/h < 0.95 and leftcount > 2):
                    leftj = i
                    break
            
            for i in range(w-1,0,-1):#right
                rightcount = 0
                for j in range(scan_size):
                    if(midx+j < h):
                        if(img[midx+j][i] == 255):
                            rightcount += 1
                    if(midx-j >=0):
                        if(img[midx-j][i] == 255):
                            rightcount += 1
                if(rightcount/h < 0.95 and rightcount > 2):
                    rightj = i
                    break
                    
            ch = downi-upi
            cw = rightj-leftj
            #print(h,w,ch,cw)
            LP,ii,jj,f = np.zeros((ch,cw)),0,0,0 #LP:單獨車牌image
            for i in range(upi,downi):
                for j in range(leftj,rightj):
                    LP[ii][jj] = img[i][j]
                    jj += 1
                ii += 1
                jj = 0
            img = LP 

            for i in range(int(h/2)):
                upcount = 0
                downcount = 0
                for j in range(int(scan_size/2)):
                    #print(midx+i,midy-j,midy+j,h)
                    if(midx+i < ch and midy-j >= 0):
                        if(img[midx+i][midy-j] == 255 and upflag == 1):
                            upcount += 1
                    if(midx+i < ch and midy+j < cw):
                        if(img[midx+i][midy+j] == 255 and upflag == 1):
                            upcount += 1

                for j in range(int(scan_size/2)):
                    if(midx-i >= 0 and midy+j < cw):
                        if(img[midx-i][midy+j] == 255 and downflag == 1):
                            downcount += 1
                    if(midx-i >= 0 and midy-j >= 0):
                        if(img[midx-i][midy-j] == 255 and downflag == 1):
                            downcount += 1

                if(upcount/w > 0.85):
                    upi = midx+i
                    upflag = 0
                if(upcount/w > 0.85):
                    downi = midx-i
                    downflag = 0
        
            ch = downi-upi
            cw = rightj-leftj
            LP,ii,jj = np.zeros((ch,cw)),0,0#LP:單獨車牌image
            for i in range(upi,downi):
                for j in range(len(img[0])):
                    LP[ii][jj] = img[i][j]
                    jj += 1
                ii += 1
                jj = 0
        
            res = [upi,leftj,h-downi,w-rightj]
            return LP,res


        smooth_kernel3 = [[1,1,1],[1,1,1],[1,1,1]]
        smooth_kernel5 = [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]
        #edge_detection_kernel = [[0,-1,-1],[1,0,-1],[1,1,0]]
        edge_detection_kernel = [[-1,0,1],[-1,0,1],[-1,0,1]]

        
        self.gray_img = cv2.resize(self.gray_img, (260, 200), interpolation=cv2.INTER_NEAREST)
        original_img = cv2.resize(self.img, (260, 200), interpolation=cv2.INTER_NEAREST)
        self.height, self.width = len(self.gray_img),len(self.gray_img[0])
        img = np.copy(self.gray_img)
        #cv2.imshow('img1',np.uint8(self.gray_img))

        self.gray_img = self.conv3(smooth_kernel3,1)
        self.height, self.width = len(self.gray_img),len(self.gray_img[0])
        #print('complete 3smooth')
        

        self.gray_img = self.conv3(edge_detection_kernel,1)
        self.height, self.width = len(self.gray_img),len(self.gray_img[0])
        #plt.imshow(np.uint8(self.gray_img))
        #plt.show()
        #cv2.imshow('img3',np.uint8(self.gray_img))
        #print('complete edge detection')

        ret,self.gray_img = cv2.threshold(np.uint8(self.gray_img), 127, 255, cv2.THRESH_BINARY)
        self.height, self.width = len(self.gray_img),len(self.gray_img[0])
        #cv2.imshow('img4',np.uint8(self.gray_img))
        #print('complete Binary')

        self.gray_img = self.Dilation(13,1)
        self.height, self.width = len(self.gray_img),len(self.gray_img[0])
        #plt.imshow(np.uint8(self.gray_img))
        #plt.show()
        #cv2.imshow('img5',np.uint8(self.gray_img))
        #print('complete Dilation')

        self.gray_img = self.Erosion(7,1)
        self.height, self.width = len(self.gray_img),len(self.gray_img[0])
        #plt.imshow(np.uint8(self.gray_img))
        #plt.show()
        #cv2.imshow('img6',np.uint8(self.gray_img))
        #print('complete Erosion')
        
        connected_components = self.connected_components()
        #print('complete Connected Components')

        mask,res= get_mask(connected_components)
        #plt.imshow(np.uint8(mask))
        #plt.show()
        #cv2.imshow('mask',np.uint8(mask))
        #print('complete get mask')
        
        
        LP = get_lp(mask,img,res)
        #plt.imshow(np.uint8(LP))
        #plt.show()
        #cv2.imshow('LP only',np.uint8(LP))
        #print('complete get License plate')
        
        new_LP,cut_res = cut(LP,10)
        #plt.imshow(np.uint8(new_LP))
        #plt.show()
        #cv2.imshow('new LP',np.uint8(new_LP))

        res[1][0] = res[1][0] + cut_res[0]
        res[1][1] = res[1][1] + cut_res[1]
        res[2][0] = res[2][0] - cut_res[2]
        res[2][1] = res[2][1] - cut_res[3]

        mask = np.zeros((self.height,self.width))
        for i in range(res[1][0],res[2][0]+1):
            for j in range(res[1][1],res[2][1]+1):
                mask[i][j] = 255   
        
        output_img = cv2.rectangle(original_img,(res[1][1],res[1][0]),(res[2][1],res[2][0]),(0,0,255),2)
        ans = pytesseract.image_to_string(np.uint8(new_LP), lang="eng")

        if(len(ans) < 4):
            #print(ans)
            ans = pytesseract.image_to_string(np.uint8(LP), lang="eng")
            #print(ans)
            if(len(ans) < 4):
                ans = 'orz'
        
        return ans
    

def AUTO_TEST():
    PATH = './Image/'
    acc = {}
    acc_text = []
    f = []
    for i in range(1,101):
        img = cv2.imread(PATH + str(i) + '.jpg')
        try:
            LP_text = IMG(img).Auto_LPR()
        except:
            LP_text = 'WTF'
        
        if(LP_text == 'orz'):
            print(i,'Fall:',LP_text)
        elif(LP_text == 'WTF'):
            f.append(i)
            print(i,'WTF')
        else:
            print(i,'Correct:',LP_text)
            acc[i] = LP_text

    print(len(acc))
    print(acc)
    print(acc_text)
    print('')
    print('f:',f)
if __name__ == "__main__":
    AUTO_TEST()