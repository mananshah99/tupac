"""
Implements the calculation of the Tamura features.

"""
from scipy.stats.stats import kurtosis
import numpy as np
from scipy.signal.signaltools import convolve2d
from math import pi

H = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
V = np.array([[-1,-1,-1],[0,0,0], [1,1,1]])

class Tamura:
    @staticmethod
    def coarseness(img):
        (rows, columns, _) = img.shape
        A = np.zeros((6,rows,columns))
        S = np.zeros((rows,columns))
        #Step 1
        for row in range(rows):
            for column in range(columns):
                size = 1
                for count in range(6):
                    beginningi = max(row - (size - 1) / 2, 0)
                    beginningj = max(column - (size - 1) / 2, 0)
                    finali = min(beginningi + size, rows)
                    finalj = min(beginningj + size, columns)
                    A[count][row][column] = img[beginningi:finali, beginningj:finalj, :].mean()
                    size = size*2
        #Step 2 and 3
        for row in range(rows):
            for column in range(columns):
                size = 1
                arrayVal = np.zeros(12)
                for count in range(6):
                    beginningi = max(row - (size - 1) / 2, 0)
                    beginningj = max(column - (size - 1) / 2, 0)
                    finali = min(beginningi + size, rows-1)
                    finalj = min(beginningj + size, columns-1)
                    arrayVal[2*count] = np.abs(A[count][finali][column] - A[count][beginningi][column])
                    arrayVal[2*count+1] = np.abs(A[count][row][finalj] - A[count][row][beginningj])
                S[row][column] = np.max(arrayVal)
        #step 4
        return S.mean()
        
    @staticmethod
    def contrast(img):
        kurt = kurtosis(img,axis=None,fisher=False)
        var = img.var()
        return var / np.power(kurt, 1. / 4.)

    @staticmethod
    def directionality(img):
        newImg = img + 4
        convV = np.zeros(newImg.shape)
        convH = np.zeros(newImg.shape)
        for i in range(newImg.shape[2]):
            convV[:,:,i] = convolve2d(newImg[:,:,i],V,mode='same',fillvalue=1)
            convH[:,:,i] = convolve2d(newImg[:,:,i],H,mode='same',fillvalue=1)
        convV[convV == 0] = 0.1
        convH[convH == 0] = 0.1
        theta1 = pi/2. + np.arctan(convV/convH)
        theta2 = pi/2 + np.arctan(convH/convV)
        return [theta1.var(),theta2.var()]
