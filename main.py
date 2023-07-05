import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import distance

# ---------------------initialitaion values-------------------------------------
Img_F = np.ones((3, 3))
Img_F_Vector = [1, 1, 1, 1, 1, 1, 1, 1, 1]
sizeF = len(Img_F)
sizeF2 = len(Img_F) // 2
N = len(Img_F_Vector)  # -----> For get length of kernel_vector N=9
n = int((N - 1) / 2)  # ------> To stop on the middle of V_kernel n=4


# *************************Functions**********************************************
# ---------------CALCULATE HISTOGRAM-------------------------------------------
def CalcHIST(MatIN):
    print("LOADING")
    rows = len(MatIN)
    columns = len(MatIN[0])
    hist_N = []
    k = 0
    while k <= 15:  # Use 15 the max because the max value can be calculate in cs altp
        h = 0
        for i in range(rows):
            for j in range(columns):
                if k == MatIN[i][j]:
                    h = h + 1
        hist_N.insert(k, h)
        k = k + 1
    return hist_N


# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# ---------------CS-ALTP Methode For Get CODE-------------------------------------------
def csaltp_CODE(Image_F_Vector):
    ALTP_V = [1, 1, 1, 1]  # ---> Create vector with intial 4 elements = 1
    for i in range(n):
        t = Image_F_Vector[(N - 1) - i] * k
        # threshold for Pi with Pi+(N/2)
        S = (Image_F_Vector[i] - Image_F_Vector[(N - 1) - i])
        if S >= t:
            ALTP_V[i] = 1
        elif abs(S) < t:
            ALTP_V[i] = 0
        elif S <= (-1 * t):
            ALTP_V[i] = -1

    return ALTP_V


# --------------- Upper & Lower code---------------
# ----------------Upper value ----------------------
def Upper_value(ALTP_V):
    U = 0
    Upper_L = []
    for i in range(n):
        if ALTP_V[i] == 1:
            Upper_L.append(1)
        else:
            Upper_L.append(0)
    for i in range(n):
        U += Upper_L[i] * 2 ** i
    return U


# --------------------------------------------------
# --------------Lower value-------------------------
def Lower_value(ALTP_V):
    L = 0
    Lower_L = []
    for i in range(n):
        if ALTP_V[i] == -1:
            Lower_L.append(1)
        else:
            Lower_L.append(0)
    for i in range(n):
        L += Lower_L[i] * 2 ** i
    return L


# ---------------------CALCULATE DESCRIPTOR OF IMAGE--------------------------------
def descriptor_IMG(PMat, NMat):
    descriptor_image = []
    P_Block = np.array_split(PMat, 30)
    N_Block = np.array_split(NMat, 30)
    for i in range(30):
        hist_P = CalcHIST(P_Block[i])
        hist_N = CalcHIST(N_Block[i])
        hist_P_N_Block = hist_P + hist_N
        # -------------------------GET FINAL DESCRIPTOR------------------------------------------------------
        descriptor_image.append(hist_P_N_Block)
    return descriptor_image


# ------------------------------------------
# ------------------------Testing Iamge OR Traning on DATASet-----------------------------------------
def testin_or_tranin(IMG, img, test):
    # ---------------------initialitaion-------------------------------------
    Img_K_Vector = []
    PosV_Mat = np.array(Image.new("L", (IMG.shape[1], IMG.shape[0])))
    NegV_Mat = np.array(Image.new("L", (IMG.shape[1], IMG.shape[0])))
    # -------------------------------------------------------------------------
    # ---------------get elements from image --------------------------------
    for m in range(sizeF2, IMG.shape[0] - sizeF2):
        for r in range(sizeF2, IMG.shape[1] - sizeF2):
            for k in range(3):
                for o in range(3):
                    Img_F[k, o] = IMG[m + k - sizeF2, r + o - sizeF2]
            # --------------------- Transfer une matrice vers un vecteur------------
            for i in range(sizeF):
                for j in range(sizeF):
                    Img_K_Vector.append(Img_F[i, j])
            # ----------SWITCH Vector[3] with Vector[5]
            permutation = Img_K_Vector[3]
            Img_K_Vector[3] = Img_K_Vector[5]
            Img_K_Vector[5] = permutation
            # ----------------------------------------------------------------------
            # --------------- # elements we are working in them --------------------------
            print("---------------Les elements-------------")
            for i in range(n):
                print(Img_K_Vector[i])
            print("---------------Les elements-------------")
            # -----------------Call Functions-----------------------------------------
            CS_Altp_Code = csaltp_CODE(Img_K_Vector)
            Img_K_Vector = []
            U_v = Upper_value(CS_Altp_Code)
            L_v = Lower_value(CS_Altp_Code)
            PosV_Mat[m][r] = U_v
            NegV_Mat[m][r] = L_v
    descriptor = descriptor_IMG(PosV_Mat, NegV_Mat)
    if test == 'NO':
        name = os.path.splitext(filename)[0]
        name_file = './FacesDATASet/' + name + '.txt'
        with open(name_file, 'wb') as f:
            for line in descriptor:
                np.savetxt(f, line, fmt='%d')
    return descriptor


# -----------------------------------------------------------------------------------------
# ---------------------------------Calculate Ecludien DISTANCE-----------------------------
def CalcDIST(IMGTest, IMGDATASET):
    distECLU = distance.euclidean(IMGTest, IMGDATASET)
    return distECLU


# -----------------------------------------------------------------------------------------
# ****************************************************************************************
# ************************main************************************************************
file_num = 0
k = float(input("Enter The Factor Of CS-ALTP\nEXAMPLE : 0.21\n"))
SelectOrd = int(input("FOR TRANING IN DATASET ----> 1\nFOR TESTING IMAGE ----> 2\n "))
if SelectOrd == 1:
    Test = 'NO'
    folder = input("Enter PATH OF DATASET\nEXAMPLE : ./Faces\n")
    for filename in os.listdir(folder):
        # chargement en mode NDG et conversion en matrice NUMPY
        imNDG = Image.open(os.path.join(folder, filename)).convert('L')
        imgMat = np.array(imNDG)  # convertion d l'image vers une matrice
        print(filename)
        testin_or_tranin(imgMat, file_num, Test)
if SelectOrd == 2:
    K = int(input("Enter The Value Of KNN\n"))
    Test = 'YES'
    PATH = input("Enter PATH OF IMAGE\nEXAMPLE : ./FacesTest/\n")
    IMG_Test = input("Enter NAME OF IMAGE\nEXAMPLE : 3.jpg\n")
    filename = IMG_Test
    imNDG = Image.open(PATH + IMG_Test).convert('L')
    imgMat = np.array(imNDG)  # convertion d l'image vers une matrice
    ImgTestfile = testin_or_tranin(imgMat, file_num, Test)
    ImgTest = []
    count = 0
    for listElem in ImgTestfile:
        for i in range(len(listElem)):
            ImgTest.append(listElem[i])
    Min_Dist = 99999
    DISTS_btw_imgs = {}
    for filename in os.listdir('./FacesDATASet/'):
        txtfile = os.path.join('./FacesDATASet/', filename)
        ImgDATASetTXT = []
        ImgDATASet = []
        with open(txtfile, 'r') as f:
            for line in f:
                ImgDATASetTXT.append(list(map(int, line.split())))
        f.close()
# -------------------KNN Methode For Get Nearest IMAGES---------------------------
        for i in range(len(ImgDATASetTXT)):
            ImgDATASet.append(ImgDATASetTXT[i][0])
        DIST_btw_imgs = CalcDIST(ImgTest, ImgDATASet)
        DISTS_btw_imgs[filename] = DIST_btw_imgs
    print("The Nearest Images with KNN = " + str(K) + " :")
    for i in range(K):
        min_file = str(min(DISTS_btw_imgs, key=DISTS_btw_imgs.get))
        name = os.path.splitext(min_file)[0]
        NAMEimg = './Faces/' + name + '.jpg'
        print(NAMEimg)
        imgshow = cv2.imread(NAMEimg)
        plt.imshow(imgshow)
        plt.show()
        del DISTS_btw_imgs[min_file]
