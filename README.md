# Cs-ALTP
# Facial recognition using CS-ALTP
Feature descriptors based on LBP have achieved encouraging performance in image recognition. As enhanced versions of LBP, Local Ternary Pattern (LTP) and Center-symmetric LBP (CS-LBP) have been successfully applied to image recognition and matching. Both use a threshold to treat noise. However, it is difficult to manually set an appropriate threshold in LTP and CS-LBP. Here we propose an adaptive local feature descriptor for facial recognition.
First, inspired by Weber’s law, we introduce an adaptive local ternary model characteristic descriptor (ALTP) based on an automatic strategy selecting the LTP threshold;
Secondly, on the basis of ALTP, we also propose a method for describing central symmetry adaptive local characteristic (CS-ALTP) for facial recognition. CS-ALTP improves CS-LBP in two ways An automatic threshold is proposed based on Weber’s law; dual channel models are used to extract more discriminatory information. Experiments based on ENT data show that ALTP and CS-ALTP have good and robust recognition performance.
# Used libraries
* OpenCV : OpenCV (for Open Computer Vision) is a free graphic library, originally developed by Intel, specialized in real-time image processing.
* NumPy : The NumPy library (http://www.numpy.org/) allows numerical calculations with Python. It introduces an easier management of the tables of numbers.
* Matplotlib : Matplotlib is a library of the Python programming language for plotting and visualizing data in graph form. It can be combined with the NumPy and SciPy python libraries for scientific computation.
* SciPy : SciPy is a free, open-source Python library used for scientific and technical computing.
   SciPy contains modules for optimization, linear algebra, integration, interpolation, special functions, FFT, signal and image processing, ODE solvers and other common tasks in science and engineering.
# functions used :
* CalcHIST :

  In this function we calculate the histogram of a block of a matrix of size M*N , the maximum value of this matrix is 15 since in CS-ALTP the maximum value calculated by this method is 15.

  
* csaltp_CODE :

  in this function we calculate the CS-ALTP code of a block 3*3 that we are processing everything depends on a threshold that we have already calculated
  
* Upper_value and Lower_value :

  The code obtained in the csaltpCODE function, we will extract 2 codes (UPPER AND LOWER code), from the code obtained following an extract algorithm 
* descriptor_IMG :

  In this function we will work with 2 matrices (P Mat and N Mat),  mat contains the positive values (upper values)  mat contain the negative values (lower values)
* testin_or_tranin :

  first step, we will go through the image pixel by pixel with a block size 3*3, take for example the 1 block that we will make a safe calculation.the first step is the transfer of our block to a vector, Before starting the calculation, the value of the index 3 is swapped with 5, Then we use the previous functions to calculate the discreper of the image.

  If you are training on a DATASET save Each dicriptor in TXT file is named with the name of its image.
* CalcDIST :

  In this function we use the eucludian distance to calculate the distance between 2 discriptors.

  
# MAIN
First we start by entering the coefficient K(weber’s law) of the cs-altp method, then the program asks us if we want to work on a new dataset or on other already saved if we are in the case of a new dataset: the first step is to enter the path of our dataset, then we do the calculations based on the defined functions of our program to have the descriptors of our dataset and save them in a file named by FacesDATASet otherwise, if we are in the case of testing the facial recognition of an image tests, the program asks us to enter the KNN value we want, then it starts working in the dataset already saved in a FacesDATASet file.
And this facial recognition done by a calculation of input image descriptor, then calculate the distance of this descriptor with the other descriptors of the FacesDATASet dataset images.
Then with the KNN one can know the images closest to our image and our KNN is based on the number k entered by the user.
