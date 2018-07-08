import cv2
import numpy as np
from matplotlib import pyplot as plt

def main1():
	#Averaging
	img = cv2.imread("in2_noise.jpg")
	kernel = np.ones((5,5),np.float32)/25
	dst = cv2.filter2D(img,-1,kernel)
	res = np.hstack((img,dst))
	cv2.imwrite("avg2_output.jpg", res)
	
def main2():
	#Median Filter
	img = cv2.imread("in2_noise.jpg")
	final = cv2.medianBlur(img, 5)
	res = np.hstack((img,final))
	cv2.imwrite("med2_output.jpg", res)
	
def main3():
	#Linear gray-level transform
	min_table = 100
	max_table = 110
	diff_table = max_table - min_table
	look_up_table = np.arange(256, dtype = 'uint8' )
	for i in range(0, min_table):
		look_up_table[i] = 0
	for i in range(min_table, max_table):
		look_up_table[i] = 255 * (i - min_table) / diff_table
	for i in range(max_table, 255):
		look_up_table[i] = 255
	img = cv2.imread('con4.jpg')
	img_contrast = cv2.LUT(img, look_up_table)
	res = np.hstack((img,img_contrast))
	cv2.imwrite("gl4.jpg", res)
	
def main4():
	#Histogram Equalization
	img = cv2.imread('con4.jpg',0)
	equ = cv2.equalizeHist(img)
	res = np.hstack((img,equ))
	cv2.imwrite('his4.jpg',res)

def main5():
	#Sharpening
	img = cv2.imread("in1.jpg")
	kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
	dst = cv2.filter2D(img,-1,kernel)
	res = np.hstack((img,dst))
	cv2.imwrite("shp1.jpg", res)

def main6():
	#Thresholding
	img = cv2.imread('in2.jpg',0)
	ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
	ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
	ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
	ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
	ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
	titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
	images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
	for i in range(6):
		plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
		plt.title(titles[i])
		plt.xticks([]),plt.yticks([])
	plt.show()
	
if __name__ == "__main__":
	main1()
	main2()
	main3()
	main4()
	main5()
	main6()