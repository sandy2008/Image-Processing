import cv2
import numpy as np
from matplotlib import pyplot as plt

def main1():
	#Distance Transformation
	img = cv2.imread('water_coins.jpg')
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	dist = cv2.distanceTransform(thresh,cv2.DIST_L2,3)

	cv2.imwrite("a.jpg",dist)

def main2():
	#Skeleton
	img = cv2.imread('water_coins.jpg',0)
	size = np.size(img)
	skel = np.zeros(img.shape,np.uint8)
 
	ret,img = cv2.threshold(img,127,255,0)
	element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	done = False
	
	while(not done):
		eroded = cv2.erode(img,element)
		temp = cv2.dilate(eroded,element)
		temp = cv2.subtract(img,temp)
		skel = cv2.bitwise_or(skel,temp)
		img = eroded.copy()
		
		zeros = size - cv2.countNonZero(img)
		if zeros==size:
			done = True
	
	cv2.imwrite("aa.jpg",skel)

def main3():
	#Otsu’s binarization method
	img = cv2.imread('water_coins.jpg', 0)
	ret2,thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	
	cv2.imwrite("otsu_output.jpg", thresh)
	
def main4():
	#order derivative (canny function concludes all Edge’s gradient and minVal and maxVal. Any edges with intensity gradient more than maxVal are sure to be edges and those below minVal are sure to be non-edges
	img = cv2.imread('water_coins.jpg',0)
	edges = cv2.Canny(img,100,200)
	
	cv2.imwrite("canny.jpg", edges)
	
def main5():
	#Prewit
	gray = cv2.imread("water_coins.jpg", 0)
	#Y
	kernel = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
	dst2 = cv2.filter2D(gray, cv2.CV_64F, kernel)
	# output
	cv2.imwrite("cnm.jpg", dst2)

def main6():
	#Sobel Edge Detection
	img = cv2.imread('water_coins.jpg')
	#Gray Scale
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#Sobel Filter in x-direction
	gray_sobelx = cv2.Sobel(gray,cv2.CV_32F,1,0)
	#Sobel Filter in y-direction
	gray_sobely = cv2.Sobel(gray,cv2.CV_32F,0,1)
	#8-bit integer conversation
	gray_abs_sobelx = cv2.convertScaleAbs(gray_sobelx) 
	gray_abs_sobely = cv2.convertScaleAbs(gray_sobely)
	#add weighted value
	gray_sobel_edge = cv2.addWeighted(gray_abs_sobelx,0.5,gray_abs_sobely,0.5,0) 
	#show edge
	cv2.imwrite('dsa.jpg',gray_sobel_edge)
	
def main7():
	#Roberts Operator
	img = cv2.imread('water_coins.jpg')
	kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
	kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
	img_prewittx = cv2.filter2D(img, -1, kernelx)
	img_prewitty = cv2.filter2D(img, -1, kernely)
	cv2.imwrite("dsadas.jpg", img_prewittx + img_prewitty)
	print(img_prewittx)
	
def main8():
	img = cv2.imread('a.jpg',0)
	LoG = cv2.Laplacian(img, cv2.CV_16S)
	minLoG = cv2.morphologyEx(LoG, cv2.MORPH_ERODE, np.ones((3,3)))
	maxLoG = cv2.morphologyEx(LoG, cv2.MORPH_DILATE, np.ones((3,3)))
	zeroCross = np.logical_or(np.logical_and(minLoG < 0,  LoG > 0), np.logical_and(maxLoG > 0, LoG < 0))
	print(255*zeroCross.astype(np.uint8))
	cv2.imwrite("dsx.jpg", 255*zeroCross.astype(np.uint8))

def main9():
	img = cv2.imread('water_coins.jpg')
	#Gray Scale
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray,75,150)
	lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=250)
	for line in lines:
		x1, y1, x2, y2 = line[0]
		cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
		
	cv2.imwrite('xsavs.jpg',img)
	
def main10():
	datafile = 'digits.png'
	img = cv2.imread(datafile)
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	
	cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
	
	x = np.array(cells)
	train = x[:, :50].reshape(-1, 400).astype(np.float32)
	test = x[:, 50:100].reshape(-1, 400).astype(np.float32)
	
	k = np.arange(10)
	train_labels = np.repeat(k, 250)[:, np.newaxis]
	test_labels = train_labels.copy()
	
	knn = cv2.ml.KNearest_create()
	knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
	ret, results, neighbours, dist = knn.findNearest(test, 5)
	
	matches = results == test_labels
	correct = np.count_nonzero(matches)
	accuracy = correct*100.0/results.size
	print(accuracy)
	
if __name__ == "__main__":
	main10()