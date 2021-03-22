''' usage :
1- python3 face_detect.py -v videos/test1.mkv
2- python3 face_detect.py -i images/img3.jpg
'''
import argparse as arg
import time
import cv2 as cv
import numpy as np
from skin_seg import *


class Face_Detector():
	def __init__(self, skin_detect):
		#skin_detect is an object from skin_seg file
		self._skin_detect = skin_detect

	@property
	def skin_detect(self):
		# set skin_detect to be an immutable field/property
		return self._skin_detect

	def Detect_Face_Img(self, img, size1, size2):
		"""this method implements the skin detection algorithm to perform a face detection in a given image.
		-inputs: 
		img : BGR image (numpy array)
		size1 : the lower size of a rectangle/face(min size) (type tuple)
		size2 : the upper size of a rectangle/face(max size) (type tuple)
		-output:
		a numpy array with all faces coordinates in a picture.
		"""

		# get the RGB_H_CbCr representation of the image(for more info, please refer to skin_seg.py)
		skin_img = self._skin_detect.RGB_H_CbCr(img)
		contours, hierarchy = cv2.findContours(skin_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		#cv2	.drawContours(img, contours, -1, (0,255,0), 1)
		#cv2.imshow("faces",img)
		#if cv2.waitKey(0) & 0xFF == ord("q"):
		#	sys.exit(0)
		rects = []
		for c in contours:
			# get the bounding rect
			x, y, w, h = cv2.boundingRect(c)
			# draw a green rectangle to visualize the bounding rect
			if (w > size1[0] and h > size1[1]) and (w < size2[0] and h < size2[1]):
				#pinhole distance
				Distance1 = 11.5*(img.shape[1]/float(w))
				#camera distance
				Distance2 = 15.0*((img.shape[1] + 226.8)/float(w))
				# print("\npinhole distance = {:.2f} cm\ncamera distance = {:.2f} cm".format(Distance1,Distance2))
				# print("Width = {} \t Height = {}".format(w,h))
				rects.append(np.asarray([x,y,w,w*1.25], dtype=np.uint16))
		return rects


def open_img(arg_):
	mg_src = arg_["image"]
	img = cv2.imread(mg_src)
	img_arr = np.array(img, 'uint8')
	return img_arr


if __name__ == "__main__":
	"""
	if len(sys.argv) == 1:
		print("Please give me a file :Image/video !!!")
		print("\n Try Again, For more info type --help to see available options")
		sys.exit(0)
	in_arg = Arg_Parser()
	skin_detect = Skin_Detect()
	size1 = (40,40)
	size2 = (300,400)
	scale_factor = 3
	Face_Detect = Face_Detector(skin_detect)
	if in_arg["image"] != None:
		img = open_img(in_arg)
		rects = Face_Detect.Detect_Face_Img(img,size1,size2)
		print(rects)
		for i,r in enumerate(rects):
			x,y,w,h = r
			cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
		cv2.imshow("faces",img)
		if cv2.waitKey(0) & 0xFF == ord("q"):
			sys.exit(0)
	if in_arg["video"] != None:
		vid = open_vid(in_arg)
		Face_Detect.Detect_Face_Vid(vid,size1,size2,scale_factor)
	"""
	skin_detect = Skin_Detect()
	size1 = (70, 70)
	size2 = (300, 400)
	scale_factor = 3
	Face_Detect = Face_Detector(skin_detect)

	videoSensor = cv.VideoCapture(0)
	while True:
		status, frame = videoSensor.read()
		rects = Face_Detect.Detect_Face_Img(frame, size1, size2)
		for i, r in enumerate(rects):
			x, y, w, h = r
			center = (x + w // 2, y + h // 2)
			frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (175, 175, 175), -1)
		cv.imshow('Video frame - face detection', frame)
		if cv.waitKey(1) & 0xFF == ord('q'):
			break

