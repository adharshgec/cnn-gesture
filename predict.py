import cv2 as cv
import math
import time

import numpy as np
from keras.preprocessing import image
from keras.models import load_model

classifier = load_model('gesture.h5')



cap = cv.VideoCapture(0)
while(cap.isOpened()):
	ret, img = cap.read()
	cv.rectangle(img, (300,300), (100,100), (0,255,0),0)
	crop_img = img[100:300, 100:300]
	crop_img = cv.resize(crop_img, (64,64),interpolation=cv.INTER_CUBIC)
	test_image = image.img_to_array(crop_img)
	test_image = np.expand_dims(test_image, axis = 0)
	result = classifier.predict(test_image)

	#all_img = np.hstack((drawing, crop_img))


	#if(result[0]!= null ):

	print(result)
	max_index = np.unravel_index(result.argmax(), result.shape)
	ascii_value=65+max_index[1]
	string= ''.join(chr(ascii_value))

	cv.putText(img,string, (250, 300), cv.FONT_HERSHEY_SIMPLEX, 2, (255,0,0),2,cv.LINE_AA)
	cv.imshow('Gesture', img)
	k = cv.waitKey(10)
	if k == 27:
		break

	#print(result[0])
# #cv.NamedWindow("camera", 1)
