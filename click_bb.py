#!/usr/bin/env python3.5

import sys
import numpy as np
import cv2
import glob

upper_left_point = None
lower_right_point = None
current_mouse_pos = None

def click(event, x, y, flags, param):
	global upper_left_point
	global lower_right_point
	global current_mouse_pos
	
	if event == cv2.EVENT_LBUTTONDOWN:
		upper_left_point = [x, y]
	
	if event == cv2.EVENT_LBUTTONUP:
		lower_right_point = [x, y]

	if event == cv2.EVENT_MOUSEMOVE:
		current_mouse_pos = [x, y]

def click_bb_on_image(image):
	global upper_left_point
	global lower_right_point
	global current_mouse_pos

	upper_left_point = None
	lower_right_point = None

	temp_image = image.copy()	
	cv2.imshow("image", temp_image)

	while True:
		temp_image = image.copy()	
		if upper_left_point != None:
			if lower_right_point != None:
				cv2.rectangle(temp_image, tuple(upper_left_point), tuple(lower_right_point), (200,0,0),1)
			elif current_mouse_pos != None:
				cv2.rectangle(temp_image, tuple(upper_left_point), tuple(current_mouse_pos), (200,0,0),1)


		cv2.imshow("image", temp_image)
		key = cv2.waitKey(1) & 0xFF

		if key == ord(" ") and upper_left_point != None and lower_right_point != None:
			return upper_left_point, lower_right_point
		if key == ord("q"):
			exit(0)
	


	


def main():
	all_bb = []
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", click)
	images = glob.glob('/user/HS204/m09113/facer2vm_project_area/data/300VW_Dataset_2015_12_14/*/frames/000001.png')
	output_file_path = '/user/HS204/m09113/facer2vm_project_area/data/300VW_Dataset_2015_12_14/bb_clicked_philipp.log'
	for i, image_path in enumerate(images):
		print ('image',image_path,'(',i,'of',len(images),')')
		image     = cv2.imread(image_path)
		upper_left_point, lower_right_point = click_bb_on_image(image)
		all_bb.append([upper_left_point[0], upper_left_point[1], lower_right_point[0], lower_right_point[1]])
		#print (upper_left_point, lower_right_point)
		open(output_file_path, 'a').write(str(image_path)+' '+str(upper_left_point[0])+' '+str(upper_left_point[1])+' '+str(lower_right_point[0])+' '+str(lower_right_point[1])+'\n')
	cv2.destroyWindow("image")





	#now write lm file 
#	landmark_file = '/user/HS204/m09113/Downloads/face_synthesis/M1000_22_L0_V9R_N_small.pts'
#	with open(landmark_file, "w") as lf:
#		lf.write('version: 1\n')
#		lf.write('n_points: 68\n')
#		lf.write('{\n')
#		for landmark in klicked_landmarks:
#			lf.write(str(landmark[0])+" "+str(landmark[1])+"\n")
#		lf.write('}\n')
#	return x, y, w, h



if __name__ == "__main__":
    main()