#!/usr/bin/env python3.5

import sys
import cv2
import glob

image_paths = sys.argv[1:]

print ('showing images:',image_paths)

for idx, image_path in enumerate(image_paths):
	image = cv2.imread(image_path, cv2.IMREAD_COLOR)
	windowname = '...'+image_path[-30:]
	cv2.namedWindow(windowname,cv2.WINDOW_NORMAL)
	cv2.resizeWindow(windowname, 300, 300)
	cv2.moveWindow(windowname, 10+idx*305, 100)
	cv2.imshow(windowname, image)

cv2.waitKey()
cv2.destroyAllWindows()