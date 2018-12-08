#!/usr/bin/env python3.5

import sys
sys.path.append('/user/HS204/m09113/eos_py/build/python/')
import eos
import numpy as np
import cv2

PATH_TO_EOS_SHARE = '/user/HS204/m09113/my_project_folder/mm_shapes_masks/3448_model/'
PATH_TO_LM_AND_IMG = '/user/HS204/m09113/eos/install/bin/data/'
#PATH_TO_LM_AND_IMG = '/user/HS204/m09113/Desktop'

OUTPUTFOLDER = '/user/HS204/m09113/my_project_folder/3dmm_video/images/'

model = eos.morphablemodel.load_model(PATH_TO_EOS_SHARE+"sfm_shape_3448.bin")
blendshapes = eos.morphablemodel.load_blendshapes(PATH_TO_EOS_SHARE+"expression_blendshapes_3448.bin")
landmark_mapper = eos.core.LandmarkMapper(PATH_TO_EOS_SHARE+'ibug_to_sfm.txt')
edge_topology = eos.morphablemodel.load_edge_topology(PATH_TO_EOS_SHARE+'sfm_3448_edge_topology.json')
contour_landmarks = eos.fitting.ContourLandmarks.load(PATH_TO_EOS_SHARE+'ibug_to_sfm.txt')
model_contour = eos.fitting.ModelContour.load(PATH_TO_EOS_SHARE+'model_contours.json')


klicked_landmarks=[]
current_landmark =[]

img_num=0

def read_pts(filename):
	"""A helper function to read ibug .pts landmarks from a file."""
	lines = open(filename).read().splitlines()
	lines = lines[3:71]

	landmarks = []
	for l in lines:
		coords = l.split()
		landmarks.append([float(coords[0])-1.0, float(coords[1])-1.0])#-1 as ibug pts files have 1 indexing

	return landmarks

def saveRendering(image, folder):
	global img_num
	print("writing image",img_num)
	cv2.imwrite(folder+format(img_num, '03d')+'.png', image)
	img_num+=1



def overlay(image, rendering):
	image_conf = (255-rendering[:,:,3])/255
	rendering_conf = rendering[:,:,3]/255
	rendering_on_image = image[:,:,:]* image_conf[:,:,None]+rendering[:,:,0:3]*rendering_conf[:,:,None]

	return rendering_on_image.astype(dtype="uint8")


def show_different_poses(image, isomap, number_of_steps, mesh, initial_pose, model_view_matrix_final, model_view_matrix_start=None):

	image_height, image_width = image.shape[:2]

	# rendering matrices starting point
	model_view_matrix_image = np.ascontiguousarray(initial_pose.get_modelview())
	projection_matrix_image = np.ascontiguousarray(initial_pose.get_projection())

	#set image specific translations
	model_view_matrix_final[0:2,3] = model_view_matrix_image[0:2,3]

	if model_view_matrix_start is not None:
		model_view_matrix_start[0:2,3] = model_view_matrix_image[0:2,3]
		model_view_matrix_image=model_view_matrix_start

	model_view_matrix_step = (model_view_matrix_final - model_view_matrix_image)/number_of_steps

	for step in range(number_of_steps):

		rendering, depth = eos.render.render(mesh, model_view_matrix_image+step*model_view_matrix_step, projection_matrix_image, image_width, image_height, isomap, True, False, False)


		rendering_on_image = overlay(image, rendering)

		cv2.imshow('rendering on image', rendering_on_image)
		saveRendering(rendering_on_image, OUTPUTFOLDER+'poses2/')
		cv2.waitKey(50)


def show_different_blendshapes(image, isomap, number_of_steps, pose, shape_coeffs, bs_coeff_start, bs_coeff_final):

	image_height, image_width = image.shape[:2]
	model_view_matrix_image = np.ascontiguousarray(pose.get_modelview())
	projection_matrix_image = np.ascontiguousarray(pose.get_projection())

	bs_coeff_step = (bs_coeff_final - bs_coeff_start)/number_of_steps

	for step in range(number_of_steps):
		mesh = eos.morphablemodel.draw_sample(model, blendshapes, shape_coeffs, bs_coeff_start+step*bs_coeff_step, [])

		rendering, depth = eos.render.render(mesh, model_view_matrix_image, projection_matrix_image, image_width, image_height, isomap, True, False, False)

		rendering_on_image = overlay(image, rendering)

		cv2.imshow('rendering on image', rendering_on_image)
		saveRendering(rendering_on_image, OUTPUTFOLDER+'blendshapes/')
		cv2.waitKey(50)


def show_different_shapes(image, isomap, number_of_steps, pose, shape_coeffs_start, shape_coeffs_final, bs_coeff):

	image_height, image_width = image.shape[:2]
	model_view_matrix_image = np.ascontiguousarray(pose.get_modelview())
	projection_matrix_image = np.ascontiguousarray(pose.get_projection())

	shape_coeff_step = (shape_coeffs_final - shape_coeffs_start)/ number_of_steps

	for step in range(number_of_steps):
		mesh = eos.morphablemodel.draw_sample(model, blendshapes, shape_coeffs_start+step*shape_coeff_step, bs_coeff, [])

		rendering, depth = eos.render.render(mesh, model_view_matrix_image, projection_matrix_image, image_width, image_height, isomap, True, False, False)

		rendering_on_image = overlay(image, rendering)

		cv2.imshow('rendering on image', rendering_on_image)
		saveRendering(rendering_on_image, OUTPUTFOLDER+'alphas/')
		cv2.waitKey(50)




def play_with_fitting():
	global img_num
	#landmarks = read_pts(PATH_TO_LM_AND_IMG+'image_0010.pts')
	#image     = cv2.imread(PATH_TO_LM_AND_IMG+'image_0010.png')
	landmarks = read_pts('/user/HS204/m09113/Desktop/DSC03124_small.pts')
	image     = cv2.imread('/user/HS204/m09113/Desktop/DSC03124_small.JPG')

	landmark_ids = list(map(str, range(1, 69))) # generates the numbers 1 to 68, as strings

	image_height, image_width = image.shape[:2]

	(mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(model, blendshapes,
			landmarks, landmark_ids, landmark_mapper,
			image_width, image_height, edge_topology, contour_landmarks, model_contour)

	isomap = eos.render.extract_texture(mesh, pose, image)
	#cv2.imwrite('/user/HS204/m09113/Desktop/DSC03124_small_isomap.JPG',isomap)

	rendering, depth = eos.render.render(mesh, np.ascontiguousarray(pose.get_modelview()), np.ascontiguousarray(pose.get_projection()), image_width, image_height, isomap, True, False, False)
	#print(np.ascontiguousarray(pose.get_modelview()))
	#exit(0)

	rendering_on_image = overlay(image, rendering)
	print("org image")
	cv2.imshow('rendering on image', rendering_on_image)
	#saveRendering(rendering_on_image, OUTPUTFOLDER)
	cv2.waitKey(4000)

	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image_gray = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

	rendering_on_image = overlay(image_gray, rendering)
	print ("gray brackground")
	cv2.imshow('rendering on image', rendering_on_image)
	#saveRendering(rendering_on_image, OUTPUTFOLDER)
	cv2.waitKey(4000)

	# show different alphas
	print ("different alphas")
	img_num=1
	shape_coeff_offset_old = np.array([0]*63)
	shape_coeff_offset_new = np.array([0]*63)
	for i in range(10):
		shape_coeff_offset_new = np.array([0]*i+[1.5]+[0]*(62-i))
	#	show_different_shapes(image_gray, isomap, 10, pose, shape_coeffs-shape_coeff_offset_old, shape_coeffs+shape_coeff_offset_new, blendshape_coeffs)
	#	show_different_shapes(image_gray, isomap, 10, pose, shape_coeffs+shape_coeff_offset_new, shape_coeffs-shape_coeff_offset_new, blendshape_coeffs)
		shape_coeff_offset_old=shape_coeff_offset_new
	#show_different_shapes(image_gray, isomap, 10, pose, shape_coeffs+shape_coeff_offset_new, shape_coeffs, blendshape_coeffs)
	cv2.waitKey(4000)	


	# show different blendshapes
	print ("different blendshapes")
	img_num=1
	#blendshapes angry, surprised, smile, laghing, sad, shocked
	strength=1.0
	expressions = [ blendshape_coeffs,
				 np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
				 np.array([strength*2, 0.0, 0.0, 0.0, 0.0, 0.0]),
				 #np.array([0.0, strength, 0.0, 0.0, 0.0, 0.0]),
				 np.array([0.0, 0.0, strength, 0.0, 0.0, 0.0]),
				 np.array([0.0, 0.0, 0.0, strength, 0.0, 0.0]),
				 np.array([0.0, 0.0, 0.0, 0.0, strength, 0.0]),
				 np.array([0.0, 0.0, 0.0, 0.0, 0.0, strength]),
				 blendshape_coeffs,
				]
	#for exp in range(len(expressions)-1):
		#show_different_blendshapes(image_gray, isomap, 20, pose, shape_coeffs, expressions[exp], expressions[exp+1])


	# model view matrices
	print ("different poses")
	img_num=1
	#frontal
	#model_view_matrix_final = np.identity(4)
	#looking left example image
	#model_view_matrix_final = np.array([[  8.77149224e-01,   4.23516445e-02,  -4.78346676e-01,   3.09561890e+02],
 	#							[ -9.75291878e-02,   9.91054893e-01,  -9.10947025e-02,   3.12726501e+02],
 	#							[  4.70209807e-01,   1.26556411e-01,   8.73433590e-01,   0.00000000e+00],
 	#							[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

	model_view_matrix_right = np.array([[  np.cos(np.radians(45)),   0,  np.sin(np.radians(45)),   0],
 								[ 0,   1,  0,   0],
 								[  -np.sin(np.radians(45)),   0,   np.cos(np.radians(45)),   0.00000000e+00],
 								[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

	model_view_matrix_left = np.array([[  np.cos(np.radians(-45)),   0,  np.sin(np.radians(-45)),   0],
 								[ 0,   1,  0,   0],
 								[  -np.sin(np.radians(-45)),   0,   np.cos(np.radians(-45)),   0.00000000e+00],
 								[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])


	show_different_poses(image_gray, isomap, 20, mesh, pose, model_view_matrix_left)
	show_different_poses(image_gray, isomap, 20, mesh, pose, np.identity(4), model_view_matrix_left)
	show_different_poses(image_gray, isomap, 20, mesh, pose, model_view_matrix_right, np.identity(4))
	show_different_poses(image_gray, isomap, 20, mesh, pose, np.ascontiguousarray(pose.get_modelview()), model_view_matrix_right)

	cv2.waitKey(2000)


def draw_lms_on_image(image, landmarks):
	
	for idx, landmark in enumerate(landmarks):
		position = np.array(landmark,dtype="uint32")
		#print (tuple(position))
		cv2.circle(image, tuple(position), 2, (0, 200, 0),2)
		#cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]]) → None
		cv2.putText(image, str(idx+1), tuple(position+2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0))
		#cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) → None

def show_lms_on_image():
	#landmarks = read_pts(PATH_TO_LM_AND_IMG+'image_0010.pts')
	#image     = cv2.imread(PATH_TO_LM_AND_IMG+'image_0010.png')
	#landmarks = read_pts('/user/HS204/m09113/Desktop/DSC03124_small.pts')
	#image     = cv2.imread('/user/HS204/m09113/Desktop/DSC03124_small.JPG')
	landmarks  = read_pts('/user/HS204/m09113/my_project_folder/CASIA_webface/landmarks/0000045/003.pts')
	image      = cv2.imread('//vol/vssp/datasets/still/CASIA-WebFace/CASIA-WebFace/0000045/003.jpg')

	draw_lms_on_image(image, landmarks)
	#cv2.imwrite('/user/HS204/m09113/Desktop/DSC03124_small_lms.JPG',image)
	cv2.imshow('image with lms', image)


def click(event, x, y, flags, param):
	global current_landmark
	
	if event == cv2.EVENT_LBUTTONUP:
		current_landmark = [x, y]

def klick_landmarks_on_image():
	global current_landmark, klicked_landmarks

	cv2.namedWindow("image")
	cv2.setMouseCallback("image", click)

	
	show_lms_on_image()
	image     = cv2.imread('/user/HS204/m09113/Downloads/face_synthesis/M1000_22_L0_V9R_N_small.JPG')
	for lm_idx in range(68):
		while True:
			temp_image = image.copy()
			lms_to_be_shown = klicked_landmarks#+current_landmark
			if len(current_landmark)>0:
				lms_to_be_shown =klicked_landmarks + [current_landmark]

			if len(lms_to_be_shown)>0:
				draw_lms_on_image(temp_image, lms_to_be_shown)

			cv2.imshow("image", temp_image)
			key = cv2.waitKey(1) & 0xFF

			if key == ord(" "):
				if len(current_landmark)>0:
					klicked_landmarks.append(current_landmark)
					break
			if key == ord("q"):
				return 0
		current_landmark=[]
	cv2.destroyWindow("image")

	#now write lm file 
	landmark_file = '/user/HS204/m09113/Downloads/face_synthesis/M1000_22_L0_V9R_N_small.pts'
	with open(landmark_file, "w") as lf:
		lf.write('version: 1\n')
		lf.write('n_points: 68\n')
		lf.write('{\n')
		for landmark in klicked_landmarks:
			lf.write(str(landmark[0])+" "+str(landmark[1])+"\n")
		lf.write('}\n')





if __name__ == "__main__":
	#play_with_fitting()
	show_lms_on_image()
	#klick_landmarks_on_image()

	cv2.waitKey()
	cv2.destroyAllWindows()
