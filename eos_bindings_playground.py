#!/usr/bin/env python3.5

import sys
sys.path.append('/user/HS204/m09113/eos_py/build/python/')
import eos
import numpy as np
import cv2
import transformations

PATH_TO_EOS_SHARE = '/user/HS204/m09113/eos/install/share/'
PATH_TO_LM_AND_IMG = '/user/HS204/m09113/eos/install/bin/data/'

model = eos.morphablemodel.load_model(PATH_TO_EOS_SHARE+"sfm_shape_3448.bin")
blendshapes = eos.morphablemodel.load_blendshapes(PATH_TO_EOS_SHARE+"expression_blendshapes_3448.bin")
landmark_mapper = eos.core.LandmarkMapper(PATH_TO_EOS_SHARE+'ibug_to_sfm.txt')
edge_topology = eos.morphablemodel.load_edge_topology(PATH_TO_EOS_SHARE+'sfm_3448_edge_topology.json')
contour_landmarks = eos.fitting.ContourLandmarks.load(PATH_TO_EOS_SHARE+'ibug_to_sfm.txt')
model_contour = eos.fitting.ModelContour.load(PATH_TO_EOS_SHARE+'model_contours.json')




def read_pts(filename):
    """A helper function to read ibug .pts landmarks from a file."""
    lines = open(filename).read().splitlines()
    lines = lines[3:71]

    landmarks = []
    for l in lines:
        coords = l.split()
        landmarks.append([float(coords[0]), float(coords[1])])

    return landmarks


def overlay(image, rendering):
	image_conf = (255-rendering[:,:,3])/255
	rendering_conf = rendering[:,:,3]/255
	rendering_on_image = image[:,:,:]* image_conf[:,:,None]+rendering[:,:,0:3]*rendering_conf[:,:,None]

	return rendering_on_image.astype(dtype="uint8")


def show_pose_normalization(image, isomap, number_of_steps, mesh, pose):

	image_height, image_width = image.shape[:2]

	# rendering matrices starting point
	model_view_matrix_image = np.ascontiguousarray(pose.get_modelview())
	projection_matrix_image = np.ascontiguousarray(pose.get_projection())


	# rendering matrices final 
	model_view_matrix_final = np.identity(4)
	model_view_matrix_final[0:2,3] = model_view_matrix_image[0:2,3]
	#projection_matrix_final = projection_matrix_image #transformations.projection_matrix([0, 0, 0], [0, 0, 1])


	model_view_matrix_step = (model_view_matrix_final - model_view_matrix_image)/number_of_steps

	for step in range(number_of_steps):

		rendering, depth = eos.render.render(mesh, model_view_matrix_image+step*model_view_matrix_step, projection_matrix_image, image_width, image_height, isomap, True, False, False)


		rendering_on_image = overlay(image, rendering)

		cv2.imshow('rendering on image', rendering_on_image)
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
		cv2.waitKey(50)



def main():
	landmarks = read_pts(PATH_TO_LM_AND_IMG+'image_0010.pts')
	image     = cv2.imread(PATH_TO_LM_AND_IMG+'image_0010.png')

	landmark_ids = list(map(str, range(1, 69))) # generates the numbers 1 to 68, as strings

	image_height, image_width = image.shape[:2]

	(mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(model, blendshapes,
	        landmarks, landmark_ids, landmark_mapper,
	        image_width, image_height, edge_topology, contour_landmarks, model_contour)

	isomap = eos.render.extract_texture(mesh, pose, image)

	rendering, depth = eos.render.render(mesh, np.ascontiguousarray(pose.get_modelview()), np.ascontiguousarray(pose.get_projection()), image_width, image_height, isomap, True, False, False)

	rendering_on_image = overlay(image, rendering)
	cv2.imshow('rendering on image', rendering_on_image)
	cv2.waitKey(4000)

	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image_gray = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

	rendering_on_image = overlay(image_gray, rendering)
	cv2.imshow('rendering on image', rendering_on_image)
	cv2.waitKey(4000)


	show_pose_normalization(image_gray, isomap, 100, mesh, pose)
	
	#blendshapes angry, surprised, smile, laghing, sad, shocked
	strength=1.0
	expressions = [ blendshape_coeffs,
				 np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
				 np.array([strength, 0.0, 0.0, 0.0, 0.0, 0.0]),
				 #np.array([0.0, strength, 0.0, 0.0, 0.0, 0.0]),
				 np.array([0.0, 0.0, strength, 0.0, 0.0, 0.0]),
				 np.array([0.0, 0.0, 0.0, strength, 0.0, 0.0]),
				 np.array([0.0, 0.0, 0.0, 0.0, strength, 0.0]),
				 np.array([0.0, 0.0, 0.0, 0.0, 0.0, strength])
				]
	for exp in range(len(expressions)-1):
		show_different_blendshapes(image_gray, isomap, 10, pose, shape_coeffs, expressions[exp], expressions[exp+1])

	cv2.destroyAllWindows()		


if __name__ == "__main__":
    main()
