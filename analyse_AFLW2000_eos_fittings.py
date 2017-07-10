#!/usr/bin/env python3.5
import sys, os, glob
import numpy as np
import obj_analysis_lib as oal
#import random
from datetime import datetime


DB_GT_BASE='/user/HS204/m09113/facer2vm_project_area/data/AFLW2000-3D/converted2obj/'
#DB_GT_BASE='/user/HS204/m09113/facer2vm_project_area/people/Philipp/AFLW2000_fittings/gt_registered_with_fitting_iter10_reg30/'
DB_FITS_BASE='/user/HS204/m09113/my_project_folder/AFLW2000_fittings/eos_single_iter200_reg30/'
DB_FITS_EVAL='/user/HS204/m09113/my_project_folder/AFLW2000_fittings/eos_single_iter200_reg30/_eval_with_own_fitting_registration/'
#DB_FITS_BASE='/user/HS204/m09113/my_project_folder/KF-ITW-single_test_08_neutral_mask/'


#EXPERIMENT= 'eos_single_iter400_reg30/'

CLEAR = True

if (not os.path.exists(DB_FITS_EVAL)):
	os.mkdir(DB_FITS_EVAL)

images = glob.glob(DB_FITS_BASE+'/image*')

gt_imp_vertices = oal.basel_imp_vertices

for image_name in images:
	#image=os.path.basename(image_name)[:-4]
	image=os.path.basename(image_name)

	fit_obj_model = DB_FITS_BASE+image+'/merged.obj'
	
	gt_obj_model = DB_GT_BASE+image+'.obj'
	
	# output: aligned gt model
	#registered_gt_obj_model = DB_GT_BASE+image+'.registered2mean845_nr-icp_gt.obj'
	#registered_gt_obj_model = gt_obj_model
	registered_gt_obj_model = DB_FITS_EVAL+image+'.registered_nr-icp_gt.obj'
	aligned_gt_obj_model = DB_FITS_EVAL+image+'.aligned_gt.obj'
	#aligned_gt_obj_model = '/user/HS204/m09113/Desktop/dummy.obj'
	
	#if (os.path.exists(aligned_gt_obj_model) and not CLEAR):
	#	print ("already done image", image)
	#	continue

	# distances logs
	distances_log = DB_FITS_EVAL+image+'.distances_v3.log'
	distances_obj = DB_FITS_EVAL+image+'.distances_v3.obj'

	print ("analysing single frame fit of image", image)
	
	#### Do the real work	
	#oal.register_and_align_KF_ITW_to_surrey(fit_obj_model, gt_imp_vertices, gt_obj_model, registered_gt_obj_model, aligned_gt_obj_model, use_vertices=oal.get_lsfm_crop_mask_surrey_3448_vertices())
	#continue

	### Now align registered model to fitted model
	gt_registered_matrix = oal.get_vertex_positions(registered_gt_obj_model, oal.surrey_imp_vertices)
	eos_matrix = oal.get_vertex_positions(fit_obj_model, oal.surrey_imp_vertices)
		
	d, Z, tform = oal.procrustes(eos_matrix, gt_registered_matrix)
		
	oal.write_aligned_obj(registered_gt_obj_model, tform, aligned_gt_obj_model)
	#continue

	#distances = oal.measure_distances_on_surface_non_registered_pymesh( source_obj_file=aligned_gt_obj_model, destination_obj_file=fit_obj_model, measure_on_source_vertices=oal.get_basel_circle_mask())
	distances = oal.measure_distances_on_surface_non_registered_pymesh( source_obj_file=aligned_gt_obj_model, destination_obj_file=fit_obj_model, measure_on_source_vertices=oal.get_lsfm_crop_mask_surrey_3448_vertices())
	#oal.write_colored_mesh(aligned_gt_obj_model, mask=oal.get_basel_circle_mask(), outputfile=distances_obj, color_values=distances)
	oal.write_colored_mesh(aligned_gt_obj_model, mask=oal.get_lsfm_crop_mask_surrey_3448_vertices(), outputfile=distances_obj, color_values=distances)

	# get eye positions
	eye_positions = oal.get_vertex_positions(fit_obj_model, oal.surrey_outer_eye_vertices)
	inter_ocular_distance = oal.calc_distance(eye_positions[0,:], eye_positions[1,:])

	with open(distances_log, "w") as dist_log:
		dist_log.write("Distances in order of vertex ids:\n")
		for dist in distances:
			dist_log.write(str(dist)+" ")
		dist_log.write("\n")
		dist_log.write("Inter eye distance (outer eye corners): \n")
		dist_log.write(str(inter_ocular_distance))		
		


exit(0)
analysis_log = DB_FITS_EVAL+'analysis.log'
with open(analysis_log, "w") as analysis:
	analysis.write(str(datetime.now())+"\n")
	analysis.write("Log file of analysing AFLW fittings\n")
	analysis.write("fitted obj model: "+fit_obj_model+"\n")
	analysis.write("vertices for alignment of fitted model: "+str(oal.surrey_imp_vertices)+"\n")
	analysis.write("ground truth model: "+gt_obj_model+"\n")
	analysis.write("vertices for alignment of gt model: "+str(gt_imp_vertices)+"\n")
	analysis.write("registered ground truth model written here: "+registered_gt_obj_model+"\n")
	analysis.write("aligned ground truth model written here: "+aligned_gt_obj_model+"\n")

	analysis.write("distances written here: "+distances_log+"\n")
	#analysis.write("corresponding_vertices_gt: "+corresponding_vertices_gt+"\n")
	analysis.write("exterior eye corners used! \n")
	analysis.write("mask cropped lsfm model used! \n")
#except Exception as e:
#	print("ERROR: " + str(e))
