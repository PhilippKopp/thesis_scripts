#!/usr/bin/env python3.5
import sys
import numpy as np
import obj_analysis_lib as oal
from datetime import datetime
#from concurrent.futures import ThreadPoolExecutor



IDS = ['/02/', '/08/', '/11/', '/13/', '/16/']
EXPRESSIONS = ['/neutral/', '/happy/', '/surprised/' ]
DB_GT_BASE='/user/HS204/m09113/facer2vm_project_area/data/KF-ITW-prerelease'
DB_FITS_BASE='/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/'
EXPERIMENT= 'multi_iter50_reg30/'



print (EXPERIMENT)

#with ThreadPoolExecutor(max_workers=1) as executor: #menpo icp stops working with more than 1 worker??????
for ID in IDS:
	for EXPRESSION in EXPRESSIONS:
		

		fit_obj_model = DB_FITS_BASE+ID+EXPRESSION+EXPERIMENT+'merged.obj'

		gt_imp_vertices = oal.get_KF_ITW_vertex_ids(ID,EXPRESSION)
		if (gt_imp_vertices == None):
			continue
		
		gt_obj_model = DB_GT_BASE+ID+EXPRESSION+'mesh.obj'
		
		# output: aligned gt model
		registered_gt_obj_model = DB_FITS_BASE+ID+EXPRESSION+EXPERIMENT+'registered_nr-icp_gt.obj'
		aligned_gt_obj_model = DB_FITS_BASE+ID+EXPRESSION+EXPERIMENT+'aligned2mask_registered_gt.obj'
		
		# analysis logs
		analysis_log = DB_FITS_BASE+ID+EXPRESSION+EXPERIMENT+'analysis.log'
		distances_log = DB_FITS_BASE+ID+EXPRESSION+EXPERIMENT+'distances_v3.log'
		distances_obj = DB_FITS_BASE+ID+EXPRESSION+EXPERIMENT+'distances_v3.obj'

		print ("aligning, registering and measuring distance of fit of ID ", ID, " and Expression ", EXPRESSION)
		
		#### Do the real work
		oal.register_and_align_KF_ITW_to_surrey(fit_obj_model, gt_imp_vertices, gt_obj_model, registered_gt_obj_model, aligned_gt_obj_model, use_vertices=oal.get_lsfm_crop_mask_surrey_3448_vertices())
		
		distances = oal.measure_distances_on_surface_non_registered_pymesh( source_obj_file=aligned_gt_obj_model, destination_obj_file=fit_obj_model, measure_on_source_vertices=oal.get_lsfm_crop_mask_surrey_3448_vertices())
		oal.write_colored_mesh(aligned_gt_obj_model, mask=oal.get_lsfm_crop_mask_surrey_3448_vertices(), outputfile=distances_obj, color_values=distances)

		# get eye positions
		eye_positions = oal.get_vertex_positions(fit_obj_model, oal.surrey_outer_eye_vertices)
		inter_ocular_distance = oal.calc_distance(eye_positions[0,:], eye_positions[1,:])

		#executor.submit(analyse,fit_obj_model, gt_imp_vertices, gt_obj_model, registered_gt_obj_model, aligned_gt_obj_model, analysis_log, distances_log)
		with open(distances_log, "w") as dist_log:
			dist_log.write("Distances in order of vertex ids:\n")
			for dist in distances:
				dist_log.write(str(dist)+" ")
			dist_log.write("\n")
			dist_log.write("Inter eye distance (outer eye corners): \n")
			dist_log.write(str(inter_ocular_distance))

		with open(analysis_log, "w") as analysis:
			analysis.write(str(datetime.now())+"\n")
			#analysis.write("Log file of analysing KF-ITW fitting on ID "+ID+" with expression "+EXPRESSION+"\n")
			analysis.write("fitted obj model: "+fit_obj_model+"\n")
			analysis.write("vertices for alignment of fitted model: "+str(oal.surrey_imp_vertices)+"\n")
			analysis.write("ground truth model: "+gt_obj_model+"\n")
			analysis.write("vertices for alignment of gt model: "+str(gt_imp_vertices)+"\n")
			analysis.write("registered ground truth model written here: "+registered_gt_obj_model+"\n")
			analysis.write("aligned ground truth model written here: "+aligned_gt_obj_model+"\n")
		
			analysis.write("distances written here: "+distances_log+"\n")
			analysis.write("exterior eye corners used! \n")
			analysis.write("mask cropped lsfm model used! \n")

