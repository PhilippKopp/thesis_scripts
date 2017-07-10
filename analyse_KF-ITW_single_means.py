#!/usr/bin/env python3.5
import sys, os, glob
import numpy as np
import obj_analysis_lib as oal
#import random
from datetime import datetime



IDS = ['/02/', '/08/', '/11/', '/13/', '/16/']
EXPRESSIONS = ['/neutral/', '/happy/', '/surprised/' ]
DB_GT_BASE='/user/HS204/m09113/facer2vm_project_area/data/KF-ITW-prerelease'
DB_FITS_BASE='/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/'


EXPERIMENT= 'single_iter400_reg30/'

CLEAR = True


for ID in IDS:
	for EXPRESSION in EXPRESSIONS:

		gt_imp_vertices = oal.get_KF_ITW_vertex_ids(ID,EXPRESSION)
		if (gt_imp_vertices == None):
			continue

		try:
			print ("analysing single mean fit of ID ", ID, " and Expression ", EXPRESSION)
			img_nums = [ os.path.basename(x)[:-11] for x in glob.glob(DB_FITS_BASE+ID+EXPRESSION+EXPERIMENT+"*.isomap.png")]

			all_meshs = []
			for img_num in img_nums:
				
				fit_obj_model = DB_FITS_BASE+ID+EXPRESSION+EXPERIMENT+img_num+'.obj'
				all_meshs.append(oal.read_mesh(fit_obj_model))

			#calculate mean
			mean_mesh_np = np.mean(all_meshs, axis=0)
			oal.write_mesh(mean_mesh_np, traingle_list_from_file=DB_FITS_BASE+ID+EXPRESSION+EXPERIMENT+img_nums[0]+'.obj', output_obj=DB_FITS_BASE+ID+EXPRESSION+EXPERIMENT+'mean.obj')
				
			
			gt_obj_model = DB_GT_BASE+ID+EXPRESSION+'mesh.obj'
			
			# output: aligned gt model
			registered_gt_obj_model = DB_FITS_BASE+ID+EXPRESSION+EXPERIMENT+'mean.registered_nr-icp_gt.obj'
			aligned_gt_obj_model = DB_FITS_BASE+ID+EXPRESSION+EXPERIMENT+'mean.aligned_registered_gt.obj'
			
			if (os.path.exists(aligned_gt_obj_model) and not CLEAR):
				continue

			# analysis logs
			distances_log = DB_FITS_BASE+ID+EXPRESSION+EXPERIMENT+'mean.distances_v3.log'
			distances_obj = DB_FITS_BASE+ID+EXPRESSION+EXPERIMENT+'mean.distances_v3.obj'

			#### Do the real work
			oal.register_and_align_KF_ITW_to_surrey(fit_obj_model, gt_imp_vertices, gt_obj_model, registered_gt_obj_model, aligned_gt_obj_model, use_vertices=oal.get_lsfm_crop_mask_surrey_3448_vertices())
		
			distances = oal.measure_distances_on_surface_non_registered_pymesh( source_obj_file=aligned_gt_obj_model, destination_obj_file=fit_obj_model, measure_on_source_vertices=oal.get_lsfm_crop_mask_surrey_3448_vertices())
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
					

			analysis_log = DB_FITS_BASE+ID+EXPRESSION+EXPERIMENT+'mean.analysis.log'
					
			with open(analysis_log, "w") as analysis:
				analysis.write(str(datetime.now())+"\n")
				analysis.write("Log file of analysing KF-ITW fitting on ID "+ID+" with expression "+EXPRESSION+"\n")
				analysis.write("fitted obj model: "+fit_obj_model+"\n")
				analysis.write("vertices for alignment of fitted model: "+str(oal.surrey_imp_vertices)+"\n")
				analysis.write("ground truth model: "+gt_obj_model+"\n")
				analysis.write("vertices for alignment of gt model: "+str(gt_imp_vertices)+"\n")
				analysis.write("registered ground truth model written here: "+registered_gt_obj_model+"\n")
				analysis.write("aligned ground truth model written here: "+aligned_gt_obj_model+"\n")
			
				analysis.write("distances written here: "+distances_log+"\n")
				analysis.write("exterior eye corners used! \n")
				analysis.write("mask cropped lsfm model used! \n")
		except Exception as e:
			print("ERROR: " + str(e))
