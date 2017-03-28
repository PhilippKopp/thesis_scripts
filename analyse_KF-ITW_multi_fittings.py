#!/usr/bin/env python3.5
import sys
import numpy as np
import obj_analysis_lib as oal
#import random
from datetime import datetime



IDS = ['/02/', '/08/', '/11/', '/13/', '/16/']
EXPRESSIONS = ['/neutral/', '/happy/', '/surprised/' ]
DB_GT_BASE='/user/HS204/m09113/facer2vm_project_area/data/KF-ITW-prerelease'
DB_FITS_BASE='/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/'
EXPERIMENT= 'multi_iter400_reg30/'


for ID in IDS:
	for EXPRESSION in EXPRESSIONS:
		
		try:
			fit_obj_model = DB_FITS_BASE+ID+EXPRESSION+EXPERIMENT+'merged.obj'
	
			gt_imp_vertices = oal.get_KF_ITW_vertex_ids(ID,EXPRESSION)
			if (gt_imp_vertices == None):
				continue
	
			print ("analysing multi frame fit of ID ", ID, " and Expression ", EXPRESSION)
				
			
			gt_obj_model = DB_GT_BASE+ID+EXPRESSION+'mesh.obj'
			
			# output: aligned gt model
			registered_gt_obj_model = DB_FITS_BASE+ID+EXPRESSION+EXPERIMENT+'registered_nr-icp_gt.obj'
			aligned_gt_obj_model = DB_FITS_BASE+ID+EXPRESSION+EXPERIMENT+'aligned_registered_gt.obj'
			
			# analysis logs
			analysis_log = DB_FITS_BASE+ID+EXPRESSION+EXPERIMENT+'analysis.log'
			distances_log = DB_FITS_BASE+ID+EXPRESSION+EXPERIMENT+'distances.log'
					
#			### Register GT model
#			gt_matrix = oal.get_vertex_positions(gt_obj_model, gt_imp_vertices)
#			surrey_matrix = oal.get_vertex_positions(fit_obj_model, oal.surrey_imp_vertices)
			#
#			oal.menpo3d_non_rigid_icp(fit_obj_model, gt_obj_model, surrey_matrix, gt_matrix, registered_gt_obj_model)
			#
#			### Now align registered model to fitted model
#			all_points = [x for x in range(3448)]
#			gt_registered_matrix = oal.get_vertex_positions(registered_gt_obj_model, all_points)
#			surrey_matrix = oal.get_vertex_positions(fit_obj_model, all_points)
			#
#			d, Z, tform = oal.procrustes(surrey_matrix, gt_registered_matrix)
			#
#			oal.write_aligned_obj(registered_gt_obj_model, tform, aligned_gt_obj_model)
			
			# calculate distances
			distances = oal.measure_distances_registered(fit_obj_model, aligned_gt_obj_model)
			
			# normalize distances by inter eye distance
			eye_positions = oal.get_vertex_positions(fit_obj_model, oal.surrey_outer_eye_vertices)
			inter_ocular_distance = oal.calc_distance(eye_positions[0,:], eye_positions[1,:])
			distances = [d / inter_ocular_distance for d in distances]
			
			
			#sort distances ascendingly
			distances.sort()
			
			with open(distances_log, "w") as dist_log:
				for dist in distances:
					dist_log.write(str(dist)+" ")
				dist_log.write("\n")


			with open(analysis_log, "a") as analysis:
				analysis.write("exterior eye corners used! \n")
					
#			with open(analysis_log, "w") as analysis:
#				analysis.write(str(datetime.now())+"\n")
#				analysis.write("Log file of analysing KF-ITW fitting on ID "+ID+" with expression "+EXPRESSION+"\n")
#				analysis.write("fitted obj model: "+fit_obj_model+"\n")
#				analysis.write("vertices for alignment of fitted model: "+str(oal.surrey_imp_vertices)+"\n")
#				analysis.write("ground truth model: "+gt_obj_model+"\n")
#				analysis.write("vertices for alignment of gt model: "+str(gt_imp_vertices)+"\n")
#				analysis.write("registered ground truth model written here: "+registered_gt_obj_model+"\n")
#				analysis.write("aligned ground truth model written here: "+aligned_gt_obj_model+"\n")
			#
#				analysis.write("distances written here: "+distances_log+"\n")
				#analysis.write("corresponding_vertices_gt: "+corresponding_vertices_gt+"\n")
		except Exception as e:
			print("ERROR: " + str(e))
