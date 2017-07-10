#!/usr/bin/env python3.5
import obj_analysis_lib as oal


IDS = ['/02/', '/08/', '/11/', '/13/', '/16/']
EXPRESSIONS = ['/neutral/', '/happy/', '/surprised/' ]
DB_GT_BASE='/user/HS204/m09113/facer2vm_project_area/data/KF-ITW-prerelease'
DB_FITS_BASE='/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/'


oal.write_colored_mesh("/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/08/neutral/multi_iter400_reg30/merged.obj", oal.surrey_imp_vertices, "/user/HS204/m09113/my_project_folder/surrey_with_colored_lms.obj")
for ID in IDS:
	for EXPRESSION in EXPRESSIONS:
		gt_imp_vertices = oal.get_KF_ITW_vertex_ids(ID,EXPRESSION)
		if (gt_imp_vertices == None):
			continue
			
		gt_obj_model = DB_GT_BASE+ID+EXPRESSION+'mesh.obj'
		outputfile = DB_FITS_BASE+ID+EXPRESSION+"gt_with_colored_lms.obj"

		oal.write_colored_mesh(gt_obj_model, gt_imp_vertices, outputfile)
		
