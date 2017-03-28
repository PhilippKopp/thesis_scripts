#!/usr/bin/env python3.5
import obj_analysis_lib as oal

FOLDER = '/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/16/neutral/multi_iter400_reg30'
FITTING = '/merged.obj'
GT = '/aligned_registered_gt.obj'

OUT = '/error_merged.obj'

oal.write_error_mesh_registered(FOLDER+FITTING, FOLDER+GT, FOLDER+OUT)