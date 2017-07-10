#!/usr/bin/env python3.5

import sys, os, glob

landmark_files = glob.glob('/vol/vssp/datasets/still/CASIA-WebFace/face_pts/TCDCNv0.1/*')
landmark_files.sort()

output_base = '/user/HS204/m09113/my_project_folder/CASIA_webface/landmarks/'

for landmark_file in landmark_files:
	print ('working on',landmark_file)
	with open(landmark_file, 'r') as lf:
		for line_idx, line in enumerate(lf):
			if line_idx%2==0: # the file names
				id_ = line[14:21]
				img = line[-8:-1]
				landmarks = None
			else:
				landmarks = [float(x) for x in line.split()]
				#print (id_)
				#print (img)
				#print (landmarks)

				# now write lms
				if not os.path.exists(output_base+id_):
					os.mkdir(output_base+id_)
				with open(output_base+id_+'/'+img[:3]+'.pts', 'w') as lf:
					lf.write('version: 1\n')
					lf.write('n_points: 68\n')
					lf.write('{\n')
					for lm_idx in range(0, len(landmarks), 2):
						lf.write(str(landmarks[lm_idx])+" "+str(landmarks[lm_idx+1])+"\n")
					lf.write('}\n')

