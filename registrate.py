import numpy as np
from menpo3d.correspond import non_rigid_icp
from menpo3d.io.output.base import export_mesh
import menpo3d.io as m3io
import menpo
import time
import numpy as np
import argparse
import ntpath
import os

parser = argparse.ArgumentParser()
parser.add_argument('--srcMesh', dest='srcMesh')
parser.add_argument('--srcLm', dest='srcLm')
parser.add_argument('--destMesh', dest='destMesh')
parser.add_argument('--destLm', dest='destLm')

src_path = '/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/02/surprised/merged.obj'
src_lm_path = '/user/HS204/m09113/scripts/test_3d_lms.lnd'

dest_path = './ScansLM/S178MDSU0401.wrl'
dest_lm_path = './ScansLM/new_S178MDSU0401.lnd'

# try something
# lm_weights = [5, 2, .5, 0, 0, 0, 0, 0]  # default weights
# lm_weights = [10, 8, 5, 3, 2, 0.5, 0, 0]
lm_weights = [25, 20, 15, 10, 8, 5, 3, 1]
# lm_weights = [2, 1, 0, 0, 0, 0, 0, 0]
# lm_weights = [25, 20, 15, 10, 5, 2, 1, 0]
# lm_weights = [100, 0, 0, 0, 0, 0, 0, 0]
# lm_weights = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]

stiff_weights = [50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2]  # default weights
# stiff_weights = [50, 20, 15, 10, 3, 1, 0.35, 0.2]
# stiff_weights = [50, 40, 30, 20, 10, 8, 5, 2]
# stiff_weights = [50, 20, 10, 5, 2, 1, 0.5, 0.2]
# stiff_weights = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]

# load pointcloud surrey model as src
src = m3io.import_mesh(src_path)
print('source loaded')


# load scan mesh as dest
#dest = m3io.import_mesh(dest_path)
#print('destination loaded')

# load landmark pointcloud as lm
src_lm_file = open(src_lm_path, 'r')
#dest_lm_file = open(dest_lm_path, 'r')
src_lm = np.loadtxt(src_lm_file)
#dest_lm = np.loadtxt(dest_lm_file)

print (src_lm)

# print(str(src_lm))
# print(str(dest_lm))

# add landmarks to mesh
src.landmarks['myLM'] = menpo.shape.PointCloud(src_lm)
dest.landmarks['myLM'] = menpo.shape.PointCloud(dest_lm)
print('landmarks loaded')

# non rigid icp pointcloud as result
result = non_rigid_icp(src, dest, eps=1e-3, landmark_group='myLM', stiffness_weights=stiff_weights, data_weights=None,
                       landmark_weights=lm_weights, generate_instances=False, verbose=True)

# export the result mesh
outputName = './Results/' + os.path.splitext(ntpath.basename(dest_path))[0]

# add landmark to name
outputName = outputName + '_lm'
for x in lm_weights:
    outputName = outputName + '_' + str(int(x*100))

outputName = outputName + '_stiff'
for x in stiff_weights:
    outputName = outputName + '_' + str(int(x*100))
outputName = outputName + '.obj'
export_mesh(result, outputName, extension='.obj', overwrite=True)

'''def data_weights():
    w_max_iter = 0.5
    w_min_iter = 0.0
    r_width = 0.5 * 0.84716526594210229
    r_mid = 0.95 * 0.84716526594210229
    y_pen = 1.7
    template = load_template()
    return generate_data_weights_per_iter(template,
                                          template.landmarks['nosetip'].lms,
                                          r_width=r_width,
                                          r_mid=r_mid,
                                          w_min_iter=w_min_iter,
                                          w_max_iter=w_max_iter,
                                          y_pen=y_pen
                                          )'''