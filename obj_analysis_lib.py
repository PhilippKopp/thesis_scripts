import sys
import numpy as np
import math



########## Variables 
# indices of vertices of certain points (seen from face):
# nosetip, right human eye outer, right human eye inner, left human eye inner, left human eye outer, nose middle down, mouth top, mouth right, mouth bottom, mouth left

# surrey registered model
surrey_eye_vertices = [171, 604] #right eye centre, left eye centre
surrey_outer_eye_vertices = [177, 610] # right eye outer, left eye outer
surrey_imp_vertices = [114, 177, 181, 614, 617, 270, 424, 398, 401, 812]

	


def get_KF_ITW_vertex_ids(ID, EXPRESSION):
	vertices = None
	# Imperial KF-ITW Ground Truth
	if (ID =='/02/'):
		if (EXPRESSION=='/neutral/'):
			vertices = [236459, 175007, 176204, 178225, 178720, 253473, 287583, 298637, 303777, 290451]  # KF-ITW GT 02 neutral
		elif (EXPRESSION=='/happy/'):
			vertices = [227198, 164924, 170017, 175546, 197434, 250220, 291404, 294816, 291809, 295019]  # KF-ITW GT 02 happy
		elif (EXPRESSION=='/surprised/'):
			vertices = [225788, 167612, 168465, 172740, 174774, 247124, 291409, 305141, 329892, 311767]  # KF-ITW GT 02 surprised
	elif (ID =='/08/'):
		if (EXPRESSION=='/neutral/'):
			vertices = [178641, 146718, 144118, 144962, 144724, 220304, 257470, 259248, 257325, 262390]  # KF-ITW GT 08 neutral
		elif (EXPRESSION=='/surprised/'):
			vertices = [181593, 146029, 145197, 150569, 172910, 226319, 260380, 277372, 319772, 277468]  # KF-ITW GT 08 surprised
	elif (ID =='/11/'):
		if (EXPRESSION=='/neutral/'):
			vertices = [207882, 155735, 155091, 158820, 163516, 230091, 267013, 262134, 267173, 269503]  # KF-ITW GT 11 neutral
		elif (EXPRESSION=='/happy/'):
			vertices = [185150, 152127, 129621, 127992, 147955, 204231, 223863, 246432, 247629, 244016]  # KF-ITW GT 11 happy
		elif (EXPRESSION=='/surprised/'):
			vertices = [177168, 145275, 120284, 124374, 118427, 221424, 238129, 261843, 279287, 261413]  # KF-ITW GT 11 surprised
	elif (ID =='/13/'):
		if (EXPRESSION=='/neutral/'):
			vertices = [174709, 142579, 142040, 128189, 128981, 196463, 238786, 249384, 239378, 240807]  # KF-ITW GT 13 neutral
		elif (EXPRESSION=='/happy/'):
			vertices = [194111, 138123, 139067, 139235, 143847, 211242, 228953, 247955, 243910, 251262]  # KF-ITW GT 13 happy
		elif (EXPRESSION=='/surprised/'):
			vertices = [188179, 136744, 136328, 135283, 137054, 206912, 224366, 259348, 285635, 260112]  # KF-ITW GT 13 surprised
	elif (ID =='/16/'):
		if (EXPRESSION=='/neutral/'):
			vertices = [176754, 139940, 143421, 146310, 147505, 195249, 225684, 226766, 225761, 229664]  # KF-ITW GT 16 neutral
		elif (EXPRESSION=='/happy/'):
			vertices = [220826, 184567, 185797, 171628, 191477, 239258, 254915, 270174, 267829, 272240]  # KF-ITW GT 16 happy

	return vertices


class OalException(Exception):
	pass

def get_vertex_positions(obj_file, imp_vertices):
	"""
	opens a obj file and searches for the imp_vertices given as indices
	returns a numpy matrix with the coordinates of all the imp_vertices
	"""
	all_coordinates = np.empty((len(imp_vertices),3), dtype=float)
	with open(obj_file, "r") as obj:

		for coor_index, imp_vertex in enumerate(imp_vertices):
			obj.seek(0)
			# header line first
			header = obj.readline()
			if header.startswith('v '): #if no header jump back to beginning
				obj.seek(0)

			# then vertices
			for index, line in enumerate(obj):
				if (index == imp_vertex):
					#print "line "+str(index)+" "+line
					coordinates = [float(i) for i in line.split()[1:]]
					all_coordinates[coor_index, :] = coordinates
	return all_coordinates						

def write_aligned_obj (input_obj, tranformation_params, output_obj):
	"""
	Takes an input obj file and transformation params as dictionary like tform = {'rotation':T, 'scale':b, 'translation':c}
	Then writes to the outputfile with the new aligned obj
	"""
		
	T = tranformation_params['rotation']
	b = tranformation_params['scale']
	c = tranformation_params['translation']

	with open(input_obj, "r") as imperial_obj:
		with open(output_obj, "w") as surrey_obj:
			for line in imperial_obj:
		
				if (line.startswith('v ')):
					coordinates = [float(i) for i in line.split()[1:]]
					coordinates = np.array(coordinates)
					new_coordinates = np.dot(b,np.dot(coordinates,T)) + c
					line_out = 'v'
					for i in new_coordinates:
						line_out= line_out + ' ' + str(i)
					line_out+='\n'
				else:
					line_out = line
				surrey_obj.write(line_out)


def read_mesh(obj_file):
	""" small helper function that loads a obj and returns a mesh as list of coordinates"""
	mesh =[]
	with open(obj_file, "r") as obj:

		# header line first
		header = obj.readline()
		if header.startswith('v '): #if no header jump back to beginning
			obj.seek(0)

		for line in obj:
			if (line.startswith('v ')):
				coordinates = [float(i) for i in line.split()[1:]]
				mesh.append(coordinates)
			else:
				break
	return mesh

def write_mesh(mesh, traingle_list_from_file, output_obj):
	with open(output_obj, "w") as out:

		# first write vertex positions from mesh
		for vertex in mesh:
			line_out = 'v'
			for coordinate in vertex:
				line_out= line_out + ' ' + str(coordinate)
			line_out+='\n'
			out.write(line_out)

		# then write triangel list from other reference file
		with open(traingle_list_from_file, "r") as reference:
			for line in reference:
				if (line.startswith('vt ') or line.startswith('f ')):
					out.write(line)


def calc_distance (a, b):
	if (not len(a)==len(b)):
		raise OalException('Can\'t calculate distance between points with different dimensions!')
	a_np = np.array(a)
	b_np = np.array(b)
	return np.linalg.norm(a_np-b_np, ord=2)
	# return np.spatial.distance.euclidean(a,b)


def measure_distances_non_registered(fitted_obj_file, aligned_gt_obj_file, measure_on_fitted_vertices=[]):
	"""
	takes a fitted obj file and an aligned gt obj file, between them the distance gets measured
	at the vertices specified in measure_on_fitted_vertices
	returns a list of distances and the vertices in the gt obj that have shortest distance
	"""
	distances =[]
	corresponding_vertices_gt =[]
	fitted_mesh = read_mesh(fitted_obj_file)
	gt_mesh = read_mesh(aligned_gt_obj_file)

	for index_fitted in range(len(fitted_mesh)):
		# if index in list of given vertices or if list empty measure all distances
		if (index_fitted in measure_on_fitted_vertices or not measure_on_fitted_vertices):

			shortest_distance = 100000000
			index_shortest = -1
			# go through entire gt mesh and find vertex with smallest distance
			for index_gt in range(len(gt_mesh)):
				distance = calc_distance(fitted_mesh[index_fitted], gt_mesh[index_gt])
				if distance< shortest_distance:
					shortest_distance = distance
					index_shortest = index_gt
			corresponding_vertices_gt.append(index_shortest)
			distances.append(shortest_distance)
			#print "for vertex "+str(index_fitted)+ " (fitted) the nearest index in gt is "+str(index_shortest)+" with a distance of "+str(shortest_distance)
	return distances, corresponding_vertices_gt

def measure_distances_registered(fitted_obj_file, aligned_gt_obj_file):
	"""
	takes two registered obj models
	returns a list of distances
	"""
	fitted_mesh = read_mesh(fitted_obj_file)
	gt_mesh = read_mesh(aligned_gt_obj_file)
	fitted_mesh_np = np.array(fitted_mesh)
	gt_mesh_np = np.array(gt_mesh)
	return np.linalg.norm(fitted_mesh_np-gt_mesh_np, ord=2, axis=1)

def pseudocolor(val, minval, maxval):
	# from here: http://stackoverflow.com/questions/10901085/range-values-to-pseudocolor
	import colorsys

	# convert val in range minval..maxval to the range 0..120 degrees which
	# correspond to the colors red..green in the HSV colorspace
	h = (float(val-minval) / (maxval-minval)) * 120
	h=120-h
	# convert hsv color (h,1,1) to its rgb equivalent
	# note: the hsv_to_rgb() function expects h to be in the range 0..1 not 0..360
	r, g, b = colorsys.hsv_to_rgb(h/360, 1., 1.)
	return r, g, b

def write_error_mesh_registered(fitted_obj_file, aligned_gt_obj_file, error_mesh):
	"""
	takes two registered obj models and a path to an output model
	writes mesh with color coding of errors
	"""
	fitted_mesh = read_mesh(fitted_obj_file)
	gt_mesh = read_mesh(aligned_gt_obj_file)
	fitted_mesh_np = np.array(fitted_mesh)
	gt_mesh_np = np.array(gt_mesh)
	distances = np.linalg.norm(fitted_mesh_np-gt_mesh_np, ord=2, axis=1)
	max_error = max(distances)
	colors = [pseudocolor(x, 0, max_error) for x in distances]

	#print (colors)
	with open(error_mesh, "w") as out:

		# first write vertex positions from mesh
		for vertex_id in range(len(fitted_mesh)):
			line_out = 'v'
			for coordinate in fitted_mesh[vertex_id]:
				line_out= line_out + ' ' + str(coordinate)
			for color in colors[vertex_id]:
				line_out= line_out + ' ' + str(color)
			line_out+='\n'
	
			out.write(line_out)
	
		# then write triangel list from other reference file
		with open(aligned_gt_obj_file, "r") as reference:
			for line in reference:
				if (line.startswith('f ')):
					out.write(line)
	



def menpo3d_non_rigid_icp (fitted_obj, gt_obj, fitted_imp_3d_points, gt_imp_3d_points, output_obj):
	import sys
	#sys.path.append("/user/HS204/m09113/scripts/menpo_playground/src/lib/python3.5/site-packages")
	#sys.path.append("/user/HS204/m09113/miniconda2/envs/menpo/lib/python2.7/site-packages/")
	from menpo3d.correspond import non_rigid_icp
	from menpo3d.io.output.base import export_mesh
	import menpo3d.io as m3io
	import menpo


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
	src = m3io.import_mesh(fitted_obj)
	
	# load scan mesh as dest
	dest = m3io.import_mesh(gt_obj)
	#print('destination loaded')
	
	# add landmarks to mesh
	src.landmarks['myLM'] = menpo.shape.PointCloud(fitted_imp_3d_points)
	dest.landmarks['myLM'] = menpo.shape.PointCloud(gt_imp_3d_points)
	#print('landmarks loaded')
	
	# non rigid icp pointcloud as result
	#marc org
	result = non_rigid_icp(src, dest, eps=1e-3, landmark_group='myLM', stiffness_weights=stiff_weights, data_weights=None,
					   landmark_weights=lm_weights, generate_instances=False, verbose=False)
	
	# export the result mesh
	export_mesh(result, output_obj, extension='.obj', overwrite=True)
	


# might be interesting:
#https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.spatial.procrustes.html

#this code stolen from here: http://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy
def procrustes(X, Y, scaling=True, reflection='best'):
	"""
	A port of MATLAB's `procrustes` function to Numpy.

	Procrustes analysis determines a linear transformation (translation,
	reflection, orthogonal rotation and scaling) of the points in Y to best
	conform them to the points in matrix X, using the sum of squared errors
	as the goodness of fit criterion.

		d, Z, [tform] = procrustes(X, Y)

	c - Translation component
	T - Orthogonal rotation and reflection component
	b - Scale component

	Z = b*Y*T + c;

	Inputs:
	------------
	X, Y    
		matrices of target and input coordinates. they must have equal
		numbers of  points (rows), but Y may have fewer dimensions
		(columns) than X.

	scaling 
		if False, the scaling component of the transformation is forced
		to 1

	reflection
		if 'best' (default), the transformation solution may or may not
		include a reflection component, depending on which fits the data
		best. setting reflection to True or False forces a solution with
		reflection or no reflection respectively.

	Outputs
	------------
	d       
		the residual sum of squared errors, normalized according to a
		measure of the scale of X, ((X - X.mean(0))**2).sum()

	Z
		the matrix of transformed Y-values

	tform   
		a dict specifying the rotation, translation and scaling that
		maps X --> Y

	"""

	n,m = X.shape
	ny,my = Y.shape

	muX = X.mean(0)
	muY = Y.mean(0)

	X0 = X - muX
	Y0 = Y - muY

	ssX = (X0**2.).sum()
	ssY = (Y0**2.).sum()

	# centred Frobenius norm
	normX = np.sqrt(ssX)
	normY = np.sqrt(ssY)

	# scale to equal (unit) norm
	X0 /= normX
	Y0 /= normY

	if my < m:
		Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

	# optimum rotation matrix of Y
	A = np.dot(X0.T, Y0)
	U,s,Vt = np.linalg.svd(A,full_matrices=False)
	V = Vt.T
	T = np.dot(V, U.T)

	if reflection is not 'best':

		# does the current solution use a reflection?
		have_reflection = np.linalg.det(T) < 0

		# if that's not what was specified, force another reflection
		if reflection != have_reflection:
			V[:,-1] *= -1
			s[-1] *= -1
			T = np.dot(V, U.T)

	traceTA = s.sum()

	if scaling:

		# optimum scaling of Y
		b = traceTA * normX / normY

		# standarised distance between X and b*Y*T + c
		d = 1 - traceTA**2

		# transformed coords
		Z = normX*traceTA*np.dot(Y0, T) + muX

	else:
		b = 1
		d = 1 + ssY/ssX - 2 * traceTA * normY / normX
		Z = normY*np.dot(Y0, T) + muX

	# transformation matrix
	if my < m:
		T = T[:my,:]
	c = muX - b*np.dot(muY, T)

	#transformation values 
	tform = {'rotation':T, 'scale':b, 'translation':c}

	return d, Z, tform


