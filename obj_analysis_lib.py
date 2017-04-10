import sys
import numpy as np
import math



########## Variables 
# indices of vertices of certain points (seen from face):
# nosetip, right human eye outer, right human eye inner, left human eye inner, left human eye outer, nose middle down, mouth top, mouth right, mouth bottom, mouth left

# surrey registered model
surrey_eye_vertices = [171, 604] #right eye centre, left eye centre
surrey_outer_eye_vertices = [177, 610] # right eye outer, left eye outer
surrey_imp_vertices = [114, 177, 181, 614, 610, 436, 424, 398, 401, 812]
lsfm_crop_mask_surrey_3448_vertices = None


ALL_POINTS = "ALL_POINTS"

def get_lsfm_crop_mask_surrey_3448_vertices(mask_txt_file="/user/HS204/m09113/Desktop/mask_3448/vertices.txt"):
	global lsfm_crop_mask_surrey_3448_vertices
	if not lsfm_crop_mask_surrey_3448_vertices:
		lsfm_crop_mask_surrey_3448_vertices = np.loadtxt(mask_txt_file).astype(int).tolist()
	return lsfm_crop_mask_surrey_3448_vertices

	


def get_KF_ITW_vertex_ids(ID, EXPRESSION):
	vertices = None
	# Imperial KF-ITW Ground Truth
	if (ID =='/02/'):
		if (EXPRESSION=='/neutral/'):
			vertices = [236459, 196875, 176204, 178225, 185449, 253473, 287583, 304858, 303777, 290451]  # KF-ITW GT 02 neutral
		elif (EXPRESSION=='/happy/'):
			vertices = [227198, 185198, 170017, 175546, 197434, 250220, 291404, 294816, 291809, 295019]  # KF-ITW GT 02 happy
		elif (EXPRESSION=='/surprised/'):
			vertices = [225788, 186609, 168465, 172740, 174774, 247124, 291409, 305141, 329892, 311767]  # KF-ITW GT 02 surprised
	elif (ID =='/08/'):
		if (EXPRESSION=='/neutral/'):
			vertices = [178641, 146718, 144118, 144962, 144724, 220304, 257470, 259248, 257325, 262390]  # KF-ITW GT 08 neutral
		elif (EXPRESSION=='/surprised/'):
			vertices = [181593, 166633, 145197, 150569, 172910, 226319, 260380, 277372, 319772, 277468]  # KF-ITW GT 08 surprised
	elif (ID =='/11/'):
		if (EXPRESSION=='/neutral/'):
			vertices = [207882, 155230, 155091, 158820, 163870, 230091, 261737, 262134, 267173, 269503]  # KF-ITW GT 11 neutral
		elif (EXPRESSION=='/happy/'):
			vertices = [185150, 152127, 129065, 128342, 147955, 223429, 223863, 246432, 247629, 244016]  # KF-ITW GT 11 happy
		elif (EXPRESSION=='/surprised/'):
			vertices = [177168, 145275, 120284, 124374, 118427, 221424, 238129, 261843, 279287, 261413]  # KF-ITW GT 11 surprised
	elif (ID =='/13/'):
		if (EXPRESSION=='/neutral/'):
			vertices = [174709, 141922, 142040, 128189, 129463, 196463, 238789, 249384, 239378, 240807]  # KF-ITW GT 13 neutral
		elif (EXPRESSION=='/happy/'):
			vertices = [194111, 138123, 139067, 139235, 143847, 211242, 229246, 247955, 243910, 251262]  # KF-ITW GT 13 happy
		elif (EXPRESSION=='/surprised/'):
			vertices = [188179, 136744, 136328, 135283, 137054, 206912, 224366, 259348, 285635, 260112]  # KF-ITW GT 13 surprised
	elif (ID =='/16/'):
		if (EXPRESSION=='/neutral/'):
			vertices = [176754, 143802, 143996, 146310, 150820, 195249, 225684, 226766, 225761, 229664]  # KF-ITW GT 16 neutral
		elif (EXPRESSION=='/happy/'):
			vertices = [220826, 185956, 185797, 171628, 173692, 239258, 254915, 270174, 267829, 272240]  # KF-ITW GT 16 happy

	return vertices


class OalException(Exception):
	pass

def get_vertex_positions(obj_file, imp_vertices):
	"""
	opens a obj file and searches for the imp_vertices given as indices
	returns a numpy matrix with the coordinates of all the imp_vertices
	"""
	
	mesh = read_mesh(obj_file)

	if imp_vertices == ALL_POINTS:
		imp_vertices = [x for x in range(len(mesh))]
	imp_coordinates = np.empty((len(imp_vertices),3), dtype=float)

	for coor_index, imp_vertex in enumerate(imp_vertices):
		imp_coordinates[coor_index, :] = mesh[imp_vertex]

	return imp_coordinates						

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
				continue
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


def measure_distances_and_correspondendences_non_registered(source_obj_file, destination_obj_file, measure_on_source_vertices=ALL_POINTS):
	"""
	takes a source obj file and a destination obj file, between them the distance gets measured
	at the vertices specified in measure_on_source_vertices
	returns a list of distances and the vertices in the gt obj that have shortest distance
	"""
	distances =[]
	corresponding_vertices_destination =[]
	source_mesh = read_mesh(source_obj_file)
	destination_mesh = read_mesh(destination_obj_file)

	for index_source in range(len(source_mesh)):
		# if index in list of given vertices or if list empty measure all distances
		if (measure_on_source_vertices==ALL_POINTS or index_source in measure_on_source_vertices):

			shortest_distance = 100000000
			index_shortest = -1
			# go through entire gt mesh and find vertex with smallest distance
			for index_destination in range(len(destination_mesh)):
				distance = calc_distance(source_mesh[index_source], destination_mesh[index_destination])
				if distance< shortest_distance:
					shortest_distance = distance
					index_shortest = index_destination
			corresponding_vertices_destination.append(index_shortest)
			distances.append(shortest_distance)
			#print "for vertex "+str(index_source)+ " (fitted) the nearest index in gt is "+str(index_shortest)+" with a distance of "+str(shortest_distance)
	return distances, corresponding_vertices_destination




def measure_distances_registered(fitted_obj_file, aligned_gt_obj_file, mask=ALL_POINTS):
	"""
	takes two registered obj models
	returns a list of distances
	"""
	fitted_mesh = read_mesh(fitted_obj_file)
	gt_mesh = read_mesh(aligned_gt_obj_file)
	fitted_mesh_np = np.array(fitted_mesh)
	gt_mesh_np = np.array(gt_mesh)
	diff = np.linalg.norm(fitted_mesh_np-gt_mesh_np, ord=2, axis=1)
	if mask!=ALL_POINTS:
		diff_mask =[]
		for index in range(len(diff)):
			if index in mask:
				diff_mask.append(diff[index])
		return diff_mask

	else:
		return diff 

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
	
def write_colored_mesh(obj_file, mask, outputfile, color_values=[]):
	"""
	takes an obj input file and a mask. Colors mask differently than rest. If color values are given they have to be same size as the mask.
	write this colored mesh as obj to outputfile
	"""
	mesh = read_mesh(obj_file)
	#color_mask = pseudocolor(0.1, 0, 1)
	#color_rest = pseudocolor(0.9, 0, 1)
	color_rest = 100, 100, 100 #should be gray

	if len(color_values) != 0:
		if len(mask)!= len(color_values):
			raise OalException("size of mask does not have same length as color values given: "+str(len(mask))+"!="+str(len(color_values)))
		max_v = max(color_values)
		min_v = min(color_values)
		colors = [pseudocolor(v,min_v, max_v) for v in color_values]
	else:
		colors = [pseudocolor(0.1, 0, 1) for v in mask]

	#load triangle list, we need that for later
	triangle_list = []
	with open(obj_file, "r") as reference:
			for line in reference:
				if (line.startswith('f ')):
					triangle_list.append(line)


	#print (colors)
	with open(outputfile, "w") as out:

		# first write vertex positions from mesh
		color_id = 0
		for vertex_id in range(len(mesh)):
			line_out = 'v'
			for coordinate in mesh[vertex_id]:
				line_out= line_out + ' ' + str(coordinate)
			if vertex_id in mask:
				for rgb in colors[color_id]:
					line_out= line_out + ' ' + str(rgb)
				color_id += 1
			else:
				for rgb in color_rest:
					line_out= line_out + ' ' + str(rgb)
			line_out+='\n'
	
			out.write(line_out)
		
		# then write triangle list we loaded before
		for triangel in triangle_list:
			out.write(triangel)



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




def register_and_align_KF_ITW_to_surrey(fit_obj_model, gt_imp_vertices, gt_obj_model, registered_gt_obj_model, aligned_gt_obj_model, use_vertices=ALL_POINTS):
	### Register GT model
	gt_matrix = get_vertex_positions(gt_obj_model, gt_imp_vertices)
	surrey_matrix = get_vertex_positions(fit_obj_model, surrey_imp_vertices)
			
	menpo3d_non_rigid_icp(fit_obj_model, gt_obj_model, surrey_matrix, gt_matrix, registered_gt_obj_model)
			
	### Now align registered model to fitted model
	gt_registered_matrix = get_vertex_positions(registered_gt_obj_model, use_vertices)
	surrey_matrix = get_vertex_positions(fit_obj_model, use_vertices)
	
	d, Z, tform = procrustes(surrey_matrix, gt_registered_matrix)
	
	write_aligned_obj(registered_gt_obj_model, tform, aligned_gt_obj_model)

def generate_tree(mesh, x_range, number_splits_x, y_range, number_splits_y):
	"""
	devides a mesh into a "tree". returns an array in the size of number_splits_x * number_splits_y with lists of vertices at these positions
	"""

	x_delta = (x_range[1]-x_range[0])/number_splits_x
	y_delta = (y_range[1]-y_range[0])/number_splits_y

	# tree is a 2d array containing lists of vertex ids
	tree = [[[] for j in range(number_splits_x)] for i in range(number_splits_y)]

	for vertex_id, coordinates in enumerate(mesh):
		if (coordinates[0]<x_range[0] or coordinates[0]>x_range[1]):
			raise OalException('ERROR: x index of vertex out of range: '+str(coordinates[0]))
		if (coordinates[1]<y_range[0] or coordinates[1]>y_range[1]):
			raise OalException('ERROR: y index of vertex out of range: '+str(coordinates[1]))
		x = int((coordinates[0]-x_range[0])/x_delta)
		y = int((coordinates[1]-y_range[0])/y_delta)
		tree[x][y].append(vertex_id)

	return tree


def measure_distances_on_surface_non_registered(source_obj_file, destination_obj_file, measure_on_source_vertices=ALL_POINTS):
	"""
	takes a source obj file and a destination obj file, between them the distance gets measured
	at the vertices specified in measure_on_source_vertices but to the surface of the destination
	returns a list of distances
	"""
	print("careful with this function!! Programmed my own and has bug that it chooses vertices that aren't on the mesh")
	x_range = np.array([-80, 80])
	y_range = np.array([-120, 120])
	number_splits_x = 20
	number_splits_y = 20
	x_delta = (x_range[1]-x_range[0])/number_splits_x
	y_delta = (y_range[1]-y_range[0])/number_splits_y

	tree_range = 1 # +- how many bins around the closest bin are included to the search for the closest vertices

	distances =[]
	source_mesh = np.array(read_mesh(source_obj_file))
	destination_mesh = np.array(read_mesh(destination_obj_file))
	destination_tree = np.array(generate_tree(destination_mesh, x_range, number_splits_x, y_range, number_splits_y))

	for index_source in range(len(source_mesh)):
		# if index in list of given vertices or if list empty measure all distances
		if (measure_on_source_vertices==ALL_POINTS or index_source in measure_on_source_vertices):

			#print ("measuring distance to distination from ", source_mesh[index_source])

			if (source_mesh[index_source][0]<x_range[0] or source_mesh[index_source][0]>x_range[1]):
				raise OalException('ERROR: x index of vertex out of range: '+str(coordinates[0]))
			if (source_mesh[index_source][1]<y_range[0] or source_mesh[index_source][1]>y_range[1]):
				raise OalException('ERROR: y index of vertex out of range: '+str(coordinates[1]))

			# calc "bins" in tree we want to go through to search closest vertices
			x = int((source_mesh[index_source][0]-x_range[0])/x_delta)
			y = int((source_mesh[index_source][1]-y_range[0])/y_delta)

			x_bottom = ( x-tree_range   if x-tree_range   >= 0   			 else 0 )
			x_top    = ( x+tree_range+1 if x+tree_range+1 <= number_splits_x else number_splits_x )
			y_bottom = ( y-tree_range   if y-tree_range   >= 0 				 else 0 )
			y_top    = ( y+tree_range+1 if y+tree_range+1 <= number_splits_y else number_splits_y )

			#print ("x is",x," xBotton",x_bottom,"and x top",x_top)

			# put all vertices of these bins into one list
			destination_vertices_subset=[]
			for x_ in range(x_bottom, x_top):
				for y_ in range(y_bottom, y_top):
					destination_vertices_subset.extend(destination_tree[x_, y_])

			#calculate distances for each vertex
			indices_and_distances = [ [i, calc_distance(source_mesh[index_source], destination_mesh[i]) ] for i in destination_vertices_subset ]
			#print (indices_and_distances)
			indices_closest_triangle = sorted(indices_and_distances, key=lambda index_and_distance: index_and_distance[1])[:3]
			#print ("closest triangle found", indices_closest_triangle)
			indices_closest_triangle = [int(i[0]) for i in indices_closest_triangle]
			#print ("that have the coordinates",destination_mesh[indices_closest_triangle[0]], ", ", destination_mesh[indices_closest_triangle[1]], " and ", destination_mesh[indices_closest_triangle[2]])
			
			# go through this subset of vertices and find vertex with smallest distance
			#indices = sorted((enumerate(destination_vertices_subset)))
			normal_vector = np.cross(destination_mesh[indices_closest_triangle[0]]-destination_mesh[indices_closest_triangle[1]], destination_mesh[indices_closest_triangle[0]]-destination_mesh[indices_closest_triangle[2]])
			normal_vector_norm = np.linalg.norm(normal_vector)
			a = np.dot(destination_mesh[indices_closest_triangle[0]], normal_vector)
			#distance = normal_vector[0]*source_mesh[index_source][0]+normal_vector[1]*source_mesh[index_source][1]+normal_vector[2]*source_mesh[index_source][2]-a
			distance = abs( ( np.dot(normal_vector, source_mesh[index_source])-a )/normal_vector_norm )
			#print ("plane function: normal vector",normal_vector,"and a",a)
			#print ("and calculated distance",distance)

			#print ("\n \n \n \n \n ")

			distances.append(distance)
			#print "for vertex "+str(index_source)+ " (fitted) the nearest index in gt is "+str(index_shortest)+" with a distance of "+str(shortest_distance)
	return distances

def measure_distances_on_surface_non_registered_pymesh(source_obj_file, destination_obj_file, measure_on_source_vertices=ALL_POINTS):
	import pymesh

	#source_mesh = np.array(read_mesh(source_obj_file))
	destination_mesh = pymesh.load_mesh(destination_obj_file);
	#for index_source in range(len(source_mesh)):
		# if index in list of given vertices or if list empty measure all distances
	#	if (measure_on_source_vertices==ALL_POINTS or index_source in measure_on_source_vertices):
	if measure_on_source_vertices==ALL_POINTS:
		measure_on_source_vertices=range(destination_mesh.num_vertices)
	source_points = get_vertex_positions(source_obj_file, measure_on_source_vertices)
	squared_distances, face_indices, closest_points = pymesh.distance_to_mesh(destination_mesh, source_points)
	distances =[math.sqrt(d2) for d2 in squared_distances]
	return distances


def read_fitting_log(fitting_log_file):
	with open(fitting_log_file, "r") as fitting_log:
		cmd = fitting_log.readline()
		
		line=''
		while not line.startswith("2017"):
			line = fitting_log.readline()
		start_time = line

		line=''
		while not line.startswith("final pca shape coefficients:"):
			line = fitting_log.readline()
		alphas = line.split(':')[1]		
		alphas = [float(i) for i in alphas.split()]


		line=''
		while not line.startswith("mean blendshape"):
			line = fitting_log.readline()
		blendshapes = line		

		line=''
		while not line.startswith("2017"):
			line = fitting_log.readline()
		end_time = line

		return alphas