import sys, glob, os
import subprocess, shlex
import datetime
from threading import Timer



# important paths
MODEL   = "/user/HS204/m09113/eos/install/share/sfm_shape_3448.bin"
MAPPING = "/user/HS204/m09113/eos/install/share/ibug_to_sfm.txt"
CONTOUR = "/user/HS204/m09113/eos/install/share/model_contours.json"
EDGETOP = "/user/HS204/m09113/eos/install/share/sfm_3448_edge_topology.json"
BLENDSH = "/user/HS204/m09113/eos/install/share/expression_blendshapes_3448.bin"


class EslException(Exception):
	pass


def assemble_command( exe, lms, imgs, out, regularisation=None, iterations=None):
	imgs_param = "-i "
	lms_param  = "-l "

	if isinstance(lms, str):
		imgs_param += imgs + " "
		lms_param  += lms + " "
	else:	
		if (len(lms) != len(imgs)):
			raise EslException('Not equal number of lm and imgs given for command!')
	
		for i in range(len(lms)):
			imgs_param = imgs_param + imgs[i] + " "
			lms_param  = lms_param  + lms[i]  + " "
			
	cmd = exe + " -m " + MODEL + " -p " + MAPPING + " -c " + CONTOUR + " -e " + EDGETOP + " -b " + BLENDSH
	cmd += " " + imgs_param + lms_param + "-o " + out
	if regularisation:
		cmd += " -r " + str(regularisation)
	if iterations:
		cmd += " -t " + str(iterations)

	return cmd

def find_imgs_to_lms (lms, extensions):
	imgs = []
	if type(extensions) is not list:
		extensions = [extensions]
	for lm in lms:
			img =[]
			for extension in extensions:
				img.extend(glob.glob(os.path.splitext(lm)[0]+extension))
			if ( len(img)!=1):
				raise EslException('Not equal number of lm and imgs found!')
			imgs.append(img[0])

	return imgs

def run(cmd, timeout_sec):

	if (sys.version_info >= (3,5)):
		try:
			completed = subprocess.run(shlex.split(cmd), timeout=timeout_sec, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			return completed.stdout.decode('utf-8'), completed.stderr.decode('utf-8')
		except subprocess.TimeoutExpired:
			message = 'Fitting got killed by timeout after '+str(timeout_sec)+' sec!'
			print (message)
			raise EslException(message)
	else:
		proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		kill_proc = lambda p: p.kill()
		timer = Timer(timeout_sec, kill_proc, [proc])
	
		try:
			timer.start()
			stdout,stderr = proc.communicate()
		finally:
			timer.cancel()
			if (proc.poll() == -9 ):
				raise EslException('Fitting probably got killed by timeout!')
			#if (proc.poll() != 0 ):
			#	raise EslException('Fitting crashed! returned '+str(proc.poll()))
			return stdout, stderr


def start_and_log(message, cmd, timeout_sec, log):
	try:
		print (message)

		with open(log, "w") as logfile:
			logfile.write(cmd+"\n \n")
			logfile.write(str(datetime.datetime.now())+"\n \n")
			stdout, stderr = run(cmd, timeout_sec) # 60sec/min * 60 min = 3600 sec 
			logfile.write(stdout + "\n \n")
			logfile.write(stderr + "\n \n")
			logfile.write(str(datetime.datetime.now()))
	except EslException as e:
		with open(log, "a") as logfile:
			logfile.write("ERROR on " + message + ": " + str(e) + "\n \n")
			
	except Exception as e:
		print("ERROR on " + message + ": " + str(e))
		





	
	


