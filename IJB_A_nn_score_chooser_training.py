#!/usr/bin/env python3.5
import sys, os, random
import numpy as np
import IJB_A_template_lib as itl
import obj_analysis_lib as oal
import tensorflow as tf

from collections import deque

#from scipy.spatial import distance

scores_base_HO2D = '/user/HS204/m09113/my_project_folder/IJB_A/fr_verification_experiments/verification_exp_99/'
scores_base_HO3D = '/user/HS204/m09113/my_project_folder/IJB_A/fr_verification_experiments/verification_exp_80/'
scores_base_PH   = '/user/HS204/m09113/my_project_folder/IJB_A/fr_verification_experiments/verification_exp_13/'

FITTING_BASE = '/user/HS204/m09113/my_project_folder/IJB_A/multi_iter75_reg30_256/verification_templates/'

NN_TRAINING_BASE = '/user/HS204/m09113/my_project_folder/IJB_A/nn_score_chooser/03/'
if not os.path.exists(NN_TRAINING_BASE):
	os.mkdir(NN_TRAINING_BASE)


os.environ['CUDA_VISIBLE_DEVICES']=''

BATCH_SIZE=100

split_used_for_training = 1

class Training_example:
	pass


# first create training dataset
comparisons_HO2D, _ = itl.read_matching_output(scores_base_HO2D+'/matching_cos/split'+str(split_used_for_training)+'.matches')
comparisons_HO3D, _ = itl.read_matching_output(scores_base_HO3D+'/matching_cos/split'+str(split_used_for_training)+'.matches')
comparisons_PH,   _ = itl.read_matching_output(scores_base_PH  +'/matching_cos/split'+str(split_used_for_training)+'.matches')

metadata_file_path = '/vol/vssp/datasets/still/IJB_A/11/split'+str(split_used_for_training)+'/verify_metadata_'+str(split_used_for_training)+'.csv'
templates_dict = itl.read_IJBA_templates_definition(metadata_file_path)


# comparisons is a list of lists with 3 elements template_id1, template_id2, score 
# for each comparison we now need the mean pose and std deviations of pose for both templates and which score is the best

if len(comparisons_HO2D)!= len(comparisons_HO3D) or len(comparisons_HO3D) != len(comparisons_PH):
	print ('serious problem here!')
	exit(0)

best=np.array([0,0,0])

training_examples = []
for i in range(len(comparisons_PH)):
#for i in range(2000):
	if i%1000==0:
		print ('loaded', i,'of',len(comparisons_PH))

	try:
		training_example = Training_example()
		fitting_log_template_A = FITTING_BASE+'split'+str(split_used_for_training)+'/'+str(comparisons_PH[i][0])+'/fitting.log'
		#print (fitting_log_template_A)
		poses_mean_A, poses_std_A = oal.read_pose_from_log(fitting_log_template_A)
		#print (poses_mean_A, poses_std_A)
		training_example.poses_mean_A = poses_mean_A
		training_example.poses_std_A  = poses_std_A

		fitting_log_template_B = FITTING_BASE+'split'+str(split_used_for_training)+'/'+str(comparisons_PH[i][1])+'/fitting.log'
		#print (fitting_log_template_B)
		poses_mean_B, poses_std_B = oal.read_pose_from_log(fitting_log_template_B)
		#print (poses_mean_B, poses_std_B)
		training_example.poses_mean_B = poses_mean_B
		training_example.poses_std_B  = poses_std_B

		positive_match = ( templates_dict[comparisons_PH[i][0]].subject_id == templates_dict[comparisons_PH[i][1]].subject_id )
		#print (positive_match)

		nn_output = [0]*3
		scores = [comparisons_HO2D[i][2], comparisons_HO3D[i][2], comparisons_PH[i][2]]
		training_example.scores = scores
		#print (scores)
		if positive_match:
			nn_output[ np.argmax(scores)] = 1
		else:
			nn_output[ np.argmin(scores)] = 1
		#print (nn_output)
		best+=nn_output
		training_example.nn_output = nn_output

		training_examples.append(training_example)
	except oal.OalException:
		pass

print ('proportion of training samples, which score to take',best)
# we now have the training set
# lets construct our nn with tf

input_layer = tf.placeholder(tf.float32, shape=(BATCH_SIZE,15))
gt_output_layer = tf.placeholder(tf.int32, shape=(BATCH_SIZE, 3))
fc1 = tf.layers.dense(input_layer, units=90 , activation=tf.nn.relu, name='fc1')
fc2 = tf.layers.dense(fc1, units=90 , activation=tf.nn.relu, name='fc2')
fc3 = tf.layers.dense(fc2, units=3 , activation=tf.nn.relu, name='fc3')

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gt_output_layer, logits=fc3))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

top_k_op = tf.nn.in_top_k(fc3, tf.argmax(gt_output_layer, axis=1), 1)
sum_correct = tf.reduce_sum(tf.cast(top_k_op, tf.float32))
accuracy = tf.divide(sum_correct,tf.constant(float(BATCH_SIZE)))

saver = tf.train.Saver()

loss_fifo = deque(100*[1.0])
acc_fifo = deque(100*[0.0])
random.seed(404)
input_vectors = np.zeros((BATCH_SIZE,15))
output_gts = np.zeros((BATCH_SIZE,3))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(100*len(training_examples)):

		# prepare batch
		batch_examples = random.sample(training_examples, BATCH_SIZE)
		for b in range(len(batch_examples)):
			input_vectors[b,:] = np.array(batch_examples[b].poses_mean_A + batch_examples[b].poses_std_A + batch_examples[b].poses_mean_B + batch_examples[b].poses_std_B + scores)
			output_gts[b,:] = np.array(batch_examples[b].nn_output)
		
		_, loss_value, acc_value = sess.run([train_step, cross_entropy, accuracy], feed_dict={input_layer: input_vectors, gt_output_layer: output_gts})
		loss_fifo.pop()
		loss_fifo.appendleft(loss_value)
		acc_fifo.pop()
		acc_fifo.appendleft(acc_value)
		if i%100==0:
			open(NN_TRAINING_BASE+'training_loss.csv', 'a').write(str(i)+' '+str(np.mean(loss_fifo))+'\n')
			open(NN_TRAINING_BASE+'training_acc.csv', 'a').write(str(i)+' '+str(np.mean(acc_fifo))+'\n')
		if i%len(training_examples)==0:
			print('Step %d: loss = %.2f accuracy = %.2f' % (i, np.mean(loss_fifo), np.mean(acc_fifo)))
			saver.save(sess, NN_TRAINING_BASE+'model.ckpt', global_step=i)



exit(0)





































