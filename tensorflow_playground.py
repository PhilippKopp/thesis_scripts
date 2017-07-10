#!/usr/bin/env python3.5

import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES']=''


#### simple Tensor add, nodes, 

#node1 = tf.constant(3.0, tf.float32)
#node2 = tf.constant(4.0) # also tf.float32 implicitly
#sess = tf.Session()
#print(sess.run([node1, node2]))
#node3 = tf.add(node1, node2)
#print("node3: ", node3)
#print("sess.run(node3): ",sess.run(node3))


#W = tf.Variable([.3], tf.float32)
#b = tf.Variable([-.3], tf.float32)
#x = tf.placeholder(tf.float32)
#linear_model = W * x + b

#y = tf.placeholder(tf.float32)
#squared_deltas = tf.square(linear_model - y)
#loss = tf.reduce_sum(squared_deltas)

#init = tf.global_variables_initializer()
#sess.run(init)

#fixW = tf.assign(W, [-1.])
#fixb = tf.assign(b, [1.])
#sess.run([fixW, fixb])
#print(sess.run(linear_model, {x:[1,2,3,4]}))
#optimizer = tf.train.GradientDescentOptimizer(0.01)
#train = optimizer.minimize(loss)

#for i in range(1000):
#  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

#print(sess.run([W, b]))

#print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))





###### mnist example

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#
#sess = tf.InteractiveSession()
##sess = tf.Session()
#
#x = tf.placeholder(tf.float32, shape=[None, 784])
#y_ = tf.placeholder(tf.float32, shape=[None, 10])
#
#W = tf.Variable(tf.zeros([784,10]))
#b = tf.Variable(tf.zeros([10]))
#
#sess.run(tf.global_variables_initializer())
#
#y = tf.matmul(x,W) + b
#
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#
#for _ in range(1000):
#  batch = mnist.train.next_batch(100)
#  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
#
#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))





def load_image(image_path):

	image_file_contents = tf.read_file(image_path)
	image_uint8 = tf.image.decode_image(image_file_contents, channels=3)
	image_uint8.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 4])
	image = tf.cast(image_uint8, tf.float32)



# test of merging
import cnn_tf_graphs
import numpy as np
import cv2
import glob

#image_path1 = tf.placeholder(tf.string)
#image1 = load_image(image_path1)
#colour1 = tf.slice(image1, [0,0,0], [-1,-1,3])
#conf1 = tf.slice(image1, [0,0,3], [-1,-1,1])
#
#image_path2 = tf.placeholder(tf.string)
#image2 = load_image(image_path2)
#colour2 = tf.slice(image2, [0,0,0], [-1,-1,3])
#conf2 = tf.slice(image2, [0,0,3], [-1,-1,1])



isomaps = []
#path1 = '/user/HS204/m09113/my_project_folder/face_synthesis_josef_mail/fitting_texture/M1000_04_L0_V0S_N_small.isomap.png'
#path2 = '/user/HS204/m09113/my_project_folder/face_synthesis_josef_mail/fitting_texture/M1000_22_L0_V9R_N_small.isomap.png'
#isomaps.append(path1)
#isomaps.append(path2)

#isomaps = glob.glob('/user/HS204/m09113/my_project_folder/IJB_A/multi_iter75_reg30/id_wise/540/*isomap.png')
isomaps = glob.glob('/user/HS204/m09113/my_project_folder/Boris/new_isomaps/*isomap.png')
print (len(isomaps))

inputs = []
placeholders = []
for i in range(len(isomaps)):
	img_np = cv2.imread(isomaps[i], cv2.IMREAD_UNCHANGED)
#	#colour_np = img_np[:,:,:3].astype(np.float32)
	inputs.append(img_np)
#
	colour = tf.placeholder(tf.float32, shape=(512, 512, 3))
	conf = tf.placeholder(tf.float32, shape=(512, 512, 1))
	placeholders.append([colour, conf])

#img1_np = cv2.imread(path1, cv2.IMREAD_UNCHANGED)
#img2_np = cv2.imread(path2, cv2.IMREAD_UNCHANGED)
#colour1_np = img1_np[:,:,:3].astype(np.float32)
#colour2_np = img2_np[:,:,:3].astype(np.float32)
#
#colour1 = tf.placeholder(tf.float32, shape=(512, 512, 3))
#colour2 = tf.placeholder(tf.float32, shape=(512, 512, 3))
#conf1 = tf.placeholder(tf.float32, shape=(512, 512, 1))
#conf2 = tf.placeholder(tf.float32, shape=(512, 512, 1))

#merged, extended_list = cnn_tf_graphs.merge_isomaps([[colour1, conf1], [colour2, conf2]])

merged = cnn_tf_graphs.merge_isomaps(placeholders)
#print (len(placeholders))
#merged, extended_list = cnn_tf_graphs.merge_isomaps([[placeholders[0][0], placeholders[0][1]], [placeholders[1][0], placeholders[1][1]] ])
#merged = cnn_tf_graphs.merge_isomaps([[colour2, conf2]])

merged_uint8 = tf.cast(merged, tf.uint8)

encoded = tf.image.encode_png(merged_uint8)
#write_file_op = tf.write_file('/user/HS204/m09113/my_project_folder/face_synthesis_josef_mail/fitting_texture/tf_merged_2.png', encoded)
write_file_op = tf.write_file('/user/HS204/m09113/my_project_folder/Boris/new_isomaps/tf_merge.png', encoded)

#for i, elem in enumerate(extended_list):
#	print ('writing elem',i)
#	norm_uint8 = tf.cast(elem[2], tf.uint8)
#
#	norm_encoded = tf.image.encode_png(norm_uint8)
#	write_file_op = tf.write_file('/user/HS204/m09113/my_project_folder/face_synthesis_josef_mail/fitting_texture/tf_im'+str(i)+'.png', encoded)


sess = tf.Session()

#sess.run(write_file_op, feed_dict={image_path1: path1, image_path2: path2})
feed_dict = {}
for i in range(len(isomaps)):
#	temp = inputs[i][:,:,:3].astype(np.float32)
#	feed_dict[placeholders[i][0]] = temp[...,::-1]
	feed_dict[placeholders[i][0]] = inputs[i][:,:,:3].astype(np.float32)[:,:,::-1]
	feed_dict[placeholders[i][1]] = inputs[i][:,:,3, None].astype(np.float32)
	#feed_dict[placeholders[i][1]] = np.load(isomaps[i].split('.')[0]+'_cnn_conf01.npy').astype(np.float32)

#print (len(feed_dict))
#if np.array_equal(colour1_np[...,::-1] , feed_dict[placeholders[0][0]]):
#	print ('okay good')
#if np.array_equal(img1_np[:,:,3, None].astype(np.float32) , feed_dict[placeholders[0][1]]):
#	print ('okay good')



#cv2.waitKey()
#cv2.destroyAllWindows()	
#cv2.imshow('window', inputs[0][:,:,:3])
#cv2.imwrite('/user/HS204/m09113/my_project_folder/face_synthesis_josef_mail/fitting_texture/tf_part1.png', inputs[0][:,:,:3])

#sess.run(write_file_op, feed_dict={placeholders[0][0]: inputs[0][:,:,:3].astype(np.float32)[...,::-1], 
#								   placeholders[1][0]: inputs[1][:,:,:3].astype(np.float32)[...,::-1], 
#								   placeholders[0][1]: inputs[0][:,:,3, None].astype(np.float32), 
#								   placeholders[1][1]: inputs[1][:,:,3, None].astype(np.float32)})


sess.run(write_file_op, feed_dict=feed_dict)

#sess.run(write_file_op, feed_dict={colour1: img1_np[:,:,:3].astype(np.float32)[...,::-1], #colour1_np[...,::-1], 
#								   colour2: img2_np[:,:,:3].astype(np.float32)[...,::-1], #colour2_np[...,::-1], 
#								   conf1: img1_np[:,:,3, None].astype(np.float32), 
#								   conf2: img2_np[:,:,3, None].astype(np.float32)})

#sess.run(write_file_op, feed_dict={colour1: inputs[0][:,:,:3].astype(np.float32)[...,::-1], #colour1_np[...,::-1], 
#								   colour2: inputs[1][:,:,:3].astype(np.float32)[...,::-1], #colour2_np[...,::-1], 
#								   conf1: inputs[0][:,:,3, None].astype(np.float32), 
#								   conf2: inputs[1][:,:,3, None].astype(np.float32)})


