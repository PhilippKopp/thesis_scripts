#!/usr/bin/env python3.5

import tensorflow as tf


def l2_loss(logits, labels):
	"""Add L2Loss to all the trainable variables.
	Add summary for "Loss" and "Loss/avg".
	Args:
		logits: Logits from inference().
		labels: Labels from distorted_inputs or inputs(). 1-D tensor
						of shape [batch_size]
	Returns:
		Loss tensor of type float.
	"""
	# Calculate the average cross entropy loss across the batch.
	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels=labels, logits=logits, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)

	# The total loss is defined as the cross entropy loss plus all of the weight
	# decay terms (L2 loss).
	return tf.add_n(tf.get_collection('losses'), name='total_loss')


def softmax_loss(logits, labels, num_classes):
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=num_classes)
	loss = tf.losses.softmax_cross_entropy(
				onehot_labels=onehot_labels, logits=logits)
	return loss


def _add_loss_summaries(total_loss):
	"""Add summaries for losses in CIFAR-10 model.
	Generates moving average for all losses and associated summaries for
	visualizing the performance of the network.
	Args:
		total_loss: Total loss from loss().
	Returns:
		loss_averages_op: op for generating moving averages of losses.
	"""
	# Compute the moving average of all individual losses and the total loss.
	loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
	losses = tf.get_collection('losses')
	loss_averages_op = loss_averages.apply(losses + [total_loss])

	# Attach a scalar summary to all individual losses and the total loss; do the
	# same for the averaged version of the losses.
	for l in losses + [total_loss]:
		# Name each loss as '(raw)' and name the moving average version of the loss
		# as the original loss name.
		tf.summary.scalar(l.op.name + ' (raw)', l)
		tf.summary.scalar(l.op.name, loss_averages.average(l))

	return loss_averages_op


def inference(network, mode, batch_size, num_classes, input_image_tensor, input_alpha_tensor=[], image_size=512):
	if network=="cifar":
		import cnn_tf_graph_cifar
		cnn_tf_graph_cifar.BATCH_SIZE = batch_size
		cnn_tf_graph_cifar.NUM_CLASSES = num_classes
		return cnn_tf_graph_cifar.inference_cifar10(input_image_tensor)
	if network=="mnist":
		import cnn_tf_graph_mnist
		return cnn_tf_graph_mnist.cnn_model(input_image_tensor, mode)
	if network=="alex":
		return inference_alex(input_image_tensor,num_classes, input_size=image_size)
	if network=="alex_with_alpha":
		return inference_alex_alpha(input_image_tensor, input_alpha_tensor, num_classes, input_size=image_size)
	if network=="dcnn":
		return inference_dcnn(input_image_tensor, num_classes, mode)
	else:
		print ('ERROR: don\'t know this network')


def merge_isomaps_linear(input_tensors):

	#print (input_tensors)
	print ('TODO: Have to add ReLU here because we don\'t have them in the network anymore')
	exit(0)
	# first we normalise the confidence maps
	with tf.name_scope('weighted_merging') as scope:
		sum_confidence = tf.add_n([x[1] for x in input_tensors], name='confidence_sum')
		sum_confidence = tf.add(sum_confidence,tf.contrib.keras.backend.epsilon())
		for i in range(len(input_tensors)):
			input_tensors[i].append(tf.divide(input_tensors[i][1], sum_confidence))

		# now devide colour values by confidence
		for i in range(len(input_tensors)):

#			#print (input_tensors[i][0].shape)
#			colour_axis = len(input_tensors[i][0].shape)-1
#			#print (colour_axis)
#
#			r, g, b = tf.split(input_tensors[i][0], num_or_size_splits=3, axis=colour_axis)
#			r_ = tf.multiply(r,input_tensors[i][2], name='multiply_r')
#			g_ = tf.multiply(g,input_tensors[i][2], name='multiply_g')
#			b_ = tf.multiply(b,input_tensors[i][2], name='multiply_b')
#			#print (input_tensors[i][1].shape)
#
#			input_tensors[i].append(tf.concat([r_,g_,b_], axis=colour_axis))

			input_tensors[i].append(tf.multiply(input_tensors[i][0],input_tensors[i][2]))
		#exit(0)

		#now add all the colours
		result = tf.add_n([ x[3] for x in input_tensors])
	return result

def merge_isomaps_softmax(input_tensors):

	# first we normalise the confidence maps
	with tf.name_scope('weighted_merging') as scope:

		conf_stack = tf.concat([x[1] for x in input_tensors], axis=3)
		#print ('conf_stack',conf_stack)
		sm = tf.nn.softmax(conf_stack, dim=-1)
		#print ('sm',sm)
		for i in range(len(input_tensors)):
			input_tensors[i].append(tf.slice(sm,[0,0,0,i],[-1,-1,-1,1]))

		# now devide colour values by confidence
		for i in range(len(input_tensors)):
			input_tensors[i].append(tf.multiply(input_tensors[i][0],input_tensors[i][2]))

		#now add all the colours
		result = tf.add_n([ x[3] for x in input_tensors])
	return result




def alex_conv(image, input_size=512):
	#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/myalexnet_forward_newtf.py

	# Convolutional Layer #1
	conv1 = tf.layers.conv2d(
			inputs=image,
			filters=96,
			kernel_size=[11, 11],
			strides=(4,4),
			padding="valid",
			activation=tf.nn.relu,
			name='conv1')

	radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
	lrn1 = tf.nn.local_response_normalization(conv1,
												  depth_radius=radius,
												  alpha=alpha,
												  beta=beta,
												  bias=bias)

	# Pooling Layer #1
	pool1 = tf.layers.max_pooling2d(inputs=lrn1, pool_size=[3, 3], strides=2)

	if input_size==512:
		conv2_stride=2
	elif input_size==256:
		conv2_stride=1
	else:
		print('ALEXNET: Don\'t know input size')
		exit(0)
	conv2 = tf.layers.conv2d(
			inputs=pool1,
			filters=256,
			kernel_size=[5, 5],
			strides=(conv2_stride,conv2_stride),
			padding="valid",
			activation=tf.nn.relu,
			name='conv2')

	lrn2 = tf.nn.local_response_normalization(conv2,
												  depth_radius=radius,
												  alpha=alpha,
												  beta=beta,
												  bias=bias)
	pool2 = tf.layers.max_pooling2d(inputs=lrn2, pool_size=[3, 3], strides=2)

	conv3 = tf.layers.conv2d(
			inputs=pool2,
			filters=348,
			kernel_size=[3, 3],
			strides=(1,1),
			padding="same",
			activation=tf.nn.relu,
			name='conv3')	

	conv4 = tf.layers.conv2d(
			inputs=conv3,
			filters=348,
			kernel_size=[3, 3],
			strides=(1,1),
			padding="same",
			activation=tf.nn.relu,
			name='conv4')	

	conv5 = tf.layers.conv2d(
			inputs=conv4,
			filters=256,
			kernel_size=[3, 3],
			strides=(1,1),
			padding="same",
			activation=tf.nn.relu,
			name='conv5')	

	pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)

	pool5_flat = tf.contrib.layers.flatten(inputs=pool5)

	return pool5_flat


def inference_alex(image, num_classes, input_size=512):
	
	pool5_flat = alex_conv(image, input_size)

	fc6 = tf.layers.dense(pool5_flat, units=4096 , activation=tf.nn.relu, name='fc6')
	fc7 = tf.layers.dense(fc6, units=512 , activation=None, name='id_features')

	#conv3_flat = tf.reshape(conv3, [-1, 68208])

	logits = tf.layers.dense(inputs=fc7, units=num_classes)

	return logits, fc7

def inference_alex_alpha(image, alphas, num_classes, input_size=512):

	pool5_flat = alex_conv(image, input_size)

	#https://www.tensorflow.org/api_docs/python/tf/contrib/keras/layers/Concatenate
	pool5_with_alphas = tf.contrib.keras.layers.Concatenate(axis=1)([pool5_flat, alphas])

	fc6 = tf.layers.dense(pool5_with_alphas, units=4096 , activation=tf.nn.relu, name='fc6')
	fc7 = tf.layers.dense(fc6, units=512 , activation=None, name='id_features')

	#conv3_flat = tf.reshape(conv3, [-1, 68208])

	logits = tf.layers.dense(inputs=fc7, units=num_classes)

	return logits, fc7


def confidence_cnn1(image):

	#with tf.name_scope('conficence_cnn1') as scope:

	# layers 1
	conv1 = tf.layers.conv2d(
			inputs=image,
			filters=32,
			kernel_size=[3, 3],
			strides=(1,1),
			padding="same",
			activation=tf.nn.relu,
			name='conv1')

	# layers 2
	conv2 = tf.layers.conv2d(
			inputs=conv1,
			filters=64,
			kernel_size=[3, 3],
			strides=(1,1),
			padding="same",
			activation=tf.nn.relu,
			name='conv2')

	# layers 3
	conv3 = tf.layers.conv2d(
			inputs=conv2,
			filters=128,
			kernel_size=[3, 3],
			strides=(1,1),
			padding="same",
			activation=tf.nn.relu,
			name='conv3')

	# layers 4
	conv4 = tf.layers.conv2d(
			inputs=conv3,
			filters=32,
			kernel_size=[3, 3],
			strides=(1,1),
			padding="same",
			activation=tf.nn.relu,
			name='conv4')

	## layers 5
	conv5 = tf.layers.conv2d(
			inputs=conv4,
			filters=1,
			kernel_size=[1, 1],
			strides=(1,1),
			padding="same",
			activation=tf.nn.relu,
			name='conv5')

	return conv5

def confidence_cnn2(image):

	#with tf.name_scope('conficence_cnn1') as scope:

	# layers 1
	conv1 = tf.layers.conv2d(
			inputs=image,
			filters=32,
			kernel_size=[3, 3],
			strides=(1,1),
			padding="same",
			activation=tf.nn.relu,
			name='conv1')

	## layers 5
	conv5 = tf.layers.conv2d(
			inputs=conv1,
			filters=1,
			kernel_size=[1, 1],
			strides=(1,1),
			padding="same",
			activation=tf.nn.relu,
			name='conv5')

	return conv5


def confidence_cnn3(image, input_size=512):

	#with tf.name_scope('conficence_cnn1') as scope:
	
	# layers 1
	conv1 = tf.layers.conv2d(
			inputs=image,
			filters=32,
			kernel_size=[3, 3],
			strides=(1,1),
			padding="same",
			activation=tf.nn.relu,
			name='conv1')
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=3)

	# layers 2
	conv2 = tf.layers.conv2d(
			inputs=pool1,
			filters=64,
			kernel_size=[3, 3],
			strides=(1,1),
			padding="same",
			activation=tf.nn.relu,
			name='conv2')
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=3)

	# layers 3
	conv3 = tf.layers.conv2d(
			inputs=pool2,
			filters=128,
			kernel_size=[3, 3],
			strides=(1,1),
			padding="same",
			activation=tf.nn.relu,
			name='conv3')
	if input_size==512:
		pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[3, 3], strides=3)
		conv4_input = pool3
	elif input_size==256:
		conv4_input = conv3

	conv4 = tf.layers.conv2d(
			inputs=conv4_input,
			filters=1,
			kernel_size=[1, 1],
			strides=(1,1),
			padding="same",
			activation=None,
			name='conv4')

	#conv5 = tf.contrib.keras.layers.UpSampling2D(size=9)(conv4)
	conv5 = tf.image.resize_bilinear(conv4, [input_size, input_size])
	#conv5 = tf.image.resize_bilinear(conv4, image.shape[1])

	return conv5

def confidence_cnn4(image, input_size=512):

	if input_size==512:
		image_padded = tf.contrib.keras.layers.ZeroPadding2D(padding=(1,1))(image)
	elif input_size==256:
		image_padded=image

	# layers 1
	conv1 = tf.layers.conv2d(
			inputs=image_padded,
			filters=32,
			kernel_size=[3, 3],
			strides=(1,1),
			padding="same",
			activation=tf.nn.relu,
			name='conv1')
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=3)

	# layers 2
	conv2 = tf.layers.conv2d(
			inputs=pool1,
			filters=64,
			kernel_size=[3, 3],
			strides=(1,1),
			padding="same",
			activation=tf.nn.relu,
			name='conv2')
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=3)

	# layers 3
	conv3 = tf.layers.conv2d(
			inputs=pool2,
			filters=128,
			kernel_size=[3, 3],
			strides=(1,1),
			padding="same",
			activation=tf.nn.relu,
			name='conv3')
	if input_size==512:
		pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[3, 3], strides=3)
		deconv1 = tf.layers.conv2d_transpose(inputs=pool3, 
									filters=64,
									kernel_size=[3,3],
									strides=(3,3),
									padding="same",
									activation=tf.nn.relu,
									name="deconv1")
		input_deconv2 = deconv1
		#print ('conf4 network with size 512: this is stupid!!!!')
	elif input_size==256:
		input_deconv2 = conv3


	deconv2 = tf.layers.conv2d_transpose(inputs=input_deconv2, 
								filters=32,
								kernel_size=[3,3],
								strides=(3,3),
								padding="valid",
								activation=tf.nn.relu,
								name="deconv2")

	deconv3 = tf.layers.conv2d_transpose(inputs=deconv2, 
								filters=1,
								kernel_size=[3,3],
								strides=(3,3),
								padding="same",
								activation=None,
								name="deconv3")
	if input_size==512:
		image_correct_size = tf.contrib.keras.layers.Cropping2D(cropping=((0,1),(0,1)))(deconv3)
	elif input_size==256:
		image_correct_size = tf.contrib.keras.layers.ZeroPadding2D(padding=(2,2))(deconv3)
		#image_correct_size = deconv3


	return image_correct_size


def confidence_cnn13(image_with_alpha, input_size=512):
	image = tf.slice(image_with_alpha,[0,0,0,0],[-1,-1,-1,3])
	alpha = tf.slice(image_with_alpha,[0,0,0,3],[-1,-1,-1,1])

	#print ('image', image)
	#print ('alpha', alpha)

	visable = tf.not_equal(alpha, tf.zeros_like(alpha))

	confidence = confidence_cnn3(image, input_size)

	final_confidence = tf.where(visable, confidence, tf.zeros_like(confidence))

	#print ('final conf', final_confidence)

	return final_confidence

def confidence_cnn14(image_with_alpha, input_size=512):
	image = tf.slice(image_with_alpha,[0,0,0,0],[-1,-1,-1,3])
	alpha = tf.slice(image_with_alpha,[0,0,0,3],[-1,-1,-1,1])

	#print ('image', image)
	#print ('alpha', alpha)

	visable = tf.not_equal(alpha, tf.zeros_like(alpha))

	confidence = confidence_cnn4(image, input_size)

	final_confidence = tf.where(visable, confidence, tf.zeros_like(confidence))

	#print ('final conf', final_confidence)

	return final_confidence

def confidence_cnn23(image_with_alpha, input_size=512):
	image = tf.slice(image_with_alpha,[0,0,0,0],[-1,-1,-1,3])
	alpha = tf.slice(image_with_alpha,[0,0,0,3],[-1,-1,-1,1])

	#print ('image', image)
	#print ('alpha', alpha)

	visable = tf.not_equal(alpha, tf.zeros_like(alpha))

	confidence = confidence_cnn3(image, input_size)

	negative_confidence = tf.multiply(tf.ones_like(confidence),tf.constant(-1.0))

	final_confidence = tf.where(visable, confidence, negative_confidence)

	#print ('final conf', final_confidence)

	return final_confidence




def inference_dcnn(image, num_classes, mode):
	#Unconstrained Face Verification using Deep CNN Features

	training = False
	if mode == tf.contrib.learn.ModeKeys.TRAIN:
		training = True
	
	img_scaled = tf.layers.max_pooling2d(inputs=image, pool_size=[5, 5], strides=5)	

	## layers 1
	conv11 = tf.layers.conv2d(
			inputs=img_scaled,
			filters=32,
			kernel_size=[3, 3],
			strides=(1,1),
			padding="same",
			activation=None,
			name='conv11')

	conv11_prelu = tf.contrib.keras.layers.PReLU(alpha_initializer= tf.constant_initializer(0.25))(conv11)

	conv12 = tf.layers.conv2d(
			inputs=conv11_prelu,
			filters=64,
			kernel_size=[3, 3],
			strides=(1,1),
			padding="same",
			activation=None,
			name='conv12')

	conv12_prelu = tf.contrib.keras.layers.PReLU(alpha_initializer= tf.constant_initializer(0.25))(conv12)

	radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
	conv12_lrn = tf.nn.local_response_normalization(conv12_prelu,
												  depth_radius=radius,
												  alpha=alpha,
												  beta=beta,
												  bias=bias)

	pool1 = tf.layers.max_pooling2d(inputs=conv12_lrn, pool_size=[2, 2], strides=2, name='pool1')

	## layers 2
	conv21 = tf.layers.conv2d(
			inputs=pool1,
			filters=64,
			kernel_size=[3, 3],
			strides=(1,1),
			padding="same",
			activation=None,
			name='conv21')

	conv21_prelu = tf.contrib.keras.layers.PReLU(alpha_initializer= tf.constant_initializer(0.25))(conv21)

	conv22 = tf.layers.conv2d(
			inputs=conv21_prelu,
			filters=128,
			kernel_size=[3, 3],
			strides=(1,1),
			padding="same",
			activation=None,
			name='conv22')

	conv22_prelu = tf.contrib.keras.layers.PReLU(alpha_initializer= tf.constant_initializer(0.25))(conv22)

	conv22_lrn = tf.nn.local_response_normalization(conv22_prelu,
												  depth_radius=radius,
												  alpha=alpha,
												  beta=beta,
												  bias=bias)

	pool2 = tf.layers.max_pooling2d(inputs=conv22_lrn, pool_size=[2, 2], strides=2, name='pool2')

	## layers 3
	conv31 = tf.layers.conv2d(
			inputs=pool2,
			filters=96,
			kernel_size=[3, 3],
			strides=(1,1),
			padding="same",
			activation=None,
			name='conv31')

	conv31_prelu = tf.contrib.keras.layers.PReLU(alpha_initializer= tf.constant_initializer(0.25))(conv31)

	conv32 = tf.layers.conv2d(
			inputs=conv31_prelu,
			filters=192,
			kernel_size=[3, 3],
			strides=(1,1),
			padding="same",
			activation=None,
			name='conv32')

	conv32_prelu = tf.contrib.keras.layers.PReLU(alpha_initializer= tf.constant_initializer(0.25))(conv32)

	pool3 = tf.layers.max_pooling2d(inputs=conv32_prelu, pool_size=[2, 2], strides=2, name='pool3', padding="SAME")

	## layers 4
	conv41 = tf.layers.conv2d(
			inputs=pool3,
			filters=128,
			kernel_size=[3, 3],
			strides=(1,1),
			padding="same",
			activation=None,
			name='conv41')

	conv41_prelu = tf.contrib.keras.layers.PReLU(alpha_initializer= tf.constant_initializer(0.25))(conv41)

	conv42 = tf.layers.conv2d(
			inputs=conv41_prelu,
			filters=256,
			kernel_size=[3, 3],
			strides=(1,1),
			padding="same",
			activation=None,
			name='conv42')

	conv42_prelu = tf.contrib.keras.layers.PReLU(alpha_initializer= tf.constant_initializer(0.25))(conv42)

	pool4 = tf.layers.max_pooling2d(inputs=conv42_prelu, pool_size=[2, 2], strides=2, name='pool4', padding="SAME")

	## layers 5
	conv51 = tf.layers.conv2d(
			inputs=pool4,
			filters=160,
			kernel_size=[3, 3],
			strides=(1,1),
			padding="same",
			activation=None,
			name='conv51')

	conv51_prelu = tf.contrib.keras.layers.PReLU(alpha_initializer= tf.constant_initializer(0.25))(conv51)

	conv52 = tf.layers.conv2d(
			inputs=conv51_prelu,
			filters=320,
			kernel_size=[3, 3],
			strides=(1,1),
			padding="same",
			activation=None,
			name='conv52')

	pool5 = tf.layers.average_pooling2d(inputs=conv52, pool_size=[7, 7], strides=1, name='pool5')

	pool5_flat = tf.contrib.layers.flatten(inputs=pool5)

	dropout = tf.layers.dropout(inputs=pool5_flat, rate=0.4, training=training)

	logits = tf.layers.dense(inputs=dropout, units=num_classes)

	return logits, pool5_flat













