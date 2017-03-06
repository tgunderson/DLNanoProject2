import os
import numpy as np
import tensorflow as tf
import random


def normalize(x):
	x = x.astype('float32')
	if x.max() > 1.0:
		x /= 255
	return x
	
def one_hot_encode(x):
	"""
	One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
	: x: List of sample Labels
	: return: Numpy array of one-hot encoded labels
	"""
	# TODO: Implement Function
	# create a np array to fit whatever number of labels we receive
	
	#print (len(x))
	#this will sometimes fail the unit test because sometimes 
	#we don't get all possible numbers:
	#one_hot = np.zeros((len(x), max(x)+1))
	#so hard code the 10:
	one_hot = np.zeros((len(x), 10))
	#print (one_hot.shape)
	# set the appropriate elements
	one_hot[np.arange(len(x)),x] = 1	
	return one_hot
	

def neural_net_image_input(image_shape):
	"""
	Return a Tensor for a bach of image input
	: image_shape: Shape of the images
	: return: Tensor for image input.
	"""
	# TODO: Implement Function
	shape = [None] 
	for item in image_shape:
		shape.append(item)
	
	return tf.placeholder(tf.float32, shape=shape, name='x')


def neural_net_label_input(n_classes):
	"""
	Return a Tensor for a batch of label input
	: n_classes: Number of classes
	: return: Tensor for label input.
	"""
	# TODO: Implement Function
	return tf.placeholder(tf.float32, shape=[None,n_classes], name='y')
	
def neural_net_keep_prob_input():
	"""
	Return a Tensor for keep probability
	: return: Tensor for keep probability.
	"""
	# TODO: Implement Function
	return tf.placeholder(tf.float32, shape=None, name='keep_prob')
	
def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
	"""
	Apply convolution then max pooling to x_tensor
	:param x_tensor: TensorFlow Tensor
	:param conv_num_outputs: Number of outputs for the convolutional layer
	:param conv_ksize: kernal size 2-D Tuple for the convolutional layer
	:param conv_strides: Stride 2-D Tuple for convolution
	:param pool_ksize: kernal size 2-D Tuple for pool
	:param pool_strides: Stride 2-D Tuple for pool
	: return: A tensor that represents convolution and max pooling of x_tensor\
	"""
	# TODO: Implement Function
	# I needed to take apart the x_tensor so I could 
	# be sure of what I was doing
	tensor_shape = x_tensor.get_shape().as_list()

	batch = tensor_shape[0]
	in_height = tensor_shape[1]
	in_width = tensor_shape[2]
	in_channels = tensor_shape[3] 
	
	filter_height = conv_ksize[0]
	filter_width = conv_ksize[1]
	pool_height = pool_ksize[0]
	pool_width = pool_ksize[1]
	
	out_channels = conv_num_outputs
		
	filter_bias = tf.Variable(tf.zeros(conv_num_outputs))
	
	padding = 'VALID'
		
	filter = tf.Variable(tf.truncated_normal((filter_height, filter_width, in_channels, out_channels)))
	
	conv = tf.nn.conv2d(x_tensor, 
						filter, 
						strides=[1, conv_strides[0], conv_strides[1], 1], 
						padding=padding) + filter_bias
						
	conv = tf.nn.max_pool(conv,
							ksize=[1,pool_height, pool_width, 1], 
							strides = [1, pool_strides[0], pool_strides[1],1], 
							padding=padding)

	return conv
	
def flatten(x_tensor):
	"""
	Flatten x_tensor to (Batch Size, Flattened Image Size)
	: x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
	: return: A tensor of size (Batch Size, Flattened Image Size).
	"""
	# TODO: Implement Function
	
	shape = x_tensor.get_shape().as_list()
	dim = np.prod(shape[1:])
	return tf.reshape(x_tensor, [-1, dim])
	
def fully_conn(x_tensor, num_outputs):
	"""
	Apply a fully connected layer to x_tensor using weight and bias
	: x_tensor: A 2-D tensor where the first dimension is batch size.
	: num_outputs: The number of output that the new tensor should be.
	: return: A 2-D tensor where the second dimension is num_outputs.
	"""
	# TODO: Implement Function
	
	#print (x_tensor.get_shape())
	
	# weights
	weights = tf.Variable(tf.random_normal([x_tensor.get_shape().as_list()[1], num_outputs]))
	# bias
	bias = tf.Variable(tf.random_normal([num_outputs]))
	# connect:
	connected = tf.reshape(x_tensor, [-1, weights.get_shape().as_list()[0]])
	# add the weights and bias
	connected = tf.add(tf.matmul(connected, weights), bias)
	
	return connected
	
def output(x_tensor, num_outputs):
	"""
	Apply a output layer to x_tensor using weight and bias
	: x_tensor: A 2-D tensor where the first dimension is batch size.
	: num_outputs: The number of output that the new tensor should be.
	: return: A 2-D tensor where the second dimension is num_outputs.
	"""
	# TODO: Implement Function
	
	# weights
	weights = tf.Variable(tf.random_normal([x_tensor.get_shape().as_list()[1], num_outputs]))
	# bias
	bias = tf.Variable(tf.random_normal([num_outputs]))
	out = tf.reshape(x_tensor, [-1, weights.get_shape().as_list()[0]])
	out = tf.add(tf.matmul(out, weights), bias)
	
	return out


def test_normalize(normalize):
	test_shape = (np.random.choice(range(1000)), 32, 32, 3)
	test_numbers = np.random.choice(range(256), test_shape)
	normalize_out = normalize(test_numbers)
	
	assert type(normalize_out).__module__ == np.__name__,\
		'Not Numpy Object'
	assert normalize_out.shape == test_shape,\
		'Incorrect Shape. {} shape found'.format(normalize_out.shape)
	assert normalize_out.max() <= 1 and normalize_out.min() >= 0,\
		'Incorect Range. {} to {} found'.format(normalize_out.min(), normalize_out.max())
	print ("Normalization OK")
	
test_normalize(normalize)
	
def test_one_hot_encode(one_hot_encode):
	test_shape = np.random.choice(range(1000))
	test_numbers = np.random.choice(range(10), test_shape)
	one_hot_out = one_hot_encode(test_numbers)
	assert type(one_hot_out).__module__ == np.__name__,\
		'Not Numpy Object'
	assert one_hot_out.shape == (test_shape, 10),\
		'Incorrect Shape. {} shape found'.format(one_hot_out.shape)
	
	n_encode_tests = 5
	test_pairs = list(zip(test_numbers, one_hot_out))
	test_indices = np.random.choice(len(test_numbers), n_encode_tests)
	labels = [test_pairs[test_i][0] for test_i in test_indices]
	enc_labels = np.array([test_pairs[test_i][1] for test_i in test_indices])
	new_enc_labels = one_hot_encode(labels)
	
	assert np.array_equal(enc_labels, new_enc_labels),\
		'Encodings returned different results for the same numbers.\n' \
		'For the first call it returned:\n' \
		'{}\n' \
		'For the second call it returned\n' \
		'{}\n' \
		'Make sure you save the map of labels to encodings outside of the function.'.format(enc_labels, new_enc_labels)
		
	print ("One Hot OK")
	
test_one_hot_encode(one_hot_encode)

def test_nn_image_inputs(neural_net_image_input):
    image_shape = (32, 32, 3)
    nn_inputs_out_x = neural_net_image_input(image_shape)
    
  #  print (nn_inputs_out_x.get_shape().as_list())
   # print ([None, image_shape[0], image_shape[1], image_shape[2]])
    assert nn_inputs_out_x.get_shape().as_list() == [None, image_shape[0], image_shape[1], image_shape[2]],\
        'Incorrect Image Shape.  Found {} shape'.format(nn_inputs_out_x.get_shape().as_list())

    assert nn_inputs_out_x.op.type == 'Placeholder',\
        'Incorrect Image Type.  Found {} type'.format(nn_inputs_out_x.op.type)

    assert nn_inputs_out_x.name == 'x:0', \
        'Incorrect Name.  Found {}'.format(nn_inputs_out_x.name)

    print('Image Input Tests Passed.')

test_nn_image_inputs(neural_net_image_input)

def test_nn_label_inputs(neural_net_label_input):
    n_classes = 10
    nn_inputs_out_y = neural_net_label_input(n_classes)

    assert nn_inputs_out_y.get_shape().as_list() == [None, n_classes],\
        'Incorrect Label Shape.  Found {} shape'.format(nn_inputs_out_y.get_shape().as_list())

    assert nn_inputs_out_y.op.type == 'Placeholder',\
        'Incorrect Label Type.  Found {} type'.format(nn_inputs_out_y.op.type)

    assert nn_inputs_out_y.name == 'y:0', \
        'Incorrect Name.  Found {}'.format(nn_inputs_out_y.name)

    print('Label Input Tests Passed.')
    
test_nn_label_inputs(neural_net_label_input)

def test_nn_keep_prob_inputs(neural_net_keep_prob_input):
    nn_inputs_out_k = neural_net_keep_prob_input()

    assert nn_inputs_out_k.get_shape().ndims is None,\
        'Too many dimensions found for keep prob.  Found {} dimensions.  It should be a scalar (0-Dimension Tensor).'.format(nn_inputs_out_k.get_shape().ndims)

    assert nn_inputs_out_k.op.type == 'Placeholder',\
        'Incorrect keep prob Type.  Found {} type'.format(nn_inputs_out_k.op.type)

    assert nn_inputs_out_k.name == 'keep_prob:0', \
        'Incorrect Name.  Found {}'.format(nn_inputs_out_k.name)

    print('Keep Prob Tests Passed.')
    
test_nn_keep_prob_inputs(neural_net_keep_prob_input)

def test_con_pool(conv2d_maxpool):
	test_x = tf.placeholder(tf.float32, [None, 32, 32, 5])
	test_num_outputs = 10
	test_con_k = (2, 2)
	test_con_s = (4, 4)
	test_pool_k = (2, 2)
	test_pool_s = (2, 2)
	
	conv2d_maxpool_out = conv2d_maxpool(test_x, test_num_outputs, test_con_k, test_con_s, test_pool_k, test_pool_s)
	
	assert conv2d_maxpool_out.get_shape().as_list() == [None, 4, 4, 10],\
		'Incorrect Shape.  Found {} shape'.format(conv2d_maxpool_out.get_shape().as_list())
		
	print('cov2d_maxpool OK!')
	
test_con_pool(conv2d_maxpool)

def test_flatten(flatten):
	test_x = tf.placeholder(tf.float32, [None, 10, 30, 6])
	flat_out = flatten(test_x)
	assert flat_out.get_shape().as_list() == [None, 10*30*6],\
	'Incorrect Shape.  Found {} shape'.format(flat_out.get_shape().as_list())

	print ('Flatten OK!')
	
test_flatten(flatten)


def test_fully_conn(fully_conn):
    test_x = tf.placeholder(tf.float32, [None, 128])
    test_num_outputs = 40

    fc_out = fully_conn(test_x, test_num_outputs)

    assert fc_out.get_shape().as_list() == [None, 40],\
        'Incorrect Shape.  Found {} shape'.format(fc_out.get_shape().as_list())

    print ('Fully Connected OK!')
    
test_fully_conn(fully_conn)

def test_output(output):
    test_x = tf.placeholder(tf.float32, [None, 128])
    test_num_outputs = 40

    output_out = output(test_x, test_num_outputs)

    assert output_out.get_shape().as_list() == [None, 40],\
        'Incorrect Shape.  Found {} shape'.format(output_out.get_shape().as_list())

    print ('output OK!')
    
test_output(output)