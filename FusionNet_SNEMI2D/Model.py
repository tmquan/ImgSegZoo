#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Utility import *
from Architecture import *

class Model(ModelDesc):
	def _get_inputs(self):
		return [InputDesc(tf.float32, [None, DIMY, DIMX, DIMC], 'image'), 
				InputDesc(tf.float32, [None, DIMY, DIMX, DIMC], 'label')]

	@auto_reuse_variable_scope
	def generator(self, image):
		return arch_generator(image)


	def _build_graph(self, inputs): 
		# Split the list of inputs
		image, label = inputs

		# Additional processing of input: normalization, transposition, permutation, etc. 
		image = convert_to_range_tanh(image)
		label = convert_to_range_tanh(label)

		A = image
		B = label

		# Build graph
		with tf.variable_scope('gen'):
			AB = self.generator(A)

		# Calculate loss and total cost
		cost_total = []
		with tf.name_scope('losses'):
			loss_recon = tf.reduce_mean(tf.abs(B - AB), name='loss_recon')
			cost_total.append(loss_recon)
		add_moving_summary(loss_recon)

		self.cost = tf.add_n(cost_total, name='cost')

		# Visualize the training
		viz = tf.concat([A, B, AB], 2)
		viz = convert_to_range_imag(viz)
		viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
		tf.summary.image('colorized', viz, max_outputs=50)


	def _get_optimizer(self):
		lr = symbolic_functions.get_scalar_var('learning_rate', 2e-4, summary=True)
		return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)