#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Utility import *

def residual(inputs, filters=64, name='residual'):
	with tf.variable_scope(name):
			#
			original  = tf.identity(inputs, name='identity') 
			conv_4x4i = tf.layers.conv2d(inputs=inputs,		filters=filters,   kernel_size=3, strides=(1, 1), padding='same', activation=tf.nn.elu)
			conv_4x4m = tf.layers.conv2d(inputs=conv_4x4i, 	filters=filters/2, kernel_size=3, strides=(1, 1), padding='same', activation=tf.nn.elu)
			conv_4x4o = tf.layers.conv2d(inputs=conv_4x4m, 	filters=filters,   kernel_size=3, strides=(1, 1), padding='same', activation=tf.nn.elu)

			summation = tf.add(original, conv_4x4o, name='summation')

			return summation

def arch_generator(image, name='generator'):
	with tf.variable_scope(name):
			# Encoding path
			e1a = tf.layers.conv2d(inputs=image, 	name='e1a', 	filters=NB_FILTERS*1, kernel_size=4, strides=(2, 2), padding='same', activation=tf.nn.elu)
			r1a = residual(inputs=e1a, 	name='r1a', 	filters=NB_FILTERS*1)
			r1a = layers.dropout(inputs=r1a)
		
			e2a = tf.layers.conv2d(inputs=r1a, 	name='e2a', 	filters=NB_FILTERS*1, kernel_size=4, strides=(2, 2), padding='same', activation=tf.nn.elu)
			r2a = residual(inputs=e2a, 	name='r2a', 	filters=NB_FILTERS*1)
			r2a = layers.dropout(inputs=r2a)

			e3a = tf.layers.conv2d(inputs=r2a, 	name='e3a', 	filters=NB_FILTERS*2, kernel_size=4, strides=(2, 2), padding='same', activation=tf.nn.elu)
			r3a = residual(inputs=e3a, 	name='r3a', 	filters=NB_FILTERS*2)
			r3a = layers.dropout(inputs=r3a)

			e4a = tf.layers.conv2d(inputs=r3a, 	name='e4a', 	filters=NB_FILTERS*2, kernel_size=4, strides=(2, 2), padding='same', activation=tf.nn.elu)
			r4a = residual(inputs=e4a, 	name='r4a', 	filters=NB_FILTERS*2)
			r4a = layers.dropout(inputs=r4a)

			e5a = tf.layers.conv2d(inputs=r4a, 	name='e5a', 	filters=NB_FILTERS*4, kernel_size=4, strides=(2, 2), padding='same', activation=tf.nn.elu)
			r5a = residual(inputs=e5a, 	name='r5a', 	filters=NB_FILTERS*4)
			r5a = layers.dropout(inputs=r5a)

			print "In1 :", image.get_shape().as_list()
			print "E1a :", e1a.get_shape().as_list()
			print "R1a :", r1a.get_shape().as_list()
			print "E2a :", e2a.get_shape().as_list()
			print "R2a :", r2a.get_shape().as_list()
			print "E3a :", e3a.get_shape().as_list()
			print "R3a :", r3a.get_shape().as_list()
			print "E4a :", e4a.get_shape().as_list()
			print "R4a :", r4a.get_shape().as_list()
			print "E5a :", e5a.get_shape().as_list()
			print "R5a :", r5a.get_shape().as_list()

			# Decoding path
			r5b = residual(inputs=r5a, 			name='r5b', 	filters=NB_FILTERS*4)
			d4b = tf.layers.conv2d_transpose(inputs=r5b, 	name="d4b", 	filters=NB_FILTERS*2, kernel_size=4, strides=(2, 2), padding='same', activation=tf.nn.elu)
			a4b = tf.add(d4b, r4a,				name="a4b")

			r4b = residual(inputs=a4b, 			name='r4b', 	filters=NB_FILTERS*2)
			d3b = tf.layers.conv2d_transpose(inputs=r4b, 	name="d3b", 	filters=NB_FILTERS*2, kernel_size=4, strides=(2, 2), padding='same', activation=tf.nn.elu)
			a3b = tf.add(d3b, r3a,				name="a3b")

			r3b = residual(inputs=a3b, 			name='r3b', 	filters=NB_FILTERS*2)
			d2b = tf.layers.conv2d_transpose(inputs=r3b, 	name="d2b", 	filters=NB_FILTERS*1, kernel_size=4, strides=(2, 2), padding='same', activation=tf.nn.elu)
			a2b = tf.add(d2b, r2a,				name="a2b")

			r2b = residual(inputs=a2b, 			name='r2b', 	filters=NB_FILTERS*1)
			d1b = tf.layers.conv2d_transpose(inputs=r2b, 	name="d1b", 	filters=NB_FILTERS*1, kernel_size=4, strides=(2, 2), padding='same', activation=tf.nn.elu)
			a1b = tf.add(d1b, r1a,				name="a1b")

			a0b = tf.layers.conv2d_transpose(inputs=a1b, 	name='a0b',  	filters=NB_FILTERS*1, kernel_size=4, strides=(2, 2), padding='same', activation=tf.nn.elu)
			out = tf.layers.conv2d(inputs=a0b, name='out', filters=DIMC, kernel_size=4, strides=(1, 1), padding='same', activation=tf.tanh) #=1

			print "R5b :", r5b.get_shape().as_list()
			print "D4b :", d4b.get_shape().as_list()
			print "A4b :", a4b.get_shape().as_list()

			print "R4b :", r4b.get_shape().as_list()
			print "D3b :", d3b.get_shape().as_list()
			print "A3b :", a3b.get_shape().as_list()

			print "R3b :", r3b.get_shape().as_list()
			print "D2b :", d2b.get_shape().as_list()
			print "A2b :", a2b.get_shape().as_list()

			print "R2b :", r2b.get_shape().as_list()
			print "D1b :", d1b.get_shape().as_list()
			print "A1b :", a1b.get_shape().as_list()
			print "A0b :", a0b.get_shape().as_list()

			print "Out :", out.get_shape().as_list()

			return out