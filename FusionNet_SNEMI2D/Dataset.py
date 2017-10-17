#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Utility import *

class ImageDataFlow(RNGDataFlow):
	def __init__(self, imageDir, labelDir, size, dtype='float32', isTrain=True):
		self.dtype		= dtype
		self.imageDir 	= imageDir
		self.labelDir 	= labelDir
		self._size 		= size
		self.isTrain 	= isTrain

	def size(self):
		return self._size

	def reset_state(self):
		self.rng = get_rng(self)

	def get_data(self, shuffle=True):
		#
		# Read and store into pairs of images and labels
		#
		images = glob.glob(self.imageDir + '/*.png')
		labels = glob.glob(self.labelDir + '/*.png')

		if self._size==None:
			self._size = len(images)

		from natsort import natsorted
		images = natsorted(images)
		labels = natsorted(labels)


		#
		# Pick randomly a pair of training instance
		#
		# seed = 2015
		# np.random.seed(seed)
		for k in range(self._size):
			rand_index = np.random.randint(0, len(images))
			rand_image = np.random.randint(0, len(images))
			rand_label = np.random.randint(0, len(labels))

			image = skimage.io.imread(images[rand_index])
			label = skimage.io.imread(labels[rand_index])

			# Downsample for test ting
			image = skimage.transform.resize(image, output_shape=(DIMY, DIMX), order=1, preserve_range=True, anti_aliasing=True)
			label = skimage.transform.resize(label, output_shape=(DIMY, DIMX), order=0, preserve_range=True)

			#TODO: augmentation here
			if self.isTrain:
				seed = np.random.randint(0, 2015)
				seed_image = np.random.randint(0, 2015)
				seed_label = np.random.randint(0, 2015)

				#TODO: augmentation here
				image = self.random_flip(image, seed=seed)		
				image = self.random_reverse(image, seed=seed)
				image = self.random_square_rotate(image, seed=seed)			
				image = self.random_elastic(image, seed=seed)
				#image = skimage.util.random_noise(image, seed=seed) # TODO

				label = self.random_flip(label, seed=seed)		
				label = self.random_reverse(label, seed=seed)
				label = self.random_square_rotate(label, seed=seed)	
				label = self.random_elastic(label, seed=seed)


			image = np.expand_dims(image, axis=0)
			label = np.expand_dims(label, axis=0)

			image = np.expand_dims(image, axis=-1)
			label = np.expand_dims(label, axis=-1)

			yield [image.astype(np.float32), 
				   label.astype(np.float32)]

	def random_flip(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			np.random.seed(seed)
			random_flip = np.random.randint(1,5)
		if random_flip==1:
			flipped = image[...,::1,::-1]
			image = flipped
		elif random_flip==2:
			flipped = image[...,::-1,::1]
			image = flipped
		elif random_flip==3:
			flipped = image[...,::-1,::-1]
			image = flipped
		elif random_flip==4:
			flipped = image
			image = flipped
		return image

	def random_reverse(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			np.random.seed(seed)
			random_reverse = np.random.randint(1,2)
		if random_reverse==1:
			reverse = image[::1,...]
		elif random_reverse==2:
			reverse = image[::-1,...]
		image = reverse
		return image

	def random_square_rotate(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			np.random.seed(seed)        
		random_rotatedeg = 90*np.random.randint(0,4)
		rotated = image.copy()
		from scipy.ndimage.interpolation import rotate
		if image.ndim==2:
			rotated = rotate(image, random_rotatedeg, axes=(0,1))
		elif image.ndim==3:
			rotated = rotate(image, random_rotatedeg, axes=(1,2))
		image = rotated
		return image
				
	def random_elastic(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		old_shape = image.shape

		if image.ndim==2:
			image = np.expand_dims(image, axis=0) # Make 3D
		new_shape = image.shape
		dimx, dimy = new_shape[1], new_shape[2]
		size = np.random.randint(4,16) #4,32
		ampl = np.random.randint(2, 5) #4,8
		du = np.random.uniform(-ampl, ampl, size=(size, size)).astype(np.float32)
		dv = np.random.uniform(-ampl, ampl, size=(size, size)).astype(np.float32)
		# Done distort at boundary
		du[ 0,:] = 0
		du[-1,:] = 0
		du[:, 0] = 0
		du[:,-1] = 0
		dv[ 0,:] = 0
		dv[-1,:] = 0
		dv[:, 0] = 0
		dv[:,-1] = 0
		import cv2
		from scipy.ndimage.interpolation    import map_coordinates
		# Interpolate du
		DU = cv2.resize(du, (new_shape[1], new_shape[2])) 
		DV = cv2.resize(dv, (new_shape[1], new_shape[2])) 
		X, Y = np.meshgrid(np.arange(new_shape[1]), np.arange(new_shape[2]))
		indices = np.reshape(Y+DV, (-1, 1)), np.reshape(X+DU, (-1, 1))
		
		warped = image.copy()
		for z in range(new_shape[0]): #Loop over the channel
			# print z
			imageZ = np.squeeze(image[z,...])
			flowZ  = map_coordinates(imageZ, indices, order=1).astype(np.float32)

			warpedZ = flowZ.reshape(image[z,...].shape)
			warped[z,...] = warpedZ		
		warped = np.reshape(warped, old_shape)
		return warped


def get_data(dataDir, isTrain=True):
	if isTrain:
		num=250
	else:
		num=100

	# Process the directories 
	names = ['trainA', 'trainB'] if isTrain else ['validA', 'validB']
	dset  = ImageDataFlow(os.path.join(dataDir, names[0]),
						  os.path.join(dataDir, names[1]),
						  num, 
						  isTrain=isTrain)
	return dset



		