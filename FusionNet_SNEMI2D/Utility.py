#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, argparse, glob

# Misc. libraries
from six.moves import map, zip, range
from natsort import natsorted 

# Array and image processing toolboxes
import numpy as np 
import skimage
import skimage.io
import skimage.transform
import skimage.segmentation


# Tensorpack toolbox
import tensorpack.tfutils.symbolic_functions as symbf

from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.utils.utils import get_rng
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope

# Tensorflow 
import tensorflow as tf

from tensorflow.layers import *

# Global definitions go here
DIMN = 100		# 
DIMB = 1		# Batch
DIMZ = 1
DIMY = 512
DIMX = 512		
DIMC = 1		# Channel

