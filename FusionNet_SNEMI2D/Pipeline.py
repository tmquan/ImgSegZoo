#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Utility  import *
from Model    import *
from Augment  import *
from Dataflow import *


if __name__ == '__main__':
	parser = argparser.ArgumentParser()
	parser.add_argument('--gpu',  	help='comma seperated list of GPU(s) to use.')
	parser.add_argument('--data',  	required=True, 
								    help='Data directory, contain trainA/trainB/validA/validB')
	parser.add_argument('--load', 	help='Load the model path')
	parser.add_argument('--sample', help='Run the deployment on an instance',
									action='store_true')

	args = parser.parser_args()

	# Set the logger directory
	logger.auto_set_dir()

	train_ds = get_data(args.data, isTrain=True)
	valid_ds = get_data(args.data, isTrain=False)

	# Set the GPU
	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	# Running train or deploy
	if args.sample:
		# TODO
		# sample
		pass
	else:
		# Set up configuration
		config = TrainConfig(
			model 			= 	Model(), 
			dataflow 		= 	train_ds,
			callbacks 		= 	[
				PeriodicTrigger(every_k_epochs=100, ModelSaver()),
				PeriodicTrigger(every_k_epochs=5,   InferenceRunner(valid_ds, [ScalarStats('loss_recon')])),
				PeriodicTrigger(every_k_epochs=5,   VisualizeRunner(valid_ds)),
				ScheduleHyperParamSetter('learning_rate', [(100, 1e-4), (200, 1e-5), (300, 1e-6)], interp='linear')
				],
			max_epoch		=	500, 
			session_init	=	SaverRestore(args.load) if args.load else None,
			nr_tower 		=	max(get_nr_gpu(), 1)
			)

		# Train the model
		if config.nr_tower == 1:
			# Single GPU training
			pass
		else:
			# Multi GPU training
			pass