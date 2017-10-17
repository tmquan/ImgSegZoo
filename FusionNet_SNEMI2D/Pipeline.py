#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Utility  import *
from Model    import *
from Augment  import *
from Dataset  import *

class VisualizeRunner(Callback):
	def _setup_graph(self):
		self.pred = self.trainer.get_predictor(
			['image', 'label'], ['viz'])

	def _before_train(self):
		global args
		self.valid_ds = get_data(args.data, isTrain=False)

	def _trigger(self):
		for image, label in self.valid_ds.get_data():
			viz_valid = self.pred(image, label)
			viz_valid = np.squeeze(np.array(viz_valid))

			#print viz_valid.shape

			self.trainer.monitors.put_image('viz_valid', viz_valid)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu',  	help='comma seperated list of GPU(s) to use.')
	parser.add_argument('--data',  	required=True, 
								    help='Data directory, contain trainA/trainB/validA/validB')
	parser.add_argument('--load', 	help='Load the model path')
	parser.add_argument('--sample', help='Run the deployment on an instance',
									action='store_true')

	args = parser.parse_args()

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
				PeriodicTrigger(ModelSaver(), every_k_epochs=100),
				PeriodicTrigger(VisualizeRunner(), every_k_epochs=5),
				PeriodicTrigger(InferenceRunner(valid_ds, [ScalarStats('losses/loss_recon')]), every_k_epochs=5),
				ScheduledHyperParamSetter('learning_rate', [(100, 1e-4), (200, 1e-5), (300, 1e-6)], interp='linear')
				],
			max_epoch		=	500, 
			session_init	=	SaverRestore(args.load) if args.load else None,
			nr_tower 		=	max(get_nr_gpu(), 1)
			)

		# Train the model
		SyncMultiGPUTrainer(config).train()
		# if config.nr_tower == 1:
		# 	# Single GPU training
		# 	SyncMultiGPUTrainer(config).train()
		# 	pass
		# else:
		# 	# Multi GPU training
		# 	SyncMultiGPUTrainer(config).train()
		# 	pass