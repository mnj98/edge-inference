import tensorflow.keras.utils as utils

class Batch(utils.Sequence):
	def __init__(self, data, batch_size):
		self.data = data
		self.batch_size = batch_size

	def __len__(self):
		return 1

	def __getitem__(self, index):
		return self.data