
from __future__ import division

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

class FeatureTransformer(nn.Module):
	'''
	take an n x d_in matrix and transform it into a n x d_out matrix
	where the n x d_in matrix is the n examples each with d_in dimensions
	'''
	def __init__(self, d_in, d_out):
		super(FeatureTransformer, self).__init__()
		self.d_in = d_in
		self.d_out = d_out
		self.linear = nn.Linear(d_in, d_out)
		self.relu = nn.ReLU()

	def forward(self, input):
		return self.relu(self.linear(input))

class ChainEncoder(nn.Module):
	'''
	encodes N chains at the same time
	assumes that each of the chains are of the same length
	'''
	def __init__(self, v_feature_lengths, e_feature_lengths, out_length, pooling):
		super(ChainEncoder, self).__init__()
		feature_enc_length = out_length
		num_layers = 1
		self.rnn_type = 'LSTM'
		self.pooling = pooling
		self.v_feature_lengths = v_feature_lengths
		self.e_feature_lengths = e_feature_lengths
		self.v_fea_encs = nn.ModuleList()
		self.e_fea_encs = nn.ModuleList()
		for d_in in self.v_feature_lengths:
			self.v_fea_encs.append(FeatureTransformer(d_in, feature_enc_length))
		for d_in in self.e_feature_lengths:
			self.e_fea_encs.append(FeatureTransformer(d_in, feature_enc_length))
		if self.rnn_type=='RNN':
			self.rnn = nn.RNN(input_size=feature_enc_length,
				hidden_size=out_length, num_layers=num_layers)
		elif self.rnn_type=='LSTM':
			self.lstm = nn.LSTM(input_size=feature_enc_length,
				hidden_size=out_length, num_layers=num_layers)

	def forward(self, input):
		'''
		input is a list of v_features, and e_features
		v_features is a list of num_vertices tuples
		each tuple is an N x d_in Variable, in which N is the batch size, and d_in is the feature length
		e_features is structured similarly
		'''
		v_features, e_features = input
		v_encs = []
		for i in xrange(len(v_features)):
			v_enc = None
			for j in xrange(len(v_features[i])):
				fea_enc = self.v_fea_encs[j]
				if v_enc is None:
					v_enc = fea_enc(v_features[i][j])
				else:
					v_enc += fea_enc(v_features[i][j])
			v_enc = v_enc / len(v_features[i])
			v_encs.append(v_enc)
		e_encs = []
		for i in xrange(len(e_features)):
			e_enc = None
			for j in xrange(len(e_features[i])):
				fea_enc = self.e_fea_encs[j]
				if e_enc is None:
					e_enc = fea_enc(e_features[i][j])
				else:
					e_enc += fea_enc(e_features[i][j])
			e_enc = e_enc / len(e_features[i])
			e_encs.append(e_enc)

		combined_encs = [0] * (len(v_encs)+len(e_encs))
		combined_encs[::2] = v_encs
		combined_encs[1::2] = e_encs
		combined_encs = torch.stack(combined_encs, dim=0) 
		# combined_encs has shape (#V+#E) x batch_size x out_length
		if self.rnn_type=='RNN':
			output, hidden = self.rnn(combined_encs)
		elif self.rnn_type=='LSTM':
			output, (hidden, cell) = self.lstm(combined_encs)
		if self.pooling=='last':
			return output[-1]
		else:
			return torch.mean(output, dim=0)

class Predictor(nn.Module):
	'''
	takes two feature vectors and produces a prediction
	'''
	def __init__(self, feature_len):
		super(Predictor, self).__init__()
		self.linear = nn.Linear(feature_len, 1)
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, vec1, vec2):
		# combined = torch.cat((vec1, vec2), dim=1)
		a = self.linear(vec1)
		b = self.linear(vec2)
		combined = torch.cat((a, b), dim=1)
		return self.logsoftmax(combined)