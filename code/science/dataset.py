
from __future__ import division

import pickle, random
import numpy as np
from itertools import cycle

import torch
from torch.autograd import Variable

all_feature_lengths = {'v_enc_onehot': 100,
					   'v_enc_embedding': 300,
					   'v_enc_dim300': 300,
					   'v_enc_dim2': 2,
					   'v_enc_dim10': 10,
					   'v_enc_dim50': 50,
					   'v_enc_dim100': 100,
					   'v_freq_freq': 1,
					   'v_freq_rank': 1,
					   'v_deg': 1,
					   'v_sense': 1,
					   'e_vertexsim': 1,
					   'e_dir': 3,
					   'e_rel': 46,
					   'e_weight': 1,
					   'e_source': 6,
					   'e_weightsource': 6,
					   'e_srank_abs': 1,
					   'e_srank_rel': 1,
					   'e_trank_abs': 1,
					   'e_trank_rel': 1,
					   'e_sense': 1}

class Dataset:
	def __init__(self, feature_names, train_test_split_fraction, gpu):
		self.feature_names = feature_names
		self.cached_features = dict()
		self.gpu = gpu
		for f in feature_names:
			print 'loading '+f
			self.cached_features[f] = pickle.load(
				open('features/%s.pkl'%f, 'rb'))
		sampled_problems = pickle.load(open(
			'../../data/science/paths.pkl', 'rb'))
		self.texts = dict()
		print 'loading problem plain texts'
		for id_num in sampled_problems:
			f_short = sampled_problems[id_num]['forward']['short']
			r_short = sampled_problems[id_num]['reverse']['short']
			self.texts[id_num+'f'] = f_short
			self.texts[id_num+'r'] = r_short
		print 'loading labeled pairs'
		self.all_pairs = [] # list of id tuples (good, bad)
		for l in open('../../data/science/answers.txt'):
			first, second, good = l.strip().split('_')
			if first==good:
				bad = second
			elif second==good:
				bad = first
			g_len = (len(self.texts[good].strip().split(' '))+1)/2
			b_len = (len(self.texts[bad].strip().split(' '))+1)/2
			if g_len!=4 or b_len!=4:
				continue
			self.all_pairs.append((good, bad))
		random.shuffle(self.all_pairs)

		split = int(train_test_split_fraction*len(self.all_pairs))
		self.train_pairs = self.all_pairs[:split]
		self.test_pairs = self.all_pairs[split:]

		self.train_pairs = self.train_pairs[:len(self.train_pairs)]
		self.cycled_train_pairs = cycle(self.train_pairs)

	def get_fea_len(self):
		return [all_feature_lengths[f] for f in self.feature_names]

	def get_v_fea_len(self):
		return [all_feature_lengths[f] for f in self.feature_names if f.startswith('v')]

	def get_e_fea_len(self):
		return [all_feature_lengths[f] for f in self.feature_names if f.startswith('e')]

	def get_chain_len(self, id):
		return len(self.get_features(id)[0])

	def get_features(self, id):
		v_features = []
		e_features = []
		for f in self.feature_names:
			if f.startswith('v'):
				v_features.append(self.cached_features[f][id])
			else:
				e_features.append(self.cached_features[f][id])
		v_features = zip(*v_features)
		e_features = zip(*e_features)
		return v_features, e_features

	def prepare_feature_placeholder(self, N):
		v_features = [[],[],[],[]]
		e_features = [[],[],[]]
		for feature in v_features:
			for f in self.feature_names:
				if f.startswith('v'):
					feature.append(
						np.zeros((N, all_feature_lengths[f]), dtype='float32')
					)
		for feature in e_features:
			for f in self.feature_names:
				if f.startswith('e'):
					feature.append(
						np.zeros((N, all_feature_lengths[f]), dtype='float32')
					)
		return v_features, e_features

	def get_train_pairs(self, N, randomize_dir=True):
		'''
		return a list of two lists, X_A and X_B, as well as a list y
		each list consists of two lists, which are vertex and edge representations
		each list consists of #V or #E lists, which are individual vertices/edges
		each list consists of several N x feature_len torch Variables, which are individual features
		currently only keeping chains of length 4
		if for i-th problem, the good chain is in X_A, then y[i]==1, else y[i]==0
		'''
		v_features_A, e_features_A = self.prepare_feature_placeholder(N)
		v_features_B, e_features_B = self.prepare_feature_placeholder(N)
		y = np.zeros(N, dtype='int64')

		for instance_idx in xrange(N):
			good, bad = next(self.cycled_train_pairs)
			if randomize_dir:
				good = good[:-1]+random.choice(['f','r'])
				bad = bad[:-1]+random.choice(['f','r'])
			v_good, e_good = self.get_features(good)
			v_bad, e_bad = self.get_features(bad)

			label = random.random()>0.5
			y[instance_idx] = label
			for v_idx in xrange(4):
				for v_fea_idx in xrange(len(v_good[v_idx])):
					if label:
						v_features_A[v_idx][v_fea_idx][instance_idx] = v_good[v_idx][v_fea_idx]
						v_features_B[v_idx][v_fea_idx][instance_idx] = v_bad[v_idx][v_fea_idx]
					else:
						v_features_B[v_idx][v_fea_idx][instance_idx] = v_good[v_idx][v_fea_idx]
						v_features_A[v_idx][v_fea_idx][instance_idx] = v_bad[v_idx][v_fea_idx]

			for e_idx in xrange(3):
				for e_fea_idx in xrange(len(e_good[e_idx])):
					if label:
						e_features_A[e_idx][e_fea_idx][instance_idx] = e_good[e_idx][e_fea_idx]
						e_features_B[e_idx][e_fea_idx][instance_idx] = e_bad[e_idx][e_fea_idx]
					else:
						e_features_B[e_idx][e_fea_idx][instance_idx] = e_good[e_idx][e_fea_idx]
						e_features_A[e_idx][e_fea_idx][instance_idx] = e_bad[e_idx][e_fea_idx]

		for features in [v_features_A, e_features_A, v_features_B, e_features_B]:
			for feature in features:
				for i in xrange(len(feature)):
					feature[i] = Variable(torch.from_numpy(feature[i]))
					if self.gpu:
						feature[i] = feature[i].cuda()
		y = Variable(torch.from_numpy(y))
		if self.gpu:
			y = y.cuda()
		return ((v_features_A, e_features_A), (v_features_B, e_features_B), y)

	def get_test_pairs(self, randomize_dir=True, return_id=False):
		'''
		return a list of two lists, X_A and X_B, as well as a list y
		each list consists of two lists, which are vertex and edge representations
		each list consists of #V or #E lists, which are individual vertices/edges
		each list consists of several N x feature_len torch Variables, which are individual features
		currently only keeping chains of length 4
		if for i-th problem, the good chain is in X_A, then y[i]==1, else y[i]==0
		'''
		N = len(self.test_pairs)
		v_features_A, e_features_A = self.prepare_feature_placeholder(N)
		v_features_B, e_features_B = self.prepare_feature_placeholder(N)
		y = np.zeros(N, dtype='int64')
		if return_id:
			ids = [[], []]

		for instance_idx in xrange(N):
			good, bad = self.test_pairs[instance_idx]
			if randomize_dir:
				good = good[:-1]+random.choice(['f','r'])
				bad = bad[:-1]+random.choice(['f','r'])
			v_good, e_good = self.get_features(good)
			v_bad, e_bad = self.get_features(bad)

			label = random.random()>0.5
			y[instance_idx] = label
			if return_id:
				if label:
					ids[0].append(good)
					ids[1].append(bad)
				else:
					ids[0].append(bad)
					ids[1].append(good)
			for v_idx in xrange(4):
				for v_fea_idx in xrange(len(v_good[v_idx])):
					if label:
						v_features_A[v_idx][v_fea_idx][instance_idx] = v_good[v_idx][v_fea_idx]
						v_features_B[v_idx][v_fea_idx][instance_idx] = v_bad[v_idx][v_fea_idx]
					else:
						v_features_B[v_idx][v_fea_idx][instance_idx] = v_good[v_idx][v_fea_idx]
						v_features_A[v_idx][v_fea_idx][instance_idx] = v_bad[v_idx][v_fea_idx]

			for e_idx in xrange(3):
				for e_fea_idx in xrange(len(e_good[e_idx])):
					if label:
						e_features_A[e_idx][e_fea_idx][instance_idx] = e_good[e_idx][e_fea_idx]
						e_features_B[e_idx][e_fea_idx][instance_idx] = e_bad[e_idx][e_fea_idx]
					else:
						e_features_B[e_idx][e_fea_idx][instance_idx] = e_good[e_idx][e_fea_idx]
						e_features_A[e_idx][e_fea_idx][instance_idx] = e_bad[e_idx][e_fea_idx]

		for features in [v_features_A, e_features_A, v_features_B, e_features_B]:
			for feature in features:
				for i in xrange(len(feature)):
					feature[i] = Variable(torch.from_numpy(feature[i]))
					if self.gpu:
						feature[i] = feature[i].cuda()
		y = Variable(torch.from_numpy(y))
		if self.gpu:
			y = y.cuda()
		if not return_id:
			return (v_features_A, e_features_A), (v_features_B, e_features_B), y
		else:
			return (v_features_A, e_features_A), (v_features_B, e_features_B), y, ids

	def get_pairs_for_ids(self, ids):
		'''
		ids are list of (first_chain, second_chain) tuples
		return a list of two lists, X_A and X_B
		each list consists of two lists, which are vertex and edge representations
		each list consists of #V or #E lists, which are individual vertices/edges
		each list consists of several N x feature_len torch Variables, which are individual features
		currently only keeping chains of length 4
		'''
		N = len(ids)
		v_features_A, e_features_A = self.prepare_feature_placeholder(N)
		v_features_B, e_features_B = self.prepare_feature_placeholder(N)

		for instance_idx, (first, second) in enumerate(ids):
			v_first, e_first = self.get_features(first)
			v_second, e_second = self.get_features(second)

			for v_idx in xrange(4):
				for v_fea_idx in xrange(len(v_first[v_idx])):
					v_features_A[v_idx][v_fea_idx][instance_idx] = v_first[v_idx][v_fea_idx]
					v_features_B[v_idx][v_fea_idx][instance_idx] = v_second[v_idx][v_fea_idx]

			for e_idx in xrange(3):
				for e_fea_idx in xrange(len(e_first[e_idx])):
					e_features_A[e_idx][e_fea_idx][instance_idx] = e_first[e_idx][e_fea_idx]
					e_features_B[e_idx][e_fea_idx][instance_idx] = e_second[e_idx][e_fea_idx]

		for features in [v_features_A, e_features_A, v_features_B, e_features_B]:
			for feature in features:
				for i in xrange(len(feature)):
					feature[i] = Variable(torch.from_numpy(feature[i]))
					if self.gpu:
						feature[i] = feature[i].cuda()
		return ((v_features_A, e_features_A), (v_features_B, e_features_B))

# if __name__ == '__main__':
# 	d = Dataset(['v_freq_freq', 'v_sense', 'e_source', 'e_dir', 'e_sense'], 0.9, False)
# 	good, bad = d.get_train_pairs(1000)
# 	v_good, e_good = good
# 	v_bad, e_bad = bad
# 	print len(v_good)
# 	for feature in e_good[0]:
# 		print feature
# 	print e_good
# 	print v_bad
# 	print e_bad
