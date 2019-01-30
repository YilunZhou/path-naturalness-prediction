
from __future__ import division

import numpy as np
import torch, sys, os
from torch import nn, optim
from torch.autograd import Variable
from model import ChainEncoder, Predictor
from dataset import Dataset
from multiprocessing import Pool

def train(features, fea_len, split_frac, out_file):
	if isinstance(out_file, str):
		out_file = open(out_file, 'w')
	d = Dataset(features, split_frac, gpu)
	print 'defining architecture'
	enc = ChainEncoder(d.get_v_fea_len(), d.get_e_fea_len(), fea_len, 'last')
	predictor = Predictor(fea_len)
	loss = nn.NLLLoss()
	if gpu:
		enc.cuda()
		predictor.cuda()
		loss.cuda()

	optimizer = optim.Adam(list(enc.parameters())+list(predictor.parameters()))

	print 'training'
	test_chain_A, test_chain_B, test_y = d.get_test_pairs()
	test_y = test_y.data.cpu().numpy()
	for train_iter in xrange(4000):
		chains_A, chains_B, y = d.get_train_pairs(1000)
		enc.zero_grad()
		predictor.zero_grad()
		output_A = enc(chains_A)
		output_B = enc(chains_B)
		softmax_output = predictor(output_A, output_B)
		loss_val = loss(softmax_output, y)
		loss_val.backward()
		optimizer.step()

		enc.zero_grad()
		predictor.zero_grad()
		output_test_A = enc(test_chain_A)
		output_test_B = enc(test_chain_B)
		softmax_output = predictor(output_test_A, output_test_B).data.cpu().numpy()
		test_y_pred = softmax_output.argmax(axis=1)
		cur_acc = (test_y_pred==test_y).sum() / len(test_y)
		print 'test acc:', cur_acc
		out_file.write('%f\n'%cur_acc)
		if train_iter%50==0:
			torch.save(enc.state_dict(), 
				'ckpt/%i_encoder.model'%train_iter)
			torch.save(predictor.state_dict(), 
				'ckpt/%i_predictor.model'%train_iter)
	out_file.close()

gpu = True
features = ['v_enc_dim300', 'v_freq_freq', 'v_deg', 'v_sense', 'e_vertexsim', 
	'e_dir', 'e_rel', 'e_weightsource', 'e_srank_rel', 'e_trank_rel', 'e_sense']
feature_len = 20
split_frac = 0.8
train(features, feature_len, split_frac, 'train.log')
