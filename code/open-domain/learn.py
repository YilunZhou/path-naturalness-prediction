
from __future__ import division

import numpy as np
import torch, sys, os
from torch import nn, optim
from torch.autograd import Variable
from model import ChainEncoder, Predictor
from dataset import Dataset
from multiprocessing import Pool

def encode(enc, fea_len, v_features, e_features, A_pls, B_pls):
	'''
	return A_code, B_code of size batch_size x fea_len
	'''
	batch_size = len(A_pls)
	all_codes = dict()
	for pl in [2,3,4,5]:
		all_codes[pl] = enc((v_features[pl], e_features[pl]))
	A_code = Variable(
		torch.from_numpy(
			np.zeros((batch_size, fea_len), dtype='float32')
		)
	)
	B_code = Variable(
		torch.from_numpy(
			np.zeros((batch_size, fea_len), dtype='float32')
		)
	)
	if gpu:
		A_code = A_code.cuda()
		B_code = B_code.cuda()
	pts = {2:0, 3:0, 4:0, 5:0}
	pt_AB = 0
	for apl, bpl in zip(A_pls, B_pls):
		A_code[pt_AB] = all_codes[apl][pts[apl]]
		pts[apl] += 1
		B_code[pt_AB] = all_codes[bpl][pts[bpl]]
		pts[bpl] += 1
		pt_AB += 1
	return A_code, B_code

def train(features, fea_len, split_frac, out_file, save=False, save_folder=None):
	'''
	hyperparameters: 
		features
		amount of training data
		feature length
	'''
	if isinstance(out_file, str):
		out_file = open(out_file, 'w')
	d = Dataset(features, split_frac, 1, gpu)
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
	test_v_features, test_e_features, test_A_pls, test_B_pls, test_y = d.get_test_pairs()
	test_y = test_y.data.cpu().numpy()
	for train_iter in xrange(12000):
		v_features, e_features, A_pls, B_pls, y = d.get_train_pairs(100)
		enc.zero_grad()
		predictor.zero_grad()
		A_code, B_code = encode(enc, fea_len, v_features, e_features, A_pls, B_pls)
		softmax_output = predictor(A_code, B_code)
		loss_val = loss(softmax_output, y)
		loss_val.backward()
		optimizer.step()

		enc.zero_grad()
		predictor.zero_grad()
		test_A_code, test_B_code = encode(enc, fea_len, test_v_features, test_e_features, test_A_pls, test_B_pls)
		softmax_output = predictor(test_A_code, test_B_code).data.cpu().numpy()
		test_y_pred = softmax_output.argmax(axis=1)
		cur_acc = (test_y_pred==test_y).sum() / len(test_y)
		out_file.write('%f\n'%cur_acc)
		out_file.flush()
		if save and train_iter%50==0:
			if save_folder[-1]=='/':
				save_folder = save_folder[:-1]
			torch.save(enc.state_dict(), 
				'%s/%i_enc.model'%(save_folder, train_iter))
			torch.save(predictor.state_dict(), 
				'%s/%i_pred.model'%(save_folder, train_iter))
	out_file.close()

if __name__ == '__main__':
	gpu = True
	features = ['v_enc_dim300', 'v_freq_freq', 'v_deg', 'v_sense', 'e_vertexsim', 
		'e_dir', 'e_rel', 'e_weightsource', 'e_srank_rel', 'e_trank_rel', 'e_sense']
	feature_len = 10
	train(features, feature_len, 0.95, 'train.log', save=True, save_folder='ckpt')
