import argparse
import time
import sys

import json
from json import encoder

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import pandas as pd
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix

from dataset import TSNDataSet
from models import VideoModel
from utils.utils import plot_confusion_matrix

from colorama import init
from colorama import Fore, Back, Style
from tqdm import tqdm
from time import sleep
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
init(autoreset=True)

# options
parser = argparse.ArgumentParser(description="Standard video-level testing")
parser.add_argument('num_class', type=str, default="classInd.txt")
parser.add_argument('modality', type=str, choices=['ALL', 'Audio','RGB', 'Flow', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus'])
parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str)
parser.add_argument('test_target_data', type=str)
parser.add_argument('result_json', type=str)

# ========================= Model Configs ==========================
parser.add_argument('--noun_target_data', type=str, default=None)
parser.add_argument('--noun_weights', type=str, default=None)
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--test_segments', type=int, default=5)
parser.add_argument('--add_fc', default=1, type=int, metavar='M', help='number of additional fc layers (excluding the last fc layer) (e.g. 0, 1, 2, ...)')
parser.add_argument('--fc_dim', type=int, default=512, help='dimension of added fc')
parser.add_argument('--baseline_type', type=str, default='frame', choices=['frame', 'video', 'tsn'])
parser.add_argument('--frame_aggregation', type=str, default='avgpool', choices=['avgpool', 'rnn', 'temconv', 'trn-m', 'none'], help='aggregation of frame features (none if baseline_type is not video)')
parser.add_argument('--dropout_i', type=float, default=0)
parser.add_argument('--dropout_v', type=float, default=0)

#------ RNN ------
parser.add_argument('--n_rnn', default=1, type=int, metavar='M',
                    help='number of RNN layers (e.g. 0, 1, 2, ...)')
parser.add_argument('--rnn_cell', type=str, default='LSTM', choices=['LSTM', 'GRU'])
parser.add_argument('--n_directions', type=int, default=1, choices=[1, 2],
                    help='(bi-) direction RNN')
parser.add_argument('--n_ts', type=int, default=5, help='number of temporal segments')

# ========================= DA Configs ==========================
parser.add_argument('--share_params', type=str, default='Y', choices=['Y', 'N'])
parser.add_argument('--use_bn', type=str, default='none', choices=['none', 'AdaBN', 'AutoDIAL'])
parser.add_argument('--use_attn_frame', type=str, default='none', choices=['none', 'TransAttn', 'general', 'DotProduct'], help='attention-mechanism for frames only')
parser.add_argument('--use_attn', type=str, default='none', choices=['none', 'TransAttn', 'general', 'DotProduct'], help='attention-mechanism')
parser.add_argument('--n_attn', type=int, default=1, help='number of discriminators for transferable attention')

# ========================= Monitor Configs ==========================
parser.add_argument('--top', default=[1, 3, 5], nargs='+', type=int, help='show top-N categories')
parser.add_argument('--verbose', default=False, action="store_true")

# ========================= Runtime Configs ==========================
parser.add_argument('--save_confusion', type=str, default=None)
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--save_attention', type=str, default=None)
parser.add_argument('--max_num', type=int, default=-1, help='number of videos to test')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--bS', default=2, help='batch size', type=int, required=False)
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', type=str, default='')

args = parser.parse_args()
gpu_count = torch.cuda.device_count()
# New approach
num_class_str = args.num_class.split(",")
# single class
if len(num_class_str) < 1:
	raise Exception("Must specify a number of classes to train")
else:
	num_class = []
	for num in num_class_str:
		num_class.append(int(num))

criterion = torch.nn.CrossEntropyLoss().cuda()

#=== Load the network ===#
print(Fore.CYAN + 'preparing the model......')
verb_net = VideoModel(num_class, args.baseline_type, args.frame_aggregation, args.modality,
		train_segments=args.test_segments if args.baseline_type == 'video' else 1, val_segments=args.test_segments if args.baseline_type == 'video' else 1,
		base_model=args.arch, add_fc=args.add_fc, fc_dim=args.fc_dim, share_params=args.share_params,
		dropout_i=args.dropout_i, dropout_v=args.dropout_v, use_bn=args.use_bn, partial_bn=False,
		n_rnn=args.n_rnn, rnn_cell=args.rnn_cell, n_directions=args.n_directions, n_ts=args.n_ts,
		use_attn=args.use_attn, n_attn=args.n_attn, use_attn_frame=args.use_attn_frame,
		verbose=args.verbose, before_softmax=False)

verb_checkpoint = torch.load(args.weights)

verb_base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(verb_checkpoint['state_dict'].items())}
verb_net.load_state_dict(verb_base_dict)
verb_net = torch.nn.DataParallel(verb_net.cuda())
verb_net.eval()

if args.noun_weights is not None:
	noun_net = VideoModel(num_class, args.baseline_type, args.frame_aggregation, args.modality,
						  train_segments=args.test_segments if args.baseline_type == 'video' else 1,
						  val_segments=args.test_segments if args.baseline_type == 'video' else 1,
						  base_model=args.arch, add_fc=args.add_fc, fc_dim=args.fc_dim, share_params=args.share_params,
						  dropout_i=args.dropout_i, dropout_v=args.dropout_v, use_bn=args.use_bn, partial_bn=False,
						  n_rnn=args.n_rnn, rnn_cell=args.rnn_cell, n_directions=args.n_directions, n_ts=args.n_ts,
						  use_attn=args.use_attn, n_attn=args.n_attn, use_attn_frame=args.use_attn_frame,
						  verbose=args.verbose, before_softmax=False)
	noun_checkpoint = torch.load(args.noun_weights)

	noun_base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(noun_checkpoint['state_dict'].items())}
	noun_net.load_state_dict(noun_base_dict)
	noun_net = torch.nn.DataParallel(noun_net.cuda())
	noun_net.eval()
else:
	noun_net = None


#=== Data loading ===#
print(Fore.CYAN + 'loading data......')

data_length = 1 if args.modality == "RGB" else 1
num_test = len(pd.read_pickle(args.test_list).index)
if args.noun_target_data is not None:
	data_set = TSNDataSet(args.test_target_data+".pkl", args.test_list, num_dataload=num_test, num_segments=args.test_segments,
		new_length=data_length, modality=args.modality,
		image_tmpl="img_{:05d}.t7" if args.modality in ['RGB', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus'] else args.flow_prefix+"{}_{:05d}.t7",
		test_mode=True, noun_data_path=args.noun_target_data+".pkl"
		)
else:
	data_set = TSNDataSet(args.test_target_data+".pkl", args.test_list, num_dataload=num_test, num_segments=args.test_segments,
		new_length=data_length, modality=args.modality,
		image_tmpl="img_{:05d}.t7" if args.modality in ['RGB', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus'] else args.flow_prefix+"{}_{:05d}.t7",
		test_mode=True
		)
data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.bS, shuffle=False, num_workers=args.workers, pin_memory=True)

data_gen = tqdm(data_loader)

output = []
attn_values = torch.Tensor()

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

def multitask_accuracy(outputs, labels, topk=(1,)):
    """
    Args:
        outputs: tuple(torch.FloatTensor), each tensor should be of shape
            [batch_size, class_count], class_count can vary on a per task basis, i.e.
            outputs[i].shape[1] can be different to outputs[j].shape[j].
        labels: tuple(torch.LongTensor), each tensor should be of shape [batch_size]
        topk: tuple(int), compute accuracy at top-k for the values of k specified
            in this parameter.
    Returns:
        tuple(float), same length at topk with the corresponding accuracy@k in.
    """
    max_k = int(np.max(topk))
    task_count = len(outputs)
    batch_size = labels[0].size(0)
    all_correct = torch.zeros(max_k, batch_size).type(torch.ByteTensor)
    if torch.cuda.is_available():
        all_correct = all_correct.cuda()
    for output, label in zip(outputs, labels):
        _, max_k_idx = output.topk(max_k, dim=1, largest=True, sorted=True)
        # Flip batch_size, class_count as .view doesn't work on non-contiguous
        max_k_idx = max_k_idx.t()
        correct_for_task = max_k_idx.eq(label.view(1, -1).expand_as(max_k_idx))
        all_correct.add_(correct_for_task)

    accuracies = []
    for k in topk:
        all_tasks_correct = torch.ge(all_correct[:k].float().sum(0), task_count)
        accuracy_at_k = float(all_tasks_correct.float().sum(0) * 100.0 / batch_size)
        accuracies.append(accuracy_at_k)
    return tuple(accuracies)

def removeDummy(attn, out_1, out_2, pred_domain, feat, batch_size):
	attn = attn[:batch_size]
	if isinstance(out_1, (list, tuple)):
		out_1 = (out_1[0][:batch_size], out_1[1][:batch_size])
	else:
		out_1 = out_1[:batch_size]
	out_2 = out_2[:batch_size]
	pred_domain = [pred[:batch_size] for pred in pred_domain]
	feat = [f[:batch_size] for f in feat]

	return attn, out_1, out_2, pred_domain, feat

def dummyData(batch_val_ori, val_size_ori, val_data):
	# add dummy tensors to keep the same batch size for each epoch (for the last epoch)
	if batch_val_ori < args.bS:
		val_data_dummy = torch.zeros(args.bS - batch_val_ori, val_size_ori[1], val_size_ori[2])
		val_data = torch.cat((val_data, val_data_dummy))

	# add dummy tensors to make sure batch size can be divided by gpu #
	if val_data.size(0) % gpu_count != 0:
		val_data_dummy = torch.zeros(gpu_count - val_data.size(0) % gpu_count, val_data.size(1), val_data.size(2))
		val_data = torch.cat((val_data, val_data_dummy))
	return val_data

results_dict = {}
def validate(val_loader, verb_model, criterion, num_class, noun_model=None, val_labels=False):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1_verb = AverageMeter()
	top5_verb = AverageMeter()
	top1_noun = AverageMeter()
	top5_noun = AverageMeter()
	top1_action = AverageMeter()
	top5_action = AverageMeter()

	# switch to evaluate mode
	verb_model.eval()
	if noun_model is not None:
		noun_model.eval()

	end = time.time()

	verb_predictions = []
	noun_predictions = []
	results_dict = {}
	for i, (val_data_all, val_label, val_id) in enumerate(val_loader):

		if noun_model is not None:

			val_size_ori = val_data_all[0].size()  # original shape
			batch_val_ori = val_size_ori[0]
			val_data = dummyData(batch_val_ori, val_size_ori, val_data_all[0])
			val_data_noun = dummyData(batch_val_ori, val_size_ori, val_data_all[1])
		else:
			val_size_ori = val_data_all.size()  # original shape
			batch_val_ori = val_size_ori[0]
			val_data = dummyData(batch_val_ori,val_size_ori,val_data_all)


		val_label_verb = val_label[0].cuda(non_blocking=True)
		val_label_noun = val_label[1].cuda(non_blocking=True)
		with torch.no_grad():

			if args.baseline_type == 'frame':
				val_label_verb_frame = val_label_verb.unsqueeze(1).repeat(1,args.num_segments).view(-1) # expand the size for all the frames
				val_label_noun_frame = val_label_noun.unsqueeze(1).repeat(1, args.num_segments).view(-1)  # expand the size for all the frames

			# compute output
			_, _, _, _, _, attn_val_verb, out_val_verb, out_val_2_verb, pred_domain_val_verb, feat_val_verb = verb_model(val_data, val_data, [0,0,0], 0, is_train=False, reverse=False)
			# ignore dummy tensors
			attn_val_verb, out_val_verb, out_val_2_verb, pred_domain_val_verb, feat_val_verb = removeDummy(attn_val_verb, out_val_verb, out_val_2_verb, pred_domain_val_verb, feat_val_verb, batch_val_ori)
			pred_verb = out_val_verb[0]

			if noun_model is not None:
				_, _, _, _, _, attn_val_noun, out_val_noun, out_val_2_noun, pred_domain_val_noun, feat_val_noun = noun_model(val_data_noun, val_data_noun, [0,0,0], 0, is_train=False, reverse=False)
				attn_val_noun, out_val_noun, out_val_2_noun, pred_domain_val_noun, feat_val_noun = removeDummy(attn_val_noun, out_val_noun, out_val_2_noun, pred_domain_val_noun, feat_val_noun, batch_val_ori)
				pred_noun = out_val_noun[1]
			else:
				pred_noun = out_val_verb[1]
			pred_verb_cpu = pred_verb.cpu().tolist()
			pred_noun_cpu = pred_noun.cpu().tolist()
			for p_verb, p_noun, id in zip(pred_verb_cpu, pred_noun_cpu, val_id):
				verb_dict = {}
				noun_dict = {}
				for i, prob in enumerate(p_verb):
					verb_dict[str(i)] = prob
				for i, prob in enumerate(p_noun):
					noun_dict[str(i)] = prob
				results_dict[id] = {'verb': verb_dict, 'noun': noun_dict}

			noun_predictions.append(torch.argmax(pred_noun, dim=-1).cpu().numpy())
			verb_predictions.append(torch.argmax(pred_verb, dim=-1).cpu().numpy())

			# measure accuracy and record loss
			label_verb = val_label_verb_frame if args.baseline_type == 'frame' else val_label_verb
			label_noun = val_label_noun_frame if args.baseline_type == 'frame' else val_label_noun

			if args.baseline_type == 'tsn':
				pred_verb = pred_verb.view(val_label.size(0), -1, num_class).mean(dim=1) # average all the segments (needed when num_segments != val_segments)
				pred_noun = pred_noun.view(val_label.size(0), -1, num_class).mean(dim=1) # average all the segments (needed when num_segments != val_segments)

			loss_verb = criterion(pred_verb, label_verb)
			loss_noun = criterion(pred_noun, label_noun)
			loss = 0.5*(loss_verb + loss_noun)
			prec1_verb, prec5_verb = accuracy(pred_verb.data, label_verb, topk=(1, 5))
			prec1_noun, prec5_noun = accuracy(pred_noun.data, label_noun, topk=(1, 5))
			prec1_action, prec5_action = multitask_accuracy((pred_verb.data, pred_noun.data), (label_verb, label_noun),
															topk=(1, 5))

			losses.update(loss.item(), out_val_verb[0].size(0))
			top1_verb.update(prec1_verb.item(), out_val_verb[0].size(0))
			top5_verb.update(prec5_verb.item(), out_val_verb[0].size(0))
			top1_noun.update(prec1_noun.item(), out_val_verb[1].size(0))
			top5_noun.update(prec5_noun.item(), out_val_verb[1].size(0))
			top1_action.update(prec1_action, out_val_verb[1].size(0))
			top5_action.update(prec5_action, out_val_verb[1].size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

	with open(args.result_json, "w") as f:
		json.dump({'results_target':results_dict, "version": "0.2",
  "challenge": "domain_adaptation","sls_pt": 0,
  "sls_tl": 0,
  "sls_td": 0}, f)
	if val_labels:
		print(('Testing Results: Prec@1 verb {top1_verb.avg:.3f}  Prec@1 noun {top1_noun.avg:.3f} Prec@1 action {top1_action.avg:.3f} Prec@5 verb {top5_verb.avg:.3f} Prec@5 noun {top5_noun.avg:.3f} Prec@5 action {top5_action.avg:.3f} Loss {loss.avg:.5f}'
		   .format(top1_verb=top1_verb, top1_noun=top1_noun, top1_action=top1_action, top5_verb=top5_verb, top5_noun=top5_noun, top5_action=top5_action, loss=losses)))
	return top1_action.avg, top1_verb.avg, top1_noun.avg


validate(data_loader, verb_net, criterion, num_class, noun_model=noun_net, val_labels=data_set.labels_available)
