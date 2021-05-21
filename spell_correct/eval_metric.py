import re
from tqdm import tqdm
from termcolor import colored

from edit_distance import SequenceMatcher
import numpy as np


class Evaluation:
	def __init__(self, already_split, pad_label=None, eos_label=None):
		self.already_split = already_split
		self.pad_label = pad_label
		self.eos_label = eos_label
		self.total_e, self.correct_e = 0, 0
		self.total_ne, self.correct_ne = 0, 0


	def spelling_correction_eval(self, src, tgt, system):
		for x, y_gold, y_pred in tqdm(zip(src, tgt, system)):
			if not self.already_split:
				y_pred = re.sub(r'(\b)(\w+)(\b)', r'\1 \2 \3', y_pred)
				y_pred = re.sub(r'( ){2,}', r'\1', y_pred)
				y_gold = re.sub(r'(\b)(\w+)(\b)', r'\1 \2 \3', y_gold)
				y_gold = re.sub(r'( ){2,}', r'\1', y_gold)
				y_pred, y_gold = y_pred.strip(), y_gold.strip()
			
			_, y_gold_e_ne = self.tokenize_and_align(
				x, y_gold,
				mask_indexes=(self.pad_label, self.pad_label),
				align_subsequences=True)

			y_pred_side, y_gold_side = self.tokenize_and_align(
				y_pred, y_gold,
				mask_indexes=(self.pad_label, self.pad_label),
				align_subsequences=False)

			# Expand gold in case there are embedded <space>'s inside the tokens
			y_gold_e_ne = [(token, tokens[1])
                            for tokens in y_gold_e_ne for token in tokens[0].split('<space>')]
			y_gold_side = [token for token in y_gold_side if token[1] != 'i']
			assert len(y_gold_e_ne) == len(y_gold_side), f'Error in alignment' 
			
			for gold_e_ne, gold_side, pred_side in zip(y_gold_e_ne, y_gold_side, y_pred_side):
				if gold_e_ne[1] == 'e':
					self.total_e += 1
					if gold_side[1] == 'e' and pred_side[1] == 'e':
						self.correct_e += 1
				elif gold_e_ne[1] == 'ne':
					self.total_ne += 1
					if gold_side[1] == 'e' and pred_side[1] == 'e':
						self.correct_ne += 1

		e_accuracy = self.correct_e / self.total_e
		ne_accuracy = self.correct_ne / self.total_ne
		self.total_e, self.correct_e = 0, 0
		self.total_ne, self.correct_ne = 0, 0
		return e_accuracy, ne_accuracy

	def tokenize_and_align(self, seq1, seq2, mask_indexes, align_subsequences):
		"""Returns an aligned version of the gold and pred tokens."""
		tokens = []
		for s_type, sent in enumerate([seq1, seq2]):
			tokens.append([])
			tokenized_sent = sent if self.already_split else sent.strip().split()
			if mask_indexes[0] and mask_indexes[1]:
				mask_index = Evaluation.find_mask_index(tokenized_sent, mask_indexes[s_type])
				tokenized_sent = tokenized_sent[:mask_index]
			for i, token in enumerate(tokenized_sent):
				tokens[-1].append([token, None])

		seq1, seq2 = Evaluation.align(tokens[0], tokens[1])
		if align_subsequences:
			seq1, seq2 = Evaluation.align_subsequences(seq1, seq2)
		return seq1, seq2

	
	@staticmethod
	def find_mask_index(seq, s_type):
		if isinstance(seq, list):
			try:
				mask_index = seq.index(s_type)
			except:
				mask_index = len(seq)
		else:
			mask_index = np.where(seq == s_type)[0]
			if mask_index.size:
				mask_index = mask_index[0]
			else:
				mask_index = len(seq)

	@staticmethod
	def align(src, tgt):
		"""Corrects misalignments between the gold and predicted tokens
		which will almost almost always have different lengths due to inserted, 
		deleted, or substituted tookens in the predicted systme output."""

		sm = SequenceMatcher(
			a=list(map(lambda x: x[0], tgt)), b=list(map(lambda x: x[0], src)))
		tgt_temp, src_temp = [], []
		opcodes = sm.get_opcodes()
		for tag, i1, i2, j1, j2 in opcodes:
			# If they are equal, do nothing except lowercase them
			if tag == 'equal':
				for i in range(i1, i2):
					tgt[i][1] = 'e'
					tgt_temp.append(tgt[i])
				for i in range(j1, j2):
					src[i][1] = 'e'
					src_temp.append(src[i])
			# For insertions and deletions, put a filler of '***' on the other one, and
			# make the other all caps
			elif tag == 'delete':
				for i in range(i1, i2):
					tgt[i][1] = 'd'
					tgt_temp.append(tgt[i])
				for i in range(i1, i2):
					src_temp.append(tgt[i])
			elif tag == 'insert':
				for i in range(j1, j2):
					src[i][1] = 'i'
					tgt_temp.append(src[i])
				for i in range(j1, j2):
					src_temp.append(src[i])
			# More complicated logic for a substitution
			elif tag == 'replace':
				for i in range(i1, i2):
					tgt[i][1] = 's'
				for i in range(j1, j2):
					src[i][1] = 's'
				tgt_temp += tgt[i1:i2]
				src_temp += src[j1:j2]

		return src_temp, tgt_temp

	@staticmethod
	def align_subsequences(src_sub, tgt_sub):
		def process_ne(src_sub, tgt_sub):
			src_temp, tgt_temp = [], []
			# If there are 'i' and 'd' tokens in addition to 's', then there is splitting
			# We should should align at the character level
			if [True for t in src_sub if t[1] != 's']:
				src_temp_, tgt_temp_ = Evaluation.soft_align(tgt_sub, src_sub)
				src_temp += src_temp_
				tgt_temp += tgt_temp_
			# Else they are already aligned but not equal
			else:
				for j in range(len(src_sub)):
					src_temp.append((src_sub[j][0], 'ne'))
					tgt_temp.append((tgt_sub[j][0], 'ne'))
			return src_temp, tgt_temp

		start, end = -1, -1
		src_temp, tgt_temp = [], []
		for i, token in enumerate(src_sub):
			op = token[1]
			if start == -1 and op == 'e':
				src_temp.append(tuple(src_sub[i]))
				tgt_temp.append(tuple(tgt_sub[i]))
			elif start == -1 and op != 'e':
				start = i
			elif start != -1 and op == 'e':
				end = i
				src_temp_, tgt_temp_ = process_ne(
					src_sub[start:end], tgt_sub[start:end])
				src_temp += src_temp_
				tgt_temp += tgt_temp_
				# Add first token with value 'e'
				src_temp.append(tuple(src_sub[i]))
				tgt_temp.append(tuple(tgt_sub[i]))
				start, end = -1, -1
		end = i + 1
		# If last operation is not e and we are in the
		# middle of a (possibly) badly aligned subsequence
		if start != -1:
			src_temp_, tgt_temp_ = process_ne(
				src_sub[start:end], tgt_sub[start:end])
			src_temp += src_temp_
			tgt_temp += tgt_temp_

		return src_temp, tgt_temp

	@staticmethod
	def soft_align(tgt, src):
		"""Alignment at the character level."""
		src = ' '.join([token[0] for token in src if token[1] != 'd'])
		tgt = ' '.join([token[0] for token in tgt if token[1] != 'i'])
		src_temp = [[char, 'n'] for char in src]
		tgt_temp = [[char, 'n'] for char in tgt]
		src_temp, tgt_temp = Evaluation.align(src_temp, tgt_temp)
		space_anchors = [0]
		for i, char in enumerate(src_temp):
			if char[0] == ' ' and char[1] == 'e':
				space_anchors.append(i + 1)
		space_anchors.append(len(src_temp) + 1)

		# At this point, each sequence of characters delimited by two space anchors
		# is most definitely a word unit (which may or may not be split)
		src_temp_, tgt_temp_ = [], []
		for i in range(len(space_anchors) - 1):
			src_sub_temp = src_temp[space_anchors[i]:space_anchors[i+1] - 1]
			tgt_sub_temp = tgt_temp[space_anchors[i]:space_anchors[i+1] - 1]
			src_sub_temp = ''.join([char[0] if char[0] != ' ' else '<space>'
                           for char in src_sub_temp if char[1] != 'd'])
			tgt_sub_temp = ''.join([char[0] if char[0] != ' ' else '<space>'
                           for char in tgt_sub_temp if char[1] != 'i'])
			src_temp_.append((src_sub_temp, 'ne'))
			tgt_temp_.append((tgt_sub_temp, 'ne'))
		return src_temp_, tgt_temp_


	def compute_metrics(self):
		precision = self.tp / (self.tp + self.fp) if self.tp or self.fp else 0
		recall = self.tp / (self.tp + self.fn) if self.tp or self.fn else 0
		fscore = 2 * precision * \
			recall / (precision + recall) if precision or recall else 0
		metrics = {'precision': precision,
                    'recall': recall,
                    'fscore': fscore}
		return metrics


	@staticmethod
	def print_metrics(metrics):
		print('recall'.ljust(11), 'precision'.ljust(11), 'fscore'.ljust(11))
		print(str(round(metrics['recall'], 2)).ljust(11),
						str(round(metrics['precision'], 2)).ljust(11),
						str(round(metrics['fscore'], 2)).ljust(11))
		print()


	def visualize_example(self, docid=None, input_type=None, data=None, pad_label=None, positive_label=None):
		if isinstance(data, tuple):
			y_pred_tokens = data[0]
			y_gold_tokens = data[1]
			x_tokens = data[2]
			if isinstance(y_gold_tokens, list):
				try:
					mask_index = y_gold_tokens.index(pad_label)
				except:
					mask_index = len(y_gold_tokens)
			else:
				mask_index = np.where(y_gold_tokens == 0)[0]
				if mask_index.size:
					mask_index = mask_index[0]
				else:
					mask_index = len(y_gold_tokens)

			boundaries_pred = [i for i, token in enumerate(
				y_pred_tokens[:mask_index]) if token == positive_label]
			boundaries_gold = [i for i, token in enumerate(
				y_gold_tokens[:mask_index]) if token == positive_label]

			for i, (p, g) in enumerate(zip(y_pred_tokens, y_gold_tokens)):
				if g == positive_label and p == positive_label:
					print(colored(x_tokens[i], 'green'), end=' ')
				elif g == positive_label and p != positive_label:
					print(colored(x_tokens[i], 'yellow'), end=' ')
				elif g != positive_label and p == positive_label:
					print(colored(x_tokens[i], 'red'), end=' ')
				else:
					print(x_tokens[i], end=' ')
			print()
		elif isinstance(data, str):
			x = None
			for docid, doc in self.dataset.val.items():
				if data in doc[input_type]:
					x = doc[input_type]
					break
			assert x, 'Document not available'

			x = re.sub(r'(\s){2,}', r'\1', x).lower()

			y_pred = [sent.text for sent in list(self.default_model(x).sents)]
			y_gold = doc['gold']

			y_gold_tokens, y_pred_tokens = Evaluation.tokenize_and_align(y_gold, y_pred)
			boundaries_gold = [i for i, token in enumerate(
				y_gold_tokens) if token[2]]
			# If token is not a deleted token (from gold), then allow it to qualify as a boundary
			boundaries_pred = [i for i, token in enumerate(
				y_pred_tokens) if token[2] and y_gold_tokens[i][3] != 'd']

			print()
			for p, g in zip(y_pred_tokens, y_gold_tokens):
				if g[2] and p[2] and g[3] != 'd':
					print(colored(p[1], 'green'), end=' ')
				elif g[2] and not (p[2] and g[3] != 'd'):
					print(colored(p[1], 'yellow'), end=' ')
				elif not g[2] and p[2] and g[3] != 'd':
					print(colored(p[1], 'red'), end=' ')
				else:
					if g[3] != 'd':
						print(p[1], end=' ')
			print()

		metadata = {'y_pred_tokens': y_pred_tokens,
                    'y_gold_tokens': y_gold_tokens,
                    'boundaries_pred': boundaries_pred,
                    'boundaries_gold': boundaries_gold}

		metadata.update(Evaluation.sensitivity_specificity(
			boundaries_pred, boundaries_gold))
		self.output_inspection[docid] = metadata

		tp = len(metadata['tp'])
		fn = len(metadata['fn'])
		fp = len(metadata['fp'])

		precision = tp / (tp + fp) if tp or fp else 0
		recall = tp / (tp + fn) if tp or fn else 0
		fscore = 2 * precision * \
			recall / (precision + recall) if precision or recall else 0

		print(f"\n{docid} | {input_type} | f1-score: {fscore}\n")

if __name__ == "__main__":
	with open('/local/ccayral/orthonormalDA/data/coda-corpus/beirut_src.txt') as f_src, \
		open('/local/ccayral/orthonormalDA/data/coda-corpus/beirut_tgt.txt') as f_tgt:
		src = f_src.readlines()
		tgt = f_tgt.readlines()
		system = ['']*len(src)
		system[0] = ' أنا بعطيك رقمه تلفونه و عنوانو .'
	evaluation = Evaluation(already_split=False)
	evaluation.spelling_correction_eval(src, tgt, system)
