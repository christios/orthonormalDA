#!/usr/bin/env python3

import re
from collections import defaultdict
from collections import Counter
import itertools
import os
import argparse

ID = 0
FORM = 1
LEMMA = 2
UPOS = 3
XPOS = 4
FEATS = 5
HEAD = 6
DEPREL = 7

source_filename, target_filename, alignment_filename = 'alignments/gumar.src', 'alignments/gumar.tgt', 'alignments/gumar.f'#sys.argv[1:4]

"""
Source Language: Italian
Target Language: French
Alignment Type: Union

If you add the -t flag to the python command, then alignments and statistics will
be displayed. Else, the sentences are processed and printed for evaluation. By default,
10 sentences are printed for testing.

Use the following command for test (and without -t for evaluation):
	python3 project.py it_pud-ud-test.conllu fr_pud-ud-test.conllu it-fr.union -t
"""

def read_sentence(fh):
	sentence = list()
	
	for i, word in enumerate(fh.readline().split()):
		fields = ['_']*10
		fields[0] = i
		fields[1] = word
		sentence.append(fields)
	return sentence

def read_alignment(fh):
	line = fh.readline()
	src2tgt = defaultdict(list)
	tgt2src = defaultdict(list)
	for st in line.split():
		(src, tgt) = st.split('-')
		src = int(src)
		tgt = int(tgt)
		src2tgt[src].append(tgt)
		tgt2src[tgt].append(src)
	return (src2tgt, tgt2src)



def store_alignment(source_sentence, target_sentence, src2tgt):

	tokens_source = [token[FORM] for token in source_sentence]
	tokens_target = [token[FORM] for token in target_sentence]
	max_num_assignments = len(max(src2tgt.values(), key=len))
	alignment = [[] for _ in range(max_num_assignments + 1)]

	# Create list of alignments
	for i in range(max_num_assignments + 1):
		for src, tgt in src2tgt.items():
			if i == 0:
				alignment[i].append((src, tokens_source[src], {'head': source_sentence[src][HEAD], 'deprel': source_sentence[src][DEPREL]}))
			elif i > 0 and len(tgt) > i - 1:
				alignment[i].append((tgt[i - 1], tokens_target[tgt[i - 1]], {'head': 0, 'deprel': 'dep'}))
			else:
				alignment[i].append(('<U>', '<U>'))

	# Missing source tokens
	missing_source_tokens = 0
	for i, token in enumerate(tokens_source):
		if i != len(tokens_source) - 1 and token != alignment[0][i][1]:
			alignment[0].insert(i, (-1, '<MS:{}>'.format(token)))
			missing_source_tokens += 1
			for j in range(1, max_num_assignments + 1):
				alignment[j].insert(i, (-1, '<MS:{}>'.format(token)))
		elif i == len(tokens_source) - 1 and len(tokens_source) != len(alignment[0]):
			alignment[0].append((-1, '<MS:{}>'.format(token)))
			missing_source_tokens += 1
			for j in range(1, max_num_assignments + 1):
				alignment[j].append((-1, '<MS:{}>'.format(token)))

	# Missing target tokens
	missing_tokens_target = [i for i in range(len(tokens_target)) if i not in list(itertools.chain.from_iterable(src2tgt.values()))]
	
	alignment_data = {'alignment': alignment, 'tokens_source': tokens_source, 'tokens_target': tokens_target,
				'max_num_assignments': max_num_assignments, 'missing_source_tokens': missing_source_tokens,
				'missing_tokens_target': missing_tokens_target}
	
	return alignment_data

def display_alignment(alignment_data):

	alignment = [al.copy() for al in alignment_data['alignment']]
	tokens_source = alignment_data['tokens_source']
	tokens_target = alignment_data['tokens_target']
	max_num_assignments = alignment_data['max_num_assignments']
	missing_source_tokens = alignment_data['missing_source_tokens']
	missing_tokens_target = alignment_data['missing_tokens_target']
	rows, columns = os.popen('stty size', 'r').read().split()
	column_width = max(len(max(tokens_source, key=len)) + 1, len(max(tokens_target, key=len)) + 1) + 5
	display_width = int(columns) // (column_width + 2)

	for i, al in enumerate(alignment):
		if i == 0:
			for j, token in enumerate(al):
				if token[0] != -1:
					alignment[i][j] = token[1].center(column_width, '-')
				else:
					alignment[i][j] = token[1].center(column_width)
		else:
			alignment[i] = [token[1].center(column_width) for token in al]

	# Alignment statistics
	print('{} out of {} source tokens not aligned to any target token'.format(missing_source_tokens, len(tokens_source)))
	if missing_tokens_target:
		print('{} out of {} target tokens are missing (in order):'.format(len(missing_tokens_target), len(tokens_target)))
		print(' | '.join([token for i, token in enumerate(tokens_target) if i in missing_tokens_target]))
	else:
		print('{} out of {} target tokens are missing'.format(len(missing_tokens_target), len(tokens_target)))
	
	for i, al in enumerate(alignment):
		alignment[i] = [token + '|' for token in al]

	# Print alignments graphically in a way that fits the screen
	offset = 0
	print('_'*(column_width + 1)*display_width + '_')
	while len(alignment[0]) - offset > display_width:
		for i in range(max_num_assignments + 1):
			if i == 0:
				print('|' + re.sub(r' ', r'-', ''.join(alignment[i][offset : offset + display_width])))
			else:
				print('|' + ''.join(alignment[i][offset : offset + display_width]))
		offset += display_width
		print('_'*(column_width + 1)*display_width + '_')
	if len(alignment[0]) - offset:
		for i in range(max_num_assignments + 1):
			if i == 0:
				print('|' + re.sub(r' ', r'-', ''.join(alignment[i][offset : len(alignment[0])])))
			else:
				print('|' + ''.join(alignment[i][offset : len(alignment[0])]))
		print('_'*(column_width + 1)*(len(alignment[0]) - offset) + '_')

	print('')

def column(matrix, i):
    return [row[i] for row in matrix]

def process_alignment_types(src2tgt, tgt2src, alignment_data):

	alignment = alignment_data['alignment']
	max_num_assignments = alignment_data['max_num_assignments']
	alignments_target_ids = [token[0] for token in list(itertools.chain.from_iterable(alignment[1:]))]
	one_to_one, unaligned, one_to_many, many_to_one, many_to_many = [], [], [], [], []

	for i in range(len(alignment[0])):
		source_token = column(alignment, i)
		source_token_id = source_token[0][0]
		if source_token_id == -1:
			unaligned.append((i, 'F'+ str(i)))
		elif len(source_token) == 2:
			if alignments_target_ids.count(src2tgt[source_token_id][0]) == 1:
				one_to_one.append((i, src2tgt[source_token_id][0]))
			else:
				many_to_many.append([(i, [src2tgt[i][0]])])
		elif source_token[2][0] == '<U>':
			if alignments_target_ids.count(src2tgt[i][0]) == 1:
				one_to_one.append((i, src2tgt[i][0]))
			else:
				many_to_many.append([(i, [src2tgt[i][0]])])
		elif source_token[2][0] != '<U>':
			one_to_many.append((i, src2tgt[i]))

	alignments_tgt_ids = [token for token in list(itertools.chain.from_iterable(tgt2src.values()))]
	for tgt, src in tgt2src.items():
		if len(src) > 1 and not [one_of_many for one_of_many in src if alignments_tgt_ids.count(one_of_many) > 1]:
			many_to_one.append((src, tgt))

	# One-to-many and many-to-many
	for align in one_to_many:
		# PH: placeholder			
		for token_id in align[1]:
			if alignments_target_ids.count(token_id) != 1:
				many_to_many.append([align])
				break
	one_to_many = [one for one in one_to_many if [one] not in many_to_many]

	# Many-to-many
	target_token_ids = [align[0][1] for align in many_to_many]
	many_to_many_temp = [many_to_many[0]] if many_to_many else []
	for source_token in many_to_many[1:]:
		merges = []
		for i, align in enumerate(many_to_many_temp):
			if [True for al in align if set(source_token[0][1]).intersection(set(al[1]))]:
				merges.append(i)
		if not merges:
			many_to_many_temp.append(source_token)
		else:
			many_to_many_temp[merges[0]] += source_token
			for i in merges[1:]:
				many_to_many_temp[merges[0]] += many_to_many_temp[i]
				many_to_many_temp[i] = None
			many_to_many_temp = [many for many in many_to_many_temp if many is not None]

	many_to_many = [many for many in many_to_many_temp if [align for align in many if len(align[1]) > 1]]
	return one_to_one, unaligned, one_to_many, many_to_one, many_to_many

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("source_filename", help="Source language file name")
	parser.add_argument("target_filename", help="Target language file name")
	parser.add_argument("alignment_filename", help="Alignments file")
	args = parser.parse_args()

	SENTENCES = 1000

	with open(args.source_filename) as source, \
		 open(args.target_filename) as target, \
		 open(args.alignment_filename) as alignment_f, \
		 open('alignments/vilems.txt', 'w') as vilems:

		for f in zip(alignment_f, source, target):
			info = tuple(map(lambda x: x.strip('\n'), f))

			src_list = info[1].split()
			source_sentence = ' '.join(reversed(src_list))
			source_sentence = re.sub(r' ', r'%20', source_sentence)

			tgt_list = info[2].split()
			target_sentence = ' '.join(reversed(tgt_list))
			target_sentence = re.sub(r' ', r'%20', target_sentence)

			align_list = reversed(info[0].split())
			align_list = [a.split('-') for a in align_list]
			for a in align_list:
				a[0] = str(abs(int(a[0]) - len(src_list) + 1))
				a[1] = str(abs(int(a[1]) - len(tgt_list) + 1))
			align_list = ['-'.join(a) for a in align_list]
			alignment = ' '.join(align_list)
			alignment = re.sub(r' ', r'%20', alignment)
			
			# if len(tgt_list) != len(src_list):
			print(f"https://vilda.net/s/slowalign/?text1={source_sentence}&text2={target_sentence}&algn={alignment}", file=vilems)


			# alignment_data = store_alignment(source_sentence, target_sentence, src2tgt)
			# print()
			# display_alignment(alignment_data)
			# print()
