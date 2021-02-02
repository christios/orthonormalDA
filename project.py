#!/usr/bin/env python3

import re
import sys
from collections import defaultdict
from collections import Counter
import itertools
import pdb
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

source_filename, target_filename, alignment_filename = sys.argv[1:4]

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

def read_sentence(fh, delete_tree=False):
	sentence = list()
	for line in fh:
		if line == '\n':
			# end of sentence
			break
		elif line.startswith('#'):
			# ignore comments
			continue
		else:
			fields = line.strip().split('\t')
			if fields[ID].isdigit():
				# make IDs 0-based to match alignment IDs
				fields[ID] = int(fields[ID])-1
				fields[HEAD] = int(fields[HEAD])-1
				if delete_tree:
					# reasonable defaults:
					fields[HEAD] = -1	   # head = root
					fields[DEPREL] = 'dep'  # generic deprel
				sentence.append(fields)
			# else special token -- continue
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

def write_sentence(sentence):
	result = list()
	for fields in sentence:
		# switch back to 1-based IDs
		fields[ID] = str(fields[ID]+1)
		fields[HEAD] = str(fields[HEAD]+1)
		result.append('\t'.join(fields))
	result.append('')
	return '\n'.join(result)

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
	parser.add_argument("-t", "--test", help="Test mode", action="store_true")
	parser.add_argument("source_filename", help="Source language file name")
	parser.add_argument("target_filename", help="Target language file name")
	parser.add_argument("alignment_filename", help="Alignments file")
	args = parser.parse_args()

	SENTENCES = 1000

	src2tgt_fl = {}
	with open(source_filename) as source, open(target_filename) as target, open(alignment_filename) as alignment_f:
		for sentence_id in range(SENTENCES):
			(src2tgt, tgt2src) = read_alignment(alignment_f)
			source_sentence = read_sentence(source)
			target_sentence = read_sentence(target, delete_tree=True)
			for source_token in source_sentence:
				source_token_id = source_token[ID]
				target_tokens = src2tgt_fl.setdefault(source_token[LEMMA].lower(), [])
				for target_token_id in src2tgt[source_token_id]:
					target_tokens.append(target_sentence[target_token_id][FORM].lower())
		for src, tgts in src2tgt_fl.items():
			src2tgt_fl[src] = Counter(tgts).most_common()

	with open(source_filename) as source, open(target_filename) as target, open(alignment_filename) as alignment_f, open('out.conllu', 'w') as out_file:
		many_to_one_stat, many_to_one_stat_total = 0, 0
		one_to_one_stat, unaligned_stat = 0, 0
		one_to_many_stat, many_to_many_stat = 0, 0
		total_num_source_tokens = 0
		intra_many_to_one = 0
		
		for sentence_id in range(SENTENCES):
			(src2tgt, tgt2src) = read_alignment(alignment_f)
			source_sentence = read_sentence(source)
			target_sentence = read_sentence(target, delete_tree=True)
			alignment_data = store_alignment(source_sentence, target_sentence, src2tgt)
			# if args.test: display_alignment(alignment_data)
			
			alignment = alignment_data['alignment']
			max_num_assignments = alignment_data['max_num_assignments']
			alignments_target_ids = [token[0] for token in list(itertools.chain.from_iterable(alignment[1:]))]
	
			one_to_one, unaligned, one_to_many, many_to_one, many_to_many = process_alignment_types(src2tgt, tgt2src, alignment_data)

			# Unaligned and one-to-one
			for align in unaligned:
				alignment[0][align[0]] = (align[0], re.sub(r'<MS:(.*)>', r'\1', alignment[0][align[0]][1]), {'head': source_sentence[align[0]][HEAD], 'deprel': source_sentence[align[0]][DEPREL]})
				alignment[1][align[0]] = ('<EMPTY>', '<EMPTY>', {'head': 0, 'deprel': 'dep'})
				for i in range(2, max_num_assignments + 1):
					alignment[i][align[0]] = ('<U>', '<U>')

			for align in one_to_one + unaligned:
				alignment[1][align[0]][2]['deprel'] = source_sentence[align[0]][DEPREL]
			for deprel_pair in itertools.permutations(one_to_one + unaligned, 2):
				if source_sentence[deprel_pair[0][0]][HEAD] == deprel_pair[1][0]:
					alignment[1][align[0]][2]['head'] = deprel_pair[1][1]
			
			# One-to-many
			for align in one_to_many:
				# PH: placeholder	
				one_to_many_true = True		
				for token_id in align[1]:
					if alignments_target_ids.count(token_id) != 1:
						one_to_many_true = False
						break
				if one_to_many_true:
					alignment[1][align[0]] = ('<PH>', '<PH>', {'head': 0, 'deprel': 'dep'}, align[1])
					for i in range(2, max_num_assignments + 1):
						alignment[i][align[0]] = ('<U>', '<U>')
			
			num_source_tokens = [one_to_one, unaligned, one_to_many, list(itertools.chain.from_iterable(many_to_many)), list(itertools.chain.from_iterable([[None]*len(many[0]) for many in many_to_one]))]
			many_to_many_len = len(list(itertools.chain.from_iterable(many_to_many)))
			one_to_one_len = len(one_to_one)
			many_to_one_len = len(list(itertools.chain.from_iterable([[None]*len(many[0]) for many in many_to_one])))
			unaligned_len = len(unaligned)
			one_to_many_len = len(one_to_many)
			
			if args.test and sentence_id < 20:
				print("="*int(os.popen('stty size', 'r').read().split()[1]), '\n')
				print('{:.0f}% of source tokens are unaligned'.format(unaligned_len/len(alignment[0])*100))
				print('{:.0f}% of source tokens are one-to-one'.format(one_to_one_len/len(alignment[0])*100))
				print('{:.0f}% of source tokens are one-to-many'.format(one_to_many_len/len(alignment[0])*100))
				print('{:.0f}% of source tokens are many-to-one'.format(many_to_one_len/len(alignment[0])*100))
				print('{:.0f}% of source tokens are many-to-many'.format(many_to_many_len/len(alignment[0])*100))
				display_alignment(alignment_data)
				print("Unaligned source IDs:", unaligned, sep='\n')
				print("One-to-one source IDs:", one_to_one, sep='\n')
				print("One-to-many source IDs:", one_to_many, sep='\n')
				print("Many-to-one source IDs:", many_to_one, sep='\n')
				print("Many-to-many source IDs:", many_to_many, '', sep='\n')
				print("="*int(os.popen('stty size', 'r').read().split()[1]))

			many_to_one_stat += sum(len(many[0]) for many in many_to_one)
			unaligned_stat += len(unaligned)
			one_to_one_stat += len(one_to_one)
			one_to_many_stat += len(one_to_many)
			many_to_many_stat += many_to_many_len
			many_to_one_stat_total += len(many_to_one)
			total_num_source_tokens += len(source_sentence)
			
			# Many-to-one
			# for align in many_to_one:
			# 	heads = [source_sentence[source_sentence[token_id][HEAD]][FORM] for token_id in align[0]]
			# 	many_to_one_tokens = [alignment[0][al][0] for al in align[0]]

			# 	if args.test: print([alignment[0][al] for al in align[0]], heads)
			# if args.test: print('\n')
			# display_alignment(alignment_data)

			# alignments_target_ids_temp = [token[0] for token in list(itertools.chain.from_iterable(alignment[1:]))]		
			# if args.test: print(alignments_target_ids_temp)
			# alignments_target_ids_temp = set(alignments_target_ids_temp)
			# if args.test: print(alignments_target_ids_temp)

			for align in one_to_one:
				target_sentence[align[1]][DEPREL] = source_sentence[align[0]][DEPREL]	# 30% deprel
			for deprel_pair in itertools.permutations(one_to_one, 2):
				if source_sentence[deprel_pair[0][0]][HEAD] == deprel_pair[1][0]:
					target_sentence[deprel_pair[0][1]][HEAD] = deprel_pair[1][1]		# 18% head

			for align in one_to_many:													# +6% deprel
				max_tgt_form_score, target_id_max = 0, 0
				src_lemma = source_sentence[align[0]][LEMMA].lower()
				tgt_types = [token_type[0] for token_type in src2tgt_fl[src_lemma]]
				for target_id in align[1]:
					tgt_form = target_sentence[target_id][FORM].lower()
					tgt_score = src2tgt_fl[src_lemma][tgt_types.index(tgt_form)][1]
					if tgt_score > max_tgt_form_score:
						max_tgt_form_score = tgt_score
						target_id_max = target_id
				target_sentence[target_id_max][DEPREL] = source_sentence[align[0]][DEPREL]
			
			for align in many_to_one:
				for source_id in align[0]:
					if source_sentence[source_id][HEAD] in align[0]:
						intra_many_to_one += 1
						break
			
			# for deprel_pair in itertools.permutations(one_to_one, 2):
			# 	if source_sentence[deprel_pair[0][0]][HEAD] == deprel_pair[1][0]:
			# 		target_sentence[deprel_pair[0][1]][HEAD] = deprel_pair[1][1]

					# TODO you should also make sure not to produce cycles
			if not args.test: print(write_sentence(target_sentence), file=out_file)

		print("-----------------")
		print("Global statistics")
		print("-----------------")
		print("There are on average {:.1f} source tokens aligned to one target token in the many-to-one relation".format(
                    many_to_one_stat/many_to_one_stat_total))
		print('{:.0f}% of source tokens are unaligned'.format(
                    unaligned_stat/total_num_source_tokens*100))
		print('{:.0f}% of source tokens are one-to-one'.format(one_to_one_stat /
                                                         total_num_source_tokens*100))
		print('{:.0f}% of source tokens are one-to-many'.format(one_to_many_stat /
                                                          total_num_source_tokens*100))
		print('{:.0f}% of source tokens are many-to-one'.format(many_to_one_stat /
                                                          total_num_source_tokens*100))
		print('{:.0f}% of source tokens are many-to-many'.format(many_to_many_stat /
                                                           total_num_source_tokens*100))
		print('{:.0f}% of many-to-one alignments are intra'.format(
                    intra_many_to_one / many_to_one_stat_total*100))
		if not args.test:
			print('\nDEPREL accuracy:')
			os.system('python3 evaluator.py -j -m deprel fr_pud-ud-test.conllu out.conllu')
			print('\nHEAD accuracy:')
			os.system('python3 evaluator.py -j -m head fr_pud-ud-test.conllu out.conllu')
