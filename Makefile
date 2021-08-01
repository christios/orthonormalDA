train:
	python3 spell_correct/train_pos.py --mode standardizer --batch_size 4 --epochs 35 --lr 0.001
	python3 spell_correct/train_pos.py --mode standardizer --batch_size 4 --epochs 35 --lr 0.001 --use_bert_enc init
	python3 spell_correct/train_pos.py --mode tagger --batch_size 4 --epochs 35 --lr 0.001 --use_bert_enc init
	python3 spell_correct/train_pos.py --mode tagger --batch_size 4 --epochs 35 --lr 0.001
train_joint:
	python3 spell_correct/train_pos.py --mode joint --batch_size 4 --epochs 40 --lr 0.0005 --use_bert_enc init