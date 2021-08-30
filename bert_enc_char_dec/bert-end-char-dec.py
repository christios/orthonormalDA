# -*- coding: utf-8 -*-


from transformers.file_utils import (
    cached_property,
    torch_required,
    logger
)
import sys
from datasets import load_dataset 
from transformers import BertTokenizerFast
import torch

from transformers import TrainingArguments, Trainer
from dataclasses import dataclass, field
from typing import Optional

from alignment_handler.vocab import Vocab
from hybrid_nmt.utils import pad_sents_char
from hybrid_nmt.nmt_model import BertEncoderCharDecoder
from alignment_handler.utils import AlignmentHandler

from eval_metric import Evaluation

encoder_max_length = 35
max_word_length = 25
model_name = 'UBC-NLP/MARBERT'

train = True

vocab = Vocab.load(
    "/local/ccayral/orthonormalDA/data/coda-corpus/beirut_vocab.json")
alignment_handler = AlignmentHandler(already_split=False, n=3)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizerFast.from_pretrained(model_name, cache_dir='/local/ccayral/.transformer_models/MARBERT_pytorch_verison')

all_data = load_dataset("/local/ccayral/orthonormalDA/bert2bert/spelling_correction_data.py")
train_data_raw = all_data['train'].train_test_split(test_size=0.1,seed=42)['train']
val_data = all_data['train'].train_test_split(test_size=0.1,seed=42)['test']
dev_data_raw = val_data.train_test_split(test_size=0.5,seed=42)['train']
test_data_raw = val_data.train_test_split(test_size=0.5,seed=42)['test']

print("Length of train data",len(train_data_raw))
print("Length of dev data",len(dev_data_raw))
print("Length of test data",len(test_data_raw))

def process_data_to_model_inputs(batch):                                                               
    # Tokenizer will automatically set [BOS] <text> [EOS]                                               
    src, tgt = alignment_handler.merge_split_src_tgt(batch['src'], batch['tgt'])

    src_bert = tokenizer(src, padding="max_length", truncation=True, max_length=encoder_max_length)
    char_ids = vocab.tgt.words2charindices(tgt)
    target_padded_chars = pad_sents_char(char_ids,
                                         vocab.tgt.char2id['<pad>'],
                                         max_sent_length=encoder_max_length,
                                         max_word_length=max_word_length)
    char_ids = vocab.src.words2charindices([sent.split() for sent in src])
    source_padded_chars = pad_sents_char(char_ids,
                                         vocab.src.char2id['<pad>'],
                                         max_sent_length=encoder_max_length,
                                         max_word_length=max_word_length)
    

    batch["input_ids"] = src_bert.input_ids                                                               
    batch["attention_mask"] = src_bert.attention_mask                                                                                                                                  
    batch["source_padded_chars"] = source_padded_chars
    batch["target_padded_chars"] = target_padded_chars
    return batch

def gumar_collator(features):
    batch = {}
    for k in ['input_ids', 'attention_mask']:
        batch[k] = torch.stack([f[k] for f in features])
    batch['target_padded_chars'] = torch.tensor(
        torch.stack([torch.stack(f['target_padded_chars']) for f in features]), dtype=torch.long).permute(1, 0, 2)
    batch['source_padded_chars'] = torch.tensor(
        torch.stack([torch.stack(f['source_padded_chars']) for f in features]), dtype=torch.long).permute(1, 0, 2)

    return batch

batch_size = 16

train_data = train_data_raw.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["src", "tgt"],
)
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "target_padded_chars", "source_padded_chars"],
)

dev_data = dev_data_raw.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["src", "tgt"],
)
dev_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "target_padded_chars", "source_padded_chars"],
)

test_data = test_data_raw.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["src", "tgt"],
)
test_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "target_padded_chars", "source_padded_chars"],
)

bert_enc_char_dec = BertEncoderCharDecoder(char_embed_size=128,
                                            hidden_size_char=256,
                                            max_tgt_len=max_word_length,
                                            vocab=vocab,
                                            tokenizer=tokenizer,
                                            device=device)

if not train:
    state_dict = torch.load(
        '/local/ccayral/orthonormalDA/bert2hybert/pytorch_model.bin')
    bert_enc_char_dec.load_state_dict(state_dict)

@dataclass
class MyTrainingArguments(TrainingArguments):
    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        self._n_gpu = torch.cuda.device_count()
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return device


def compute_metrics(pred):
    evaluation = Evaluation(already_split=False)
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    e_accuracy, ne_accuracy = evaluation.spelling_correction_eval(dev_data_raw.to_dict()['src'], label_str, pred_str)
    return {"e_accuracy": round(e_accuracy*100, 1), "ne_accuracy": round(ne_accuracy*100, 1)}

training_args = MyTrainingArguments(
    learning_rate=1e-3,
    remove_unused_columns=False,
    output_dir="/local/ccayral/orthonormalDA/bert2hybert",
    save_total_limit=1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size//2,
    gradient_accumulation_steps = 2,
    do_eval=True,
    evaluation_strategy ="epoch",
    do_train=True,
    logging_steps=500,  
    save_steps= 32965 // ( batch_size * 2),  
    warmup_steps=100,
    eval_steps=10,
    # max_steps=16, # delete for full training
    num_train_epochs=10,# uncomment for full training
    overwrite_output_dir=True,
    fp16=True,
    # prediction_loss_only=True
)

trainer = Trainer(
    model=bert_enc_char_dec,
    args=training_args,
    data_collator=gumar_collator,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=dev_data,
)

if len(sys.argv) > 1 and sys.argv[1] == 'train' or train:
    trainer.train()
    if sys.argv[2] == 'save':
        trainer._save("./bert2hybert")
        tokenizer.save_pretrained("./bert2hybert")
elif len(sys.argv) > 1 and sys.argv[1] == 'test' or not train:
    outputs = trainer.predict(test_data)
    pass
