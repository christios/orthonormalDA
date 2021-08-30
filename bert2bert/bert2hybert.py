# -*- coding: utf-8 -*-

from transformers.file_utils import (
    cached_property,
    torch_required,
    logger
)
from transformers import EncoderDecoderModel
import sys
from datasets import load_dataset 
from sacrebleu import corpus_bleu
from transformers import BertTokenizerFast, EncoderDecoderModel
import torch

from seq2seq_trainer import Seq2SeqTrainer
from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Optional

from hybrid_nmt.vocab import Vocab
from hybrid_nmt.utils import pad_sents_char
from hybrid_nmt.nmt_model import Bert2HyBERT

from eval_metric import Evaluation

encoder_max_length = decoder_max_length = 50
model_name = 'UBC-NLP/MARBERT'

train = True

vocab = Vocab.load(
    "/local/ccayral/orthonormalDA/data/coda-corpus/beirut_vocab.json")

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

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
    inputs = tokenizer(batch["src"], padding="max_length", truncation=True, max_length=encoder_max_length)
    outputs = tokenizer(batch["tgt"], padding="max_length", truncation=True, max_length=decoder_max_length)
    
    char_ids = vocab.tgt.words2charindices([sent.split() for sent in batch["tgt"]])
    target_padded_chars = pad_sents_char(char_ids,
                                         vocab.tgt.char2id['<pad>'],
                                         max_sent_length=encoder_max_length,
                                         max_word_length=30)
                                                                                                        
    batch["input_ids"] = inputs.input_ids                                                               
    batch["attention_mask"] = inputs.attention_mask                                                     
    batch["decoder_input_ids"] = outputs.input_ids                                                      
    batch["labels"] = outputs.input_ids.copy()                                                          
    # mask loss for padding                                                                             
    batch["labels"] = [                                                                                 
        [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
    ]                     
    batch["decoder_attention_mask"] = outputs.attention_mask                                                                              
    batch["target_padded_chars"] = target_padded_chars                                                                                                     
    return batch


def gumar_collator(features):
    batch = {}
    for k in ['attention_mask', 'decoder_attention_mask',
                'decoder_input_ids', 'input_ids', 'labels']:
        batch[k] = torch.stack([f[k] for f in features])
    batch['target_padded_chars'] = torch.tensor(
        torch.stack([torch.stack(f['target_padded_chars']) for f in features]), dtype=torch.long).permute(1, 0, 2)

    return batch



batch_size = 8

train_data = train_data_raw.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["src", "tgt"],
)
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels", "target_padded_chars"],
)

dev_data = dev_data_raw.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["src", "tgt"],
)
dev_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels", "target_padded_chars"],
)

test_data = test_data_raw.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["src", "tgt"],
)
test_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels", "target_padded_chars"],
)

bert2hybert = Bert2HyBERT(char_embed_size=128,
                          hidden_size_char=256,
                          decoder_max_len=decoder_max_length,
                          num_beams=1,
                          vocab=vocab,
                          no_char_decoder=True,
                          tokenizer=tokenizer,
                          device=device)
custom_config = {}
if not train:
    state_dict = torch.load(
        '/local/ccayral/orthonormalDA/bert2hybert/pytorch_model.bin')
    bert2hybert.load_state_dict(state_dict)
    custom_config = {
        "output_hidden_states": True,
        "return_dict_in_generate": True
    }

@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    label_smoothing: Optional[float] = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (if not zero)."}
    )
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to SortishSamler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    lr_scheduler: Optional[str] = field(
        default="linear", metadata={"help": f"Which lr scheduler to use."}
    )

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

# set training arguments - these params are not really tuned, feel free to change
training_args = Seq2SeqTrainingArguments(
    learning_rate=5e-5,
    remove_unused_columns=False,
    output_dir="/local/ccayral/orthonormalDA/bert2hybert",
    save_total_limit=1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size//2,
    gradient_accumulation_steps = 2,
    predict_with_generate=True,
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
)


# instantiate trainer
trainer = Seq2SeqTrainer(
    custom_config=custom_config,
    model=bert2hybert,
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
