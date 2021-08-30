from transformers import BertTokenizerFast, EncoderDecoderModel

from torch.utils.data.dataloader import DataLoader
from transformers import default_data_collator
from torch.utils.data.sampler import SequentialSampler
import torch
import os
from seq2seq_trainer import Seq2SeqTrainer
from transformers import TrainingArguments

from hybrid_nmt.nmt_model import Bert2HyBERT
from hybrid_nmt.vocab import Vocab

model_name = 'UBC-NLP/MARBERT'
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
vocab = Vocab.load("/local/ccayral/orthonormalDA/data/hybrid/vocab.json")

tokenizer = BertTokenizerFast.from_pretrained(
    model_name, cache_dir='/local/ccayral/.transformer_models/MARBERT_pytorch_verison')

state_dict = torch.load(
    '/local/ccayral/orthonormalDA/bert2hybert/pytorch_model.bin')
model = Bert2HyBERT(char_embed_size=128,
                    hidden_size_char=256,
                    vocab=vocab,
                    no_char_decoder=False,
                    tokenizer=tokenizer,
                    device=device)
model.load_state_dict(state_dict)

# model.to(device)
# model.eval()

training_args = Seq2SeqTrainingArguments(
    remove_unused_columns=False,
    output_dir="./model",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size//2,
    gradient_accumulation_steps=2,
    predict_with_generate=True,
    do_eval=True,
    evaluation_strategy="epoch",
    do_train=True,
    logging_steps=500,
    save_steps=32965 // (batch_size * 2),
    warmup_steps=100,
    eval_steps=10,
    #max_steps=16, # delete for full training
    num_train_epochs=5,  # uncomment for full training
    overwrite_output_dir=True,
    save_total_limit=0,
    fp16=True,
)

# instantiate trainer
trainer = Seq2SeqTrainer(
    # config=config,
    model=bert2hybert,
    args=training_args,
    data_collator=gumar_collator,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=dev_data,
)

text = ["مرحبا كيف الحال"]
print(text)

inputs = tokenizer.encode_plus(text, return_tensors='pt')

outputs = model.generate(input_ids=inputs.input_ids.to(device),
                         attention_mask=inputs.attention_mask.to(device),
                         num_beams=10,
                         min_length=3,
                         top_p=0.9,
                         temperature=1,
                         length_penalty=2)

preds = tokenizer.batch_decode(outputs)
print(preds)

# def process_data_to_model_inputs(batch):                                                               
#     # Tokenizer will automatically set [BOS] <text> [EOS]                                               
#     inputs = tokenizer(batch["src"], padding="longest", truncation=True, max_length=encoder_max_length)
#     outputs = tokenizer(batch["tgt"], padding="longest", truncation=True, max_length=decoder_max_length)
                                                                                                        
#     batch["input_ids"] = inputs.input_ids                                                               
#     batch["attention_mask"] = inputs.attention_mask                                                     
#     batch["decoder_input_ids"] = outputs.input_ids                                                      
#     batch["labels"] = outputs.input_ids.copy()                                                          
#     # mask loss for padding                                                                             
#     batch["labels"] = [                                                                                 
#         [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
#     ]                     
#     batch["decoder_attention_mask"] = outputs.attention_mask                                                                              
                                                                                                         
#     return batch

# batch_size=16
# encoder_max_length=150
# decoder_max_length=150

# all_data = load_dataset("spelling_correction_data.py")
# train_data = all_data['train'].train_test_split(test_size=0.1,seed=42)['train']
# val_data = all_data['train'].train_test_split(test_size=0.1,seed=42)['test']
# dev_data = val_data.train_test_split(test_size=0.5,seed=42)['train']
# test_data = val_data.train_test_split(test_size=0.5,seed=42)['test']

# dev_data = dev_data.map(
#     process_data_to_model_inputs, 
#     batched=True, 
#     batch_size=batch_size, 
#     remove_columns=["src", "tgt"],
# )
# dev_data.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
# )

# test_data = test_data.map(
#     process_data_to_model_inputs, 
#     batched=True, 
#     batch_size=batch_size, 
#     remove_columns=["src", "tgt"],
# )
# test_data.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
# )

# dev_sampler = SequentialSampler(dev_data)

# dev_dataloader = DataLoader(
#     dev_data,
#     sampler=dev_sampler,
#     collate_fn=default_data_collator,
#     batch_size=batch_size
# )

# test_sampler = SequentialSampler(test_data)

# test_dataloader = DataLoader(
#     test_data,
#     sampler=test_sampler,
#     collate_fn=default_data_collator,
#     batch_size=batch_size
# )

# tl_loss = []
# for i , inputs in enumerate(tqdm(test_dataloader)):
#   for k, v in inputs.items():
#       inputs[k] = v.to(device)
  
#   with torch.no_grad():
#     loss = model(**inputs).loss.cpu().detach()
#   tl_loss.append(loss)

# torch.exp(torch.stack(tl_loss).mean()) # exponential of the average loss

# torch.exp(torch.stack(tl_loss)).mean() #average of per batch eponential of the loss

# torch.exp(torch.stack(tl_loss).mean()) # exponential of the average loss

# torch.exp(torch.stack(tl_loss)).mean() #average of per batch eponential of the loss

# torch.exp(torch.stack(tl_loss).mean()) # exponential of the average loss

# torch.exp(torch.stack(tl_loss)).mean() #average of per batch eponential of the loss

