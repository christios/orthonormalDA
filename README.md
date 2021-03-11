# Gulf Arabic Spelling Standardizer

## Introduction
Because creating clean corpuses of Dialectal Arabic does not suffice, since the real-world data will most of the time come in an extremely noisy format, this project attempts to ease the amount of spontaneous orthography in raw data by making use of the generalization properties of neural networks on sequences. For that reason, a Sequence-to-Sequence (seq2seq) Recurrent Neural Network (RNN) will be used in a Neural Machine Translation (NMT) setting, in which we will be "translating" from our raw inputs (i.e., words) to the CODA* standard form. Seq2seq models have enjoyed great success in a variety of tasks involving sequence classification, and most notably, NMT.

## Model default parameters
The model is implemented using PyTorch (version 1.7.1) with the following parameters chosen after some experimentation: Adam optimization, dropout after the embedding layer with 0.5 probability, learning rate of 0.001 with learning rate decay (on plateau), source and target embedding size of 128, 1-layer encoder, and RNN hidden dimension of 256 for both encoder and decoder, Luong attention with attention dimension of 16, large batch size of 256 to benefit from GPU parallelism, and 26 epochs of training.

## Usage

### Training
    python3 spell_correct_cle_tf.py


### Predicting
A file containing the predictions of our test set (`predictions.txt`) is already avaiable in this folder. Tokens enclosed in \<r> are the source tokens, \<g> are the gold tokens, and \<s> are the system tokens (our predictions). To reproduce these results, either retrain the model:

    python3 spell_correct_cle_tf.py --predict

or load the pretrained model weights (`spelling_std.pt`) obtained by using the parameters mentioned in the above section:

    python3 spell_correct_cle_tf.py --predict --load spelling_std.pt


### Flags

    -h, --help              Help message.
    --batch_size            Number of elements per batch.
    
    --cle_dim               Character-level embeddings dimension (in our case a 
                            sequence is a word not a sentence).
    
    --epochs                Number of epochs.
    
    --rnn_dim               RNN hidden layer dimension (for both encoder and 
                            decoder).
    
    --attn_dim              Attention layer output dimension.
    
    --dropout               Dropout probability (after embedding) in both
                            encoder and decoder.
    
    --max_word_len          Maximum sequence (word) length to allow in the data.
    
    --max_num_pair_types    Maximum number of occurences per type of source-target 
                            pair in the data.
    
    --load                  Name of file to load model weights from without 
                            training.
    
    --save                  Name of file to save model weights to after training.
    
    --bigram                Augments dataset to bigrams as described in the 
                            documentation.
    
    --predict               After training or loading weights, outputs predictions 
                            based on the test set inputs.
    
    --seed                  Random seed (for reproducibility).