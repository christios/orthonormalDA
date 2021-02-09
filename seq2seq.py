import tensorflow as tf
import tensorflow_addons as tfa

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import unicodedata
import re
import numpy as np
import os
import io
import time
import pickle
import argparse

from gumar_dataset import GumarDataset


class Network:
    class Encoder(tf.keras.Model):

        def __init__(self, vocab_size, we_dim, cle_dim, enc_units, batch_sz):
            super(Network.Encoder, self).__init__()
            self.batch_sz = batch_sz
            self.enc_units = enc_units
            self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                       output_dim=we_dim)

            self.lstm_layer = tf.keras.layers.LSTM(self.enc_units,
                                                   return_sequences=True,
                                                   return_state=True,
                                                   recurrent_initializer='glorot_uniform')

        def call(self, x, hidden):
            x = self.embedding(x)
            output, h, c = self.lstm_layer(x, initial_state=hidden)
            return output, h, c

        def initialize_hidden_state(self):
            return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))]

    class Decoder(tf.keras.Model):

        def __init__(self,
                     vocab_size,
                     we_dim,
                     cle_dim,
                     dec_units,
                     batch_sz,
                     max_length_input,
                     max_length_output,
                     attention_type='luong'):

            super(Network.Decoder, self).__init__()
            self.max_length_output = max_length_output
            self.max_length_input = max_length_input
            self.batch_sz = batch_sz
            self.dec_units = dec_units
            self.attention_type = attention_type

            self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                       output_dim=we_dim)
            # Final Dense layer on which softmax will be applied
            self.fc = tf.keras.layers.Dense(vocab_size)
            # Define the fundamental cell for decoder recurrent structure
            self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)
            self.sampler = tfa.seq2seq.sampler.TrainingSampler()
            # Create attention mechanism with memory = None
            self.attention_mechanism = self.build_attention_mechanism(dec_units=self.dec_units,
                                                                      memory=None,
                                                                      memory_sequence_length=self.batch_sz *
                                                                      [self.max_length_input],
                                                                      attention_type=self.attention_type)
            # Wrap attention mechanism with the fundamental rnn cell of decoder
            self.rnn_cell = self.build_rnn_cell(batch_sz)
            # Define the decoder with respect to fundamental rnn cell
            self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell,
                                                    sampler=self.sampler,
                                                    output_layer=self.fc)

        def build_rnn_cell(self, batch_sz):
            rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnn_cell,
                                                    self.attention_mechanism,
                                                    attention_layer_size=self.dec_units)
            return rnn_cell

        def build_attention_mechanism(self,
                                      dec_units,
                                      memory,
                                      memory_sequence_length,
                                      attention_type='luong'):
            """ type: Which sort of attention (Bahdanau, Luong)
                dec_units: final dimension of attention outputs 
                memory: encoder hidden states of shape (batch_size, max_length_input, enc_units)
                memory_sequence_length: 1d array of shape (batch_size) with every element set to
                                        max_length_input (for masking purpose)"""

            if(attention_type == 'bahdanau'):
                return tfa.seq2seq.BahdanauAttention(units=dec_units,
                                                     memory=memory,
                                                     memory_sequence_length=memory_sequence_length)
            else:
                return tfa.seq2seq.LuongAttention(units=dec_units,
                                                  memory=memory,
                                                  memory_sequence_length=memory_sequence_length)

        def build_initial_state(self,
                                batch_sz,
                                encoder_state,
                                Dtype):
            decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_sz,
                                                                    dtype=Dtype)
            decoder_initial_state = decoder_initial_state.clone(
                cell_state=encoder_state)
            return decoder_initial_state

        def call(self,
                 inputs,
                 initial_state):
            x = self.embedding(inputs)
            outputs, _, _ = self.decoder(x,
                                         initial_state=initial_state,
                                         sequence_length=self.batch_sz*[self.max_length_output-1])
            return outputs

    def __init__(self,
                 args,
                 dataset_len,
                 src_words_vocab_len,
                 tgt_words_vocab_len,
                 max_length_input,
                 max_length_output):

        self.encoder = Network.Encoder(vocab_size=src_words_vocab_len,
                                       we_dim=args.we_dim,
                                       cle_dim=args.cle_dim,
                                       enc_units=args.rnn_dim,
                                       batch_sz=args.batch_size)

        self.decoder = Network.Decoder(vocab_size=tgt_words_vocab_len,
                                       we_dim=args.we_dim,
                                       cle_dim=args.cle_dim,
                                       dec_units=args.rnn_dim,
                                       batch_sz=args.batch_size,
                                       max_length_input=max_length_input,
                                       max_length_output=max_length_output,
                                       attention_type='luong')

        self.steps_per_epoch = dataset_len // args.batch_size
        self.optimizer = tf.keras.optimizers.Adam()

    def loss_function(self, real, pred):
        # real shape = (BATCH_SIZE, max_length_output)
        # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        loss = cross_entropy(y_true=real, y_pred=pred)
        # output 0 for y=0 else output 1
        mask = tf.logical_not(tf.math.equal(real, 0))
        mask = tf.cast(mask, dtype=loss.dtype)
        loss = mask * loss
        loss = tf.reduce_mean(loss)
        return loss

    @tf.function
    def train_step(self, inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_h, enc_c = self.encoder(inp, enc_hidden)
            dec_input = targ[:, :-1]  # Ignore <end> token
            real = targ[:, 1:]         # ignore <start> token
            # Set the AttentionMechanism object with encoder_outputs
            self.decoder.attention_mechanism.setup_memory(enc_output)
            # Create AttentionWrapperState as initial_state for decoder
            decoder_initial_state = self.decoder.build_initial_state(
                args.batch_size, [enc_h, enc_c], tf.float32)
            pred = self.decoder(dec_input, decoder_initial_state)
            logits = pred.rnn_output
            loss = self.loss_function(real, logits)
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def train_epoch(self, dataset, args):
        for epoch in range(args.epochs):
            start = time.time()
            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0
            # print(enc_hidden[0].shape, enc_hidden[1].shape)
            progbar = tf.keras.utils.Progbar(self.steps_per_epoch)
            for i, batch in enumerate(dataset.tf_data.take(self.steps_per_epoch)):
                source, target = batch[0], batch[1]
                batch_loss = self.train_step(source, target, enc_hidden)
                total_loss += batch_loss
                iteration = int(self.optimizer.iterations)
                progbar.update(i + 1)
                # if iteration % 10 == 0:
                #     print('Epoch {} Step {} Loss {:.4f}'.format(epoch + 1,
                #                                                 iteration,
                #                                                 batch_loss.numpy()))

            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                total_loss / int(self.optimizer.iterations)))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16,
                        type=int, help="Batch size.")
    parser.add_argument("--buffer_size", default=32000,
                        type=int, help="tf.Dataset buffer size.")
    parser.add_argument("--cle_dim", default=64, type=int,
                        help="Character-level embeddings dimension.")
    parser.add_argument("--we_dim", default=64, type=int,
                        help="Word-level embeddings dimension.")
    parser.add_argument("--epochs", default=10, type=int,
                        help="Number of epochs.")
    parser.add_argument("--max_sentences", default=5000,
                        type=int, help="Maximum number of sentences to load.")
    parser.add_argument("--rnn_dim", default=64, type=int,
                        help="RNN cell dimension.")
    parser.add_argument("--cell", default='gru',
                        type=str, help="RNN cell type.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int,
                        help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False,
                        action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    if os.path.exists('data/gumar'):
        with open('data/gumar', 'rb') as g:
            gumar = pickle.load(g)
    else:
        gumar = GumarDataset('annotated-gumar-corpus')
        with open('data/gumar', 'wb') as g:
            pickle.dump(gumar, g)

    for f in range(GumarDataset.Dataset.FACTORS):
        gumar.train.data[f].sentences_words_ids = tf.keras.preprocessing.sequence.pad_sequences(
            gumar.train.data[f].sentences_words_ids, padding='post')

    train_src = gumar.train.data[GumarDataset.Dataset.SOURCE].sentences_words_ids
    train_tgt = gumar.train.data[GumarDataset.Dataset.TARGET].sentences_words_ids
    gumar.train.tf_data = tf.data.Dataset.from_tensor_slices(
        (train_src, train_tgt))
    gumar.train.tf_data = gumar.train.tf_data.shuffle(
        args.buffer_size).batch(args.batch_size, drop_remainder=True)

    network = Network(args,
                      dataset_len=gumar.train.size,
                      src_words_vocab_len=len(
                          gumar.train.data[gumar.train.SOURCE].words_map),
                      tgt_words_vocab_len=len(
                          gumar.train.data[gumar.train.TARGET].words_map),
                      max_length_input=max(
                          len(s) for s in gumar.train.data[gumar.Dataset.SOURCE].sentences_words_ids),
                      max_length_output=max(
                          len(s) for s in gumar.train.data[gumar.Dataset.TARGET].sentences_words_ids))

    for epoch in range(args.epochs):
        network.train_epoch(gumar.train, args)
