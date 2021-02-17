import tensorflow as tf
import tensorflow_addons as tfa

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import re
import numpy as np
import os
import time
import pickle
import argparse
from random import randint
from collections import Counter

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
            self.bi_layer = tf.keras.layers.Bidirectional(
                layer=self.lstm_layer, merge_mode='sum')

        def call(self, x, hidden):
            x = self.embedding(x)
            output, h, c = self.bi_layer(x, initial_state=hidden)
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

        self.encoder.build((args.batch_size, max_length_input))
        self.decoder.build(30)
        self.decoder.summary()
        
        self.steps_per_epoch_train = dataset_len[0] // args.batch_size
        self.steps_per_epoch_dev = dataset_len[1] // args.batch_size
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
            # progbar = tf.keras.utils.Progbar(self.steps_per_epoch)
            print(f'\nEpoch {epoch+1}/{args.epochs}')
            for i, batch in enumerate(dataset.tf_data.take(self.steps_per_epoch_train)):
                source, target = batch[0], batch[1]
                batch_loss = self.train_step(source, target, enc_hidden)
                total_loss += batch_loss
                iteration = int(self.optimizer.iterations)
                # progbar.update(i + 1)
                if iteration % 100 == 0:
                    rand_word = randint(0, args.batch_size - 1)
                    predictions = self.predict_batch(source[rand_word:rand_word+1])
                    raw = "".join(dataset.data[dataset.SOURCE].chars_map[i]
                                for i in source.numpy()[rand_word] if i)[5:-5]
                    gold_coda = "".join(
                        dataset.data[dataset.TARGET].chars_map[i] for i in target.numpy()[rand_word] if i)[5:-5]
                    system_coda = "".join(dataset.data[dataset.TARGET].chars_map[i]
                                        for i in predictions[0] if i != GumarDataset.Factor.EOW)
                    status = "{}{}{}".format(f'<r>{raw}<r>'.rjust(25), f'<g>{gold_coda}<g>'.rjust(25), f'<s>{system_coda}<s>'.rjust(25))
                    print(f'Batch {i+1}/{self.steps_per_epoch_train}\t| Loss {batch_loss.numpy():.4f} {status}')
            metrics = self.evaluate(gumar.dev, 'dev', args)
            print(f"Train. Loss {total_loss / self.steps_per_epoch_train:.4f} | Val. Accuracy: {metrics['coda_accuracy']:.4f}")


    def predict_batch(self, sentences):
        batch_size = sentences.shape[0]
        enc_start_state = [tf.zeros((batch_size, self.encoder.enc_units)),
                           tf.zeros((batch_size, self.encoder.enc_units))]
        enc_out, enc_h, enc_c = self.encoder(sentences, enc_start_state)

        start_tokens = tf.fill([batch_size], GumarDataset.Factor.BOS)
        end_token = GumarDataset.Factor.EOS

        greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()
        # Instantiate BasicDecoder object
        decoder_instance = tfa.seq2seq.BasicDecoder(
            cell=self.decoder.rnn_cell, sampler=greedy_sampler, output_layer=self.decoder.fc)
        # Setup Memory in decoder stack
        self.decoder.attention_mechanism.setup_memory(enc_out)

        decoder_initial_state = self.decoder.build_initial_state(
            batch_size, [enc_h, enc_c], tf.float32)
        ### Since the BasicDecoder wraps around Decoder's rnn cell only, you have to ensure that the inputs to BasicDecoder
        ### decoding step is output of embedding layer. tfa.seq2seq.GreedyEmbeddingSampler() takes care of this.
        ### You only need to get the weights of embedding layer and pass this callable to BasicDecoder's call() function
        decoder_embedding_matrix = self.decoder.embedding.variables[0]
        outputs, _, _ = decoder_instance(decoder_embedding_matrix, start_tokens=start_tokens,
                                        end_token=end_token, initial_state=decoder_initial_state)
        return outputs.sample_id.numpy()

    def evaluate(self, dataset, dataset_name, args):
        correct_coda_forms, total_coda_forms = 0, 0
        for i, batch in enumerate(dataset.tf_data.take(self.steps_per_epoch_dev)):
            source, target = batch[0], batch[1].numpy()[:, 1:]
            target[target == 3] = 0
            predictions = self.predict_batch(source)
            # Compute whole coda accuracy
            resized_predictions = np.concatenate(
                [predictions, np.zeros_like(target)], axis=1)[:, :target.shape[1]]
            resized_predictions[resized_predictions == 3] = 0
            total_coda_forms += target.shape[0]
            correct_coda_forms += np.sum(np.all(target == resized_predictions * (
                target != GumarDataset.Factor.PAD), axis=1))

        metrics = {"coda_accuracy": correct_coda_forms / total_coda_forms}
        return metrics


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

    if os.path.exists('data/gumar_char'):
        with open('data/gumar_char', 'rb') as g:
            gumar = pickle.load(g)
            print("Length of dataset before capping sentence length:", len(
                gumar.train.data[gumar.Dataset.SOURCE].sentences_chars_ids))
            temp = []
            for i in range(len(gumar.train.data[0].sentences_chars_ids)):
                if len(gumar.train.data[0].sentences_chars_ids[i]) < 30 and \
                    len(gumar.train.data[1].sentences_chars_ids[i]) < 30:
                    temp.append(
                        (gumar.train.data[0].sentences_chars_ids[i], gumar.train.data[1].sentences_chars_ids[i]))
            gumar.train.data[0].sentences_chars_ids = [c[0] for c in temp]
            gumar.train.data[1].sentences_chars_ids = [c[1] for c in temp]
            print("Length of dataset after capping sentence length to 30:", len(
                gumar.train.data[gumar.Dataset.SOURCE].sentences_chars_ids))
            token_pairs_fl = Counter()
            words_src, words_tgt = [], []
            src_sentences_chars = [''.join(map(lambda x: gumar.train.data[gumar.train.SOURCE].chars_map[x], word))
                               for word in gumar.train.data[0].sentences_chars_ids]
            tgt_sentences_chars = [''.join(map(lambda x: gumar.train.data[gumar.train.TARGET].chars_map[x], word))
                                   for word in gumar.train.data[1].sentences_chars_ids]
            for i, token_pair in enumerate(zip(src_sentences_chars, tgt_sentences_chars)):
                if token_pair[0] != token_pair[1]:
                    words_src.append(
                        gumar.train.data[0].sentences_chars_ids[i])
                    words_tgt.append(
                        gumar.train.data[1].sentences_chars_ids[i])
                token_pairs_fl.update([token_pair])
            gumar.train._data[0].sentences_chars_ids = words_src
            gumar.train._data[1].sentences_chars_ids = words_tgt
            print(len(
                gumar.train.data[gumar.Dataset.SOURCE].sentences_chars_ids))

    else:
        gumar = GumarDataset('annotated-gumar-corpus', add_bow_eow=True, max_sentence_len=20)
        with open('data/gumar', 'wb') as g:
            pickle.dump(gumar, g)

    for dataset in [gumar.train, gumar.dev]:
        for f in range(GumarDataset.Dataset.FACTORS):
            dataset.data[f].sentences_chars_ids = tf.keras.preprocessing.sequence.pad_sequences(
                dataset.data[f].sentences_chars_ids, padding='post')

    train_src = gumar.train.data[GumarDataset.Dataset.SOURCE].sentences_chars_ids
    train_tgt = gumar.train.data[GumarDataset.Dataset.TARGET].sentences_chars_ids
    dev_src = gumar.dev.data[GumarDataset.Dataset.SOURCE].sentences_chars_ids[:1000]
    dev_tgt = gumar.dev.data[GumarDataset.Dataset.TARGET].sentences_chars_ids[:1000]
    gumar.train.tf_data = tf.data.Dataset.from_tensor_slices((train_src, train_tgt)).shuffle(
        args.buffer_size).batch(args.batch_size, drop_remainder=True)
    gumar.dev.tf_data = tf.data.Dataset.from_tensor_slices((dev_src, dev_tgt)).shuffle(
        args.buffer_size).batch(args.batch_size, drop_remainder=True)

    network = Network(args,
                      dataset_len=(len(gumar.train.data[gumar.Dataset.SOURCE].sentences_chars_ids),
                                   len(gumar.dev.data[gumar.Dataset.SOURCE].sentences_chars_ids[:100])),
                      src_words_vocab_len=len(
                          gumar.train.data[gumar.train.SOURCE].chars_map),
                      tgt_words_vocab_len=len(
                          gumar.train.data[gumar.train.TARGET].chars_map),
                      max_length_input=max(
                          len(s) for s in gumar.train.data[gumar.Dataset.SOURCE].sentences_chars_ids),
                      max_length_output=max(
                          len(s) for s in gumar.train.data[gumar.Dataset.TARGET].sentences_chars_ids))

    for epoch in range(args.epochs):
        network.train_epoch(gumar.train, args)
