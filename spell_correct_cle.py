import tensorflow as tf
import tensorflow_addons as tfa
import re
import numpy as np
import os
import datetime
import argparse
import pickle
from collections import Counter
from random import randint

from gumar_dataset import GumarDataset

class Network:

    class SpellingCorrector(tf.keras.Model):
        def __init__(self, args, src_chars_vocab_len, tgt_chars_vocab_len):
            super().__init__()

            self.tgt_chars_vocab_len = tgt_chars_vocab_len
            
            self.source_embedding = tf.keras.layers.Embedding(
                input_dim=src_chars_vocab_len, output_dim=args.cle_dim, mask_zero=True)
            # units does not refer to the number of character to be inputted to the encoder
            # but to the hidden state size
            gru_cell = tf.keras.layers.GRU(
                units=args.rnn_dim, return_sequences=True)
            self.source_rnn = tf.keras.layers.Bidirectional(
                layer=gru_cell, merge_mode='sum')

            self.target_embedding = tf.keras.layers.Embedding(
                input_dim=tgt_chars_vocab_len, output_dim=args.cle_dim, mask_zero=False)
            # Use Cell because we will be generating them one by one
            self.target_rnn_cell = tf.keras.layers.GRUCell(units=args.rnn_dim)
            self.target_output_layer = tf.keras.layers.Dense(
                units=tgt_chars_vocab_len)

            self.attention_source_layer = tf.keras.layers.Dense(
                units=args.rnn_dim)
            self.attention_state_layer = tf.keras.layers.Dense(
                units=args.rnn_dim)
            self.attention_weight_layer = tf.keras.layers.Dense(units=1)


        class DecoderTraining(tfa.seq2seq.BaseDecoder):
            def __init__(self, spelling_corrector, *args, **kwargs):
                self.spelling_corrector = spelling_corrector
                super().__init__.__wrapped__(self, *args, **kwargs)

            @property
            def batch_size(self):
                return tf.shape(self.source_states)[0]

            @property
            def output_size(self):
                # `tgt_chars_vocab_len` is the number of logits per each output element
                return tf.TensorShape(self.spelling_corrector.tgt_chars_vocab_len)

            @property
            def output_dtype(self):
                # Type of the logits
                return tf.float32

            def with_attention(self, inputs, states):
                """Implementation of Bahdanau attention.
                `W1`, `W2`, and `V` are the attention weights which
                will be learned by the network. At each decoding step,
                we are adding the decoder hidden state to each encoder
                hidden state.
                - score = FC(tanh(FC(EO) + FC(H))) [with EO -> encoder output and
                    H -> decoder hidden state size (rnn_dim)]
                - attention weights = softmax(score, axis = 1)
                - context vector = sum(attention weights * EO, axis = 1)

                Args:
                    inputs (tf.Tensor): 
                    states (tf.Tensor): Query vector with shape == (batch_size, H)
                                        which is the hidden state vector of the previously decoded
                                        hidden state.

                Returns:
                    list: inputs to the encoder and the attention vector
                """
                V = self.spelling_corrector.attention_weight_layer
                # W1_hd shape == (batch_size, H) -> how much of the decoder state am I passing?
                W1_hd = self.spelling_corrector.attention_state_layer(states)
                # `self.source_states` is the vector of encoder hidden states
                # `self.source_states` shape == (batch_size, max_length, H)
                # W2_he shape == (batch_size, max_length, H) -> how much of each encoder state am I passing?
                W2_he = self.spelling_corrector.attention_source_layer(self.source_states)
                # Need to expand W1_hd shape to (batch_size, 1, H) for broadcasting to apply during addition
                # score shape before V == (batch_size, max_length, H)
                # score shape after V == (batch_size, max_length, 1) -> score for each input word/character
                score = V(tf.nn.tanh(tf.expand_dims(W1_hd, 1) + W2_he))
                # Softmax by default is applied on the last axis but here we want to apply it on the 1st axis,
                # since the shape of score is (batch_size, max_length, 1). max_length is the length
                # of our input. Since we are trying to assign a weight to each input, softmax should be applied
                # on that axis.
                # weights shape == (batch_size, max_length, 1) -> one weight for each input word/character
                weights = tf.nn.softmax(score, axis=1)
                # Multiply each word by its corresponding weight
                attention = weights * self.source_states
                # attention shape == (batch_size, H)
                attention = tf.reduce_sum(attention, axis=1)
                # This result is fed back into the currently decoded cell
                return tf.keras.layers.concatenate([inputs, attention])

            def initialize(self, layer_inputs, initial_state=None, mask=None):
                self.source_states, self.targets = layer_inputs
                finished = tf.fill([self.batch_size], False)
                # Decoder inputs
                inputs = self.spelling_corrector.target_embedding(
                    tf.fill([self.batch_size], GumarDataset.Factor.BOW))
                # Initialization: since no character has been decoded yet,
                # we use the first word/character of the encoder input as a fake
                # decoder hidden state. The idea is that it is most relevant for decoding
                # the first letter and contains all following characters via the backward RNN.
                states = self.source_states[:, 0, :]
                inputs = self.with_attention(inputs, states)
                return finished, inputs, states

            def step(self, time, inputs, states, training):
                """Teacher forcing mode, feeding in training gold manually."""
                # `states` is the hidden state of previous decoder cell
                outputs, [states] = self.spelling_corrector.target_rnn_cell(inputs, [states])
                # Overwrite `outputs` by passing them through the prediction layer
                outputs = self.spelling_corrector.target_output_layer(outputs)
                # self.targets[:, time] is the `time`-th char/word
                # self.targets[:, time] shape == (batch_size, 1, cle_dim)
                next_inputs = self.spelling_corrector.target_embedding(self.targets[:, time])
                # `finished` is True if `time`-th char from `self.targets` is EOW, False otherwise
                finished = (self.targets[:, time] == GumarDataset.Factor.EOW)
                # Pass `next_inputs` and `states` (hidden state of previous decoder cell) to the attention layer
                next_inputs = self.with_attention(next_inputs, states)
                return outputs, states, next_inputs, finished


        class DecoderPrediction(DecoderTraining):
            @property
            def output_size(self):
                # Describes a scalar element, because we are generating scalar predictions now.
                return tf.TensorShape([])

            @property
            def output_dtype(self):
                return tf.int32

            def initialize(self, layer_inputs, initial_state=None, mask=None):
                self.source_states = layer_inputs
                # Use the same initialization as in DecoderTraining.
                finished = tf.fill([self.batch_size], False)
                inputs = self.spelling_corrector.target_embedding(
                    tf.fill([self.batch_size], GumarDataset.Factor.BOW))
                states = self.source_states[:, 0, :]
                inputs = self.with_attention(inputs, states)
                return finished, inputs, states

            def step(self, time, inputs, states, training):
                """Autoregressive mode."""
                outputs, [states] = self.spelling_corrector.target_rnn_cell(inputs, [
                                                                    states])
                outputs = self.spelling_corrector.target_output_layer(outputs)
                # Choose best character/word (with highest value)
                # axis 0 is the batch level, and axis 1 is the logits level (of dimension self.tgt_chars_vocab_len)
                outputs = tf.argmax(outputs, axis=1, output_type=tf.int32)
                # Autoregressive generation
                next_inputs = self.spelling_corrector.target_embedding(outputs)
                finished = (outputs == GumarDataset.Factor.EOW)
                next_inputs = self.with_attention(next_inputs, states)
                return outputs, states, next_inputs, finished


        def call(self, inputs):
            # If `inputs` is a list of two elements, we are in the teacher forcing mode.
            # Otherwise, we run in autoregressive mode.
            if isinstance(inputs, list) and len(inputs) == 2:
                source_sentences_chars, target_sentences_chars = inputs
            else:
                source_sentences_chars, target_sentences_chars = inputs, None
            source_sentences_chars_shape = tf.shape(source_sentences_chars)
            
            # Encoder:
            # Get indices of valid words and reshape the `source_sentences_chars`
            # so that it is a list of valid sequences, instead of a
            # matrix of sequences, some of them padding ones. The encoder
            # just takes in words (not sentences), this is why we need to remove
            # paddings that acted as words to build the batch. Hence, each batch has a 
            # a different maximum sequence length.
            valid_words = tf.cast(
                tf.where(source_sentences_chars[:, :, 0] != 0), tf.int32)
            # shape after `tf.gather_nd` == (# words (i.e., excluding words which are fully padded), length of longest word)
            source_sentences_chars = tf.gather_nd(source_sentences_chars, valid_words)
            if target_sentences_chars is not None:
                target_sentences_chars = tf.gather_nd(target_sentences_chars, valid_words)

            source_embedding = self.source_embedding(source_sentences_chars)
            source_states = self.source_rnn(source_embedding)

            # Run the appropriate decoder
            if target_sentences_chars is not None:
                decoderTraining = self.DecoderTraining(self)
                output_layer, _, output_lens = decoderTraining(
                    [source_states, target_sentences_chars])
            else:
                # tf.shape(source_sentences_chars)[1] + 10 indicates that the longest prediction
                # must be at most 10 characters longer than the longest input.
                # Then run it on `source_states`, storing the first result
                # in `output_layer` and the third result in `output_lens`.
                decoderPrediction = self.DecoderPrediction(
                    self, maximum_iterations=tf.shape(source_sentences_chars)[1] + 10)
                output_layer, _, output_lens = decoderPrediction(source_states)

            # Reshape the output to the original matrix of words
            # and explicitly set mask for loss and metric computation.
            output_layer = tf.scatter_nd(valid_words, output_layer, tf.concat(
                [source_sentences_chars_shape[:2], tf.shape(output_layer)[1:]], axis=0))
            output_layer._keras_mask = tf.sequence_mask(tf.scatter_nd(
                valid_words, output_lens, source_sentences_chars_shape[:2]))
            return output_layer

    def __init__(self, args, src_chars_vocab_len, tgt_chars_vocab_len):
        self.exceptions = 0
        self.spelling_corrector = self.SpellingCorrector(
            args, src_chars_vocab_len, tgt_chars_vocab_len)

        self.spelling_corrector.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.SparseCategoricalAccuracy(
                name="character_accuracy")],
        )
        self.writer = tf.summary.create_file_writer(
            args.logdir, flush_millis=10 * 1000)

    def append_eow(self, sequences):
        """Append EOW character after end of every given sequence."""
        # Add one <pad> char to each word
        padded_sequences = np.pad(sequences, [[0, 0], [0, 0], [0, 1]])
        # padded_sequences != 0 is the same array with True wherever the cell is != 0 
        ends = np.logical_xor(padded_sequences != 0, np.pad(
            sequences, [[0, 0], [0, 0], [1, 0]], constant_values=1) != 0)
        # Add <eow> to each word before its padding starts
        padded_sequences[ends] = GumarDataset.Factor.EOW
        return padded_sequences

    def train_epoch(self, dataset, args):
        for batch in dataset.batches(args.batch_size):
            # Create `targets` by appending EOW after target words
            targets = self.append_eow(batch[dataset.TARGET].sentences_chars_ids)
            sources = batch[dataset.SOURCE].sentences_chars_ids
            metrics = self.spelling_corrector.train_on_batch(x=[sources, targets],
                                                             y=targets)
            # Generate the summaries each 10 steps
            iteration = int(self.spelling_corrector.optimizer.iterations)
            if iteration % 10 == 0:
                tf.summary.experimental.set_step(iteration)
                metrics = dict(zip(self.spelling_corrector.metrics_names, metrics))

                rand_sent = randint(0, args.batch_size)
                rand_word = randint(0, batch[dataset.SOURCE].sentences_chars_ids[rand_sent].shape[0])
                predictions = self.predict_batch(
                    batch[dataset.SOURCE].sentences_chars_ids[rand_sent]).numpy()
                
                raw = "".join(dataset.data[dataset.SOURCE].chars_map[i]
                               for i in batch[dataset.SOURCE].sentences_chars_ids[rand_sent, rand_word] if i)
                gold_coda = "".join(
                    dataset.data[dataset.TARGET].chars_map[i] for i in targets[rand_sent, rand_word] if i)
                system_coda = "".join(dataset.data[dataset.TARGET].chars_map[i]
                                       for i in predictions[rand_sent, rand_word] if i != GumarDataset.Factor.EOW)
                status = ", ".join([*["{}={:.4f}".format(name, value) for name, value in metrics.items()],
                                    "{}{}{}".format(raw.rjust(25), gold_coda.rjust(25), system_coda.rjust(25))])
                print("Step {}:".format(iteration), status)
                
                with self.writer.as_default():
                    for name, value in metrics.items():
                        tf.summary.scalar("train/{}".format(name), value)
                    tf.summary.text("train/prediction", status)


    @tf.function(experimental_relax_shapes=True)
    def predict_batch(self, sentences_chars):
        return self.spelling_corrector(sentences_chars)

    def evaluate(self, dataset, dataset_name, args):
        correct_coda_forms, total_coda_forms = 0, 0
        for batch in dataset.batches(args.batch_size):
            predictions = self.predict_batch(
                batch[dataset.SOURCE].sentences_chars_ids).numpy()

            # Compute whole coda accuracy
            targets = self.append_eow(batch[dataset.TARGET].sentences_chars_ids)
            # Make `predictions` the same shape as the `targets` by padding zeros
            resized_predictions = np.concatenate(
                [predictions, np.zeros_like(targets)], axis=2)[:, :, :targets.shape[2]]
            # shape: (batch_size, words) with True wherever the first character
            # of a word is not EOW
            valid_coda_forms = targets[:, :, 0] != GumarDataset.Factor.EOW
            total_coda_forms += np.sum(valid_coda_forms)
            correct_coda_forms += np.sum(valid_coda_forms * np.all(targets ==
                                                           resized_predictions * (targets != 0), axis=2))

        metrics = {"coda_accuracy": correct_coda_forms / total_coda_forms}
        with self.writer.as_default():
            tf.summary.experimental.set_step(
                self.spelling_corrector.optimizer.iterations)
            for name, value in metrics.items():
                tf.summary.scalar("{}/{}".format(dataset_name, name), value)

        return metrics

    def train(self, gumar, args):
        for epoch in range(args.epochs):
            self.train_epoch(gumar.train, args)
            metrics = self.evaluate(gumar.dev, "dev", args)
            print("Evaluation on {}, epoch {}: {}".format(
                "dev", epoch + 1, metrics))

    def predict(self, dataset, args):
        predictions = []
        for batch in dataset.batches(args.batch_size):
            prediction_batch = self.predict_batch(
                batch[dataset.SOURCE].sentences_chars_ids).numpy().tolist()
            for sentence in prediction_batch:
                predictions.append(sentence)
        return predictions


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16,
                        type=int, help="Batch size.")
    parser.add_argument("--cle_dim", default=64, type=int,
                        help="CLE embedding dimension.")
    parser.add_argument("--epochs", default=10, type=int,
                        help="Number of epochs.")
    parser.add_argument("--max_sentences", default=5000,
                        type=int, help="Maximum number of sentences to load.")
    parser.add_argument("--rnn_dim", default=64, type=int,
                        help="RNN cell dimension.")
    parser.add_argument("--cell", default='gru', type=str,
                        help="RNN cell type.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int,
                        help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False,
                        action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub(
            "(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    if os.path.exists('data/gumar'):
        with open('data/gumar', 'rb') as g:
            gumar = pickle.load(g)
    else:
        gumar = GumarDataset('annotated-gumar-corpus')
        with open('data/gumar', 'wb') as g:
            pickle.dump(gumar, g)

    print(gumar.train.get_token_num())
    token_pairs_fl = Counter()
    sentences_src, sentences_tgt = [], []
    for idx in range(gumar.train.size):
        sentences_src.append([])
        sentences_tgt.append([])
        for i, token_pair in enumerate(zip(gumar.train.data[0].sentences_words[idx], gumar.train.data[1].sentences_words[idx])):
            if token_pairs_fl[token_pair] < 100:
                sentences_src[-1].append(
                    gumar.train.data[0].sentences_chars_ids[idx][i])
                sentences_tgt[-1].append(
                    gumar.train.data[1].sentences_chars_ids[idx][i])
            token_pairs_fl.update([token_pair])
    gumar.train._data[0].sentences_chars_ids = sentences_src
    gumar.train._data[1].sentences_chars_ids = sentences_tgt
    print(gumar.train.get_token_num())

    # Create the network and train
    network = Network(args,
                      src_chars_vocab_len=len(
                          gumar.train.data[gumar.train.SOURCE].chars_map),
                      tgt_chars_vocab_len=len(gumar.train.data[gumar.train.TARGET].chars_map))
    
    for epoch in range(args.epochs):
        network.train_epoch(gumar.train, args)
        metrics = network.evaluate(gumar.dev, "dev", args)
        print("Evaluation on {}, epoch {}: {}".format("dev", epoch + 1, metrics))

    metrics = network.evaluate(gumar.test, "test", args)
    with open("spelling_corrector.out", "w") as out_file:
        print("{:.2f}".format(100 * metrics["coda_accuracy"]), file=out_file)
