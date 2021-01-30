#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from morpho_dataset import MorphoDataset

class Network:
    class Lemmatizer(tf.keras.Model):
        def __init__(self, args, num_source_chars, num_target_chars):
            super().__init__()

            self.num_target_chars = num_target_chars
            # TODO(lemmatizer_noattn): Define
            # - `source_embedding` as a masked embedding layer of source chars into args.cle_dim dimensions
            self.source_embedding = tf.keras.layers.Embedding(
                input_dim=num_source_chars, output_dim=args.cle_dim, mask_zero=True)

            # TODO: Define
            # - `source_rnn` as a bidirectional GRU with args.rnn_dim units, returning _whole sequences_,
            #   summing opposite directions
            gru_cell = tf.keras.layers.GRU(units=args.rnn_dim, return_sequences=True)
            self.source_rnn = tf.keras.layers.Bidirectional(layer=gru_cell, merge_mode='sum')

            # TODO(lemmatizer_noattn): Define
            # - `target_embedding` as an unmasked embedding layer of target chars into args.cle_dim dimensions
            # - `target_rnn_cell` as a GRUCell with args.rnn_dim units
            # - `target_output_layer` as a Dense layer into `num_target_chars`
            self.target_embedding = tf.keras.layers.Embedding(
                input_dim=num_target_chars, output_dim=args.cle_dim, mask_zero=False)
            self.target_rnn_cell = tf.keras.layers.GRUCell(units=args.rnn_dim)
            self.target_output_layer = tf.keras.layers.Dense(units=num_target_chars)

            # TODO: Define
            # - `attention_source_layer` as a Dense layer with args.rnn_dim outputs
            # - `attention_state_layer` as a Dense layer with args.rnn_dim outputs
            # - `attention_weight_layer` as a Dense layer with 1 output
            self.attention_source_layer = tf.keras.layers.Dense(units=args.rnn_dim)
            self.attention_state_layer = tf.keras.layers.Dense(units=args.rnn_dim)
            self.attention_weight_layer = tf.keras.layers.Dense(units=1)

        class DecoderTraining(tfa.seq2seq.BaseDecoder):
            def __init__(self, lemmatizer, *args, **kwargs):
                self.lemmatizer = lemmatizer
                super().__init__.__wrapped__(self, *args, **kwargs)

            @property
            def batch_size(self):
                 # TODO(lemmatizer_noattn): Return the batch size of self.source_states, using tf.shape
                return tf.shape(self.source_states)[0]
            @property
            def output_size(self):
                 # TODO(lemmatizer_noattn): Return `tf.TensorShape(number of logits per each output element)`
                 # By output element we mean characters.
                return tf.TensorShape(self.lemmatizer.num_target_chars)
            @property
            def output_dtype(self):
                 # TODO(lemmatizer_noattn): Return the type of the logits
                return tf.float32

            def with_attention(self, inputs, states):
                # TODO: Compute the Bahdanau attention.
                # - Pass `states` though self._model.attention_state_layer.
                W1_hd = self.lemmatizer.attention_state_layer(states)
                # - Take self.source_states` and pass it through the self.lemmatizer.attention_source_layer.
                #   Because self.source_states does not change, you should in fact do it in `initialize`.
                W2_he = self.lemmatizer.attention_source_layer(self.source_states)
                # - Sum the two outputs. However, the first has shape [a, b, c] and the second [a, c]. Therefore,
                #   expand the second to [a, b, c] or [a, 1, c] (the latter works because of broadcasting rules).
                # - Pass the sum through `tf.tanh` and through the self._model.attention_weight_layer.
                
                # score shape == (batch_size, max_length, rnn_dim)
                score = tf.nn.tanh(tf.expand_dims(W1_hd, 1) + W2_he)
                # score shape == (batch_size, max_length, 1)
                score = self.lemmatizer.attention_weight_layer(score)
                # - Then, run softmax on a suitable axis, generating `weights`.
                # weights shape == (batch_size, max_length, 1) (one weight for each input rnn cell)
                weights = tf.nn.softmax(score, axis=1)
                # - Multiply `self.source_states` with `weights` and sum the result in the axis
                #   corresponding to characters, generating `attention`. Therefore, `attention` is a fixed-size
                #   representation for every batch element, independently on how many characters had
                #   the corresponding input forms.
                attention = weights * self.source_states
                # attention shape == (batch_size, encoder_hidden_size = rnn_dim)
                attention = tf.reduce_sum(attention, axis=1)
                # - Finally concatenate `inputs` and `attention` (in this order) and return the result.
                return tf.keras.layers.concatenate([inputs, attention])

            def initialize(self, layer_inputs, initial_state=None, mask=None):
                self.source_states, self.targets = layer_inputs

                # TODO(lemmatizer_noattn): Define `finished` as a vector of self.batch_size of `False` [see tf.fill].
                finished = tf.fill([self.batch_size], False)
                # TODO(lemmatizer_noattn): Define `inputs` as a vector of self.batch_size of MorphoDataset.Factor.BOW,
                #   embedded using self.lemmatizer.target_embedding
                inputs = self.lemmatizer.target_embedding(
                    tf.fill([self.batch_size], MorphoDataset.Factor.BOW))
                # TODO: Define `states` as the representation of the first character
                #   in `source_states`. The idea is that it is most relevant for generating
                #   the first letter and contains all following characters via the backward RNN.
                states = self.source_states[:, 0, :]
                # TODO: Pass `inputs` through `self._with_attention(inputs, states)`.
                inputs = self.with_attention(inputs, states)
                return finished, inputs, states

            def step(self, time, inputs, states, training):
                # TODO(lemmatizer_noattn): Pass `inputs` and `[states]` through self.lemmatizer.target_rnn_cell,
                #   which returns `(outputs, [states])`.
                outputs, [states] = self.lemmatizer.target_rnn_cell(inputs, [states])
                # TODO(lemmatizer_noattn): Overwrite `outputs` by passing them through self.lemmatizer.target_output_layer,
                outputs = self.lemmatizer.target_output_layer(outputs)
                # TODO(lemmatizer_noattn): Define `next_inputs` by embedding `time`-th chars from `self.targets`.
                # self.targets[:, time] shape == (batch_size, 1, cle_dim)
                next_inputs = self.lemmatizer.target_embedding(self.targets[:, time])
                # TODO(lemmatizer_noattn): Define `finished` as True if `time`-th char from `self.targets` is EOW, False otherwise.
                finished = (self.targets[:, time] == MorphoDataset.Factor.EOW)
                # TODO: Pass `next_inputs` through `self._with_attention(next_inputs, states)`.
                next_inputs = self.with_attention(next_inputs, states)
                return outputs, states, next_inputs, finished

        class DecoderPrediction(DecoderTraining):
            @property
            def output_size(self):
                 # TODO(lemmatizer_noattn): Return `tf.TensorShape()` describing a scalar element,
                 # because we are generating scalar predictions now.
                return tf.TensorShape([])
            @property
            def output_dtype(self):
                 # TODO(lemmatizer_noattn): Return the type of the generated predictions
                return tf.int32

            def initialize(self, layer_inputs, initial_state=None, mask=None):
                self.source_states = layer_inputs
                # TODO(DecoderTraining): Use the same initialization as in DecoderTraining.
                finished = tf.fill([self.batch_size], False)
                inputs = self.lemmatizer.target_embedding(
                    tf.fill([self.batch_size], MorphoDataset.Factor.BOW))
                states = self.source_states[:, 0, :]
                inputs = self.with_attention(inputs, states)
                return finished, inputs, states

            def step(self, time, inputs, states, training):
                # TODO(DecoderTraining): Pass `inputs` and `[states]` through self.lemmatizer.target_rnn_cell,
                #   which returns `(outputs, [states])`.
                outputs, [states] = self.lemmatizer.target_rnn_cell(inputs, [states])
                # TODO(DecoderTraining): Overwrite `outputs` by passing them through self.lemmatizer.target_output_layer,
                outputs = self.lemmatizer.target_output_layer(outputs)
                # TODO: Overwrite `outputs` by passing them through `tf.argmax` on suitable axis and with
                #   `output_type=tf.int32` parameter.
                # axis 0 is the batch level, and axis 1 is the logits level (of dimension self.num_target_chars)
                outputs = tf.argmax(outputs, axis=1, output_type=tf.int32)
                # TODO: Define `next_inputs` by embedding the `outputs`
                next_inputs = self.lemmatizer.target_embedding(outputs)
                # TODO: Define `finished` as True if `outputs` are EOW, False otherwise.
                finished = (outputs == MorphoDataset.Factor.EOW)
                # TODO: Pass `next_inputs` through `self._with_attention(next_inputs, states)`.
                next_inputs = self.with_attention(next_inputs, states)
                return outputs, states, next_inputs, finished

        def call(self, inputs):
            # If `inputs` is a list of two elements, we are in the teacher forcing mode.
            # Otherwise, we run in autoregressive mode.
            if isinstance(inputs, list) and len(inputs) == 2:
                source_charseqs, target_charseqs = inputs
            else:
                source_charseqs, target_charseqs = inputs, None
            source_charseqs_shape = tf.shape(source_charseqs)

            # Get indices of valid lemmas and reshape the `source_charseqs`
            # so that it is a list of valid sequences, instead of a
            # matrix of sequences, some of them padding ones.
            valid_words = tf.cast(tf.where(source_charseqs[:, :, 0] != 0), tf.int32)
            source_charseqs = tf.gather_nd(source_charseqs, valid_words)
            if target_charseqs is not None:
                target_charseqs = tf.gather_nd(target_charseqs, valid_words)

            # TODO: Embed source_charseqs using `source_embedding`
            source_embedding = self.source_embedding(source_charseqs)
            # TODO: Run source_rnn on the embedded sequences, returning outputs in `source_states`.
            source_states = self.source_rnn(source_embedding)

            # Run the appropriate decoder
            if target_charseqs is not None:
                # TODO(lemmatizer_noattn): Create a self.DecoderTraining by passing `self` to its constructor.
                # Then run it on `[source_states, target_charseqs]` input,
                # storing the first result in `output_layer` and the third result in `output_lens`.
                decoderTraining = self.DecoderTraining(self)
                output_layer, _, output_lens = decoderTraining(
                    [source_states, target_charseqs])
            else:
                # TODO(lemmatizer_noattn): Create a self.DecoderPrediction by using:
                # - `self` as first argument to its constructor
                # - `maximum_iterations=tf.shape(source_charseqs)[1] + 10` as
                #   another argument, which indicates that the longest prediction
                #   must be at most 10 characters longer than the longest input.
                # Then run it on `source_states`, storing the first result
                # in `output_layer` and the third result in `output_lens`.
                decoderPrediction = self.DecoderPrediction(
                    self, maximum_iterations=tf.shape(source_charseqs)[1] + 10)
                output_layer, _, output_lens = decoderPrediction(source_states)

            # Reshape the output to the original matrix of lemmas
            # and explicitly set mask for loss and metric computation.
            output_layer = tf.scatter_nd(valid_words, output_layer, tf.concat([source_charseqs_shape[:2], tf.shape(output_layer)[1:]], axis=0))
            output_layer._keras_mask = tf.sequence_mask(tf.scatter_nd(valid_words, output_lens, source_charseqs_shape[:2]))
            return output_layer

    def __init__(self, args, num_source_chars, num_target_chars):
        self.lemmatizer = self.Lemmatizer(args, num_source_chars, num_target_chars)

        self.lemmatizer.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="character_accuracy")],
        )
        self.writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def append_eow(self, sequences):
        """Append EOW character after end of every given sequence."""
        padded_sequences = np.pad(sequences, [[0, 0], [0, 0], [0, 1]])
        ends = np.logical_xor(padded_sequences != 0, np.pad(sequences, [[0, 0], [0, 0], [1, 0]], constant_values=1) != 0)
        padded_sequences[ends] = MorphoDataset.Factor.EOW
        return padded_sequences

    def train_epoch(self, dataset, args):
        for batch in dataset.batches(args.batch_size):
            # TODO(lemmatizer_noattn): Create `targets` by append EOW after target lemmas
            targets = self.append_eow(batch[dataset.LEMMAS].charseqs)
            # TODO(lemmatizer_noattn): Train the lemmatizer using `train_on_batch` method, storing
            # result metrics in `metrics`. You need to pass the `targets`
            # both on input and as gold labels.
            metrics = self.lemmatizer.train_on_batch(x=[batch[dataset.FORMS].charseqs, targets],
                                                     y=targets)
            # Generate the summaries each 10 steps
            iteration = int(self.lemmatizer.optimizer.iterations)
            if iteration % 10 == 0:
                tf.summary.experimental.set_step(iteration)
                metrics = dict(zip(self.lemmatizer.metrics_names, metrics))

                predictions = self.predict_batch(batch[dataset.FORMS].charseqs[:1]).numpy()
                form = "".join(dataset.data[dataset.FORMS].alphabet[i] for i in batch[dataset.FORMS].charseqs[0, 0] if i)
                gold_lemma = "".join(dataset.data[dataset.LEMMAS].alphabet[i] for i in targets[0, 0] if i)
                system_lemma = "".join(dataset.data[dataset.LEMMAS].alphabet[i] for i in predictions[0, 0] if i != MorphoDataset.Factor.EOW)
                status = ", ".join([*["{}={:.4f}".format(name, value) for name, value in metrics.items()],
                                    "{} {} {}".format(form, gold_lemma, system_lemma)])
                print("Step {}:".format(iteration), status)

                with self.writer.as_default():
                    for name, value in metrics.items():
                        tf.summary.scalar("train/{}".format(name), value)
                    tf.summary.text("train/prediction", status)

    @tf.function(experimental_relax_shapes=True)
    def predict_batch(self, charseqs):
        return self.lemmatizer(charseqs)

    def evaluate(self, dataset, dataset_name, args):
        correct_lemmas, total_lemmas = 0, 0
        for batch in dataset.batches(args.batch_size):
            predictions = self.predict_batch(batch[dataset.FORMS].charseqs).numpy()

            # Compute whole lemma accuracy
            targets = self.append_eow(batch[dataset.LEMMAS].charseqs)
            resized_predictions = np.concatenate([predictions, np.zeros_like(targets)], axis=2)[:, :, :targets.shape[2]]
            valid_lemmas = targets[:, :, 0] != MorphoDataset.Factor.EOW

            total_lemmas += np.sum(valid_lemmas)
            correct_lemmas += np.sum(valid_lemmas * np.all(targets == resized_predictions * (targets != 0), axis=2))

        metrics = {"lemma_accuracy": correct_lemmas / total_lemmas}
        with self.writer.as_default():
            tf.summary.experimental.set_step(self.lemmatizer.optimizer.iterations)
            for name, value in metrics.items():
                tf.summary.scalar("{}/{}".format(dataset_name, name), value)

        return metrics
    
    def train(self, morpho, args):
        for epoch in range(args.epochs):
            self.train_epoch(morpho.train, args)
            metrics = self.evaluate(morpho.dev, "dev", args)
            print("Evaluation on {}, epoch {}: {}".format("dev", epoch + 1, metrics))

    def predict(self, dataset, args):
        predictions = []
        for batch in dataset.batches(args.batch_size):
            prediction_batch = self.predict_batch(batch[dataset.FORMS].charseqs).numpy().tolist()
            for sentence in prediction_batch:
                predictions.append(sentence)
        return predictions
            

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
    parser.add_argument("--cle_dim", default=64, type=int, help="CLE embedding dimension.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--max_sentences", default=5000, type=int, help="Maximum number of sentences to load.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--rnn_dim", default=64, type=int, help="RNN cell dimension.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = tf.initializers.GlorotUniform(seed=args.seed)
        tf.keras.utils.get_custom_objects()["orthogonal"] = tf.initializers.Orthogonal(seed=args.seed)
        tf.keras.utils.get_custom_objects()["uniform"] = tf.initializers.RandomUniform(seed=args.seed)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Create the network and train
    network = Network(args,
                      num_source_chars=len(morpho.train.data[morpho.train.FORMS].alphabet),
                      num_target_chars=len(morpho.train.data[morpho.train.LEMMAS].alphabet))
    for epoch in range(args.epochs):
        network.train_epoch(morpho.train, args)
        metrics = network.evaluate(morpho.dev, "dev", args)
        print("Evaluation on {}, epoch {}: {}".format("dev", epoch + 1, metrics))

    metrics = network.evaluate(morpho.test, "test", args)
    with open("lemmatizer.out", "w") as out_file:
        print("{:.2f}".format(100 * metrics["lemma_accuracy"]), file=out_file)
