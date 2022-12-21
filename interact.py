#!/usr/bin/env python3
# encoding: UTF-8

"""
Filename: interact.py
Author:   David Oniani
E-mail:   oniani.david@mayo.edu

Description:

    This is an implementation of an interactive chatbot that answer questions
    related to COVID-19/Novel Coronavirus.

    It relies on two state-of-the-art models: GPT-2 and USE (Universal Sentence
    Encoder).

    We use 774M model of GPT-2.
"""

import json
import os
import re
import sys

import fire

import numpy as np
import tensorflow as tf

sys.path.insert(1, 'src')
import cleaner
import encoder
import model
import sample
import similarity
import tflex


# String, which MODEL to use
MODEL_NAME = "vapebros"

# Integer seed for random number generators, fix seed to reproduce results
SEED = None

# Number of samples to return total
NSAMPLES = 1

# Number of batches (only affects speed/memory, must divide nsamples)
BATCH_SIZE = 1

# Number of tokens in generated text, if None (default), is determined by the
# hyperparameters of the model
LENGTH = None

# Float value controlling randomness in boltzmann distribution. Lower
# TEMPERATURE results in less random completions. As the TEMPERATURE
# approaches zero, the MODEL will become deterministic and repetitive.
# Higher TEMPERATURE results in more random completions.
TEMPERATURE = 0.7

# Integer value controlling diversity. 1 means only 1 word is considered
# for each step (token), resulting in deterministic completions, while 40
# means 40 words are considered at each step. 0 (default) is a special
# setting meaning no restrictions. 40 generally is a good value.
TOP_K = 40

# Path to parent folder containing MODEL subfolders
# (i.e. contains the <MODEL_NAME> folder)
MODELS_DIR = "models"

# Path to the saved MODEL info
CHECKPOINT = "models/vapebros/model-77.hdf5"

TRUNCATE = False

SPLIT_CONTEXT = 0.4
INCLUDE_PREFIX=True
TOP_P = 0.8


def main():
    """Run the MODEL interactively."""

    print("\nWelcome to COVID-19 chatbot!")
    print("The input prompt will appear shortly\n\n")

    models_dir = os.path.expanduser(os.path.expandvars(MODELS_DIR))

    assert NSAMPLES % BATCH_SIZE == 0

    enc = encoder.get_encoder(MODEL_NAME)
    hparams = model.default_hparams()

    with open(os.path.join(models_dir, MODEL_NAME, "hparams.json")) as file:
        hparams.override_from_dict(json.load(file))

    if LENGTH is None:
        length = hparams.n_ctx // 2

    elif LENGTH > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: {}".format(
                hparams.n_ctx
            )
        )

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [BATCH_SIZE, None])
        np.random.seed(SEED)
        tf.set_random_seed(SEED)
        output = sample.sample_sequence(
            hparams=hparams,
            length=length,
            context=context,
            batch_size=BATCH_SIZE,
            temperature=TEMPERATURE,
            top_k=TOP_K,
        )

        saver = tflex.Saver()
        saver.restore(sess, CHECKPOINT)

        while True:
            question = input("COVID-19 CHATBOT> ")

            while not question:
                print("Prompt should not be empty!")
                question = input("COVID-19 CHATBOT> ")

            context_tokens = [enc.encode(question)] * BATCH_SIZE

            # custom for full length text
            total_tokens = len(context_tokens[0])
            generated_once = False
            gen_texts = []
            answers = ""
            split_length = int(1023 * SPLIT_CONTEXT)
            split_output_length = min(length, 1023 - split_length)

            for _ in range(NSAMPLES // BATCH_SIZE):
                gen_text = [np.array([])] * BATCH_SIZE
                truncated = [False] * BATCH_SIZE
                while False in truncated:
                    num_tokens = 1023 - (len(context_tokens[0]))

                    if generated_once:
                        new_split_output_length = min(length - total_tokens, 1023 - split_length)
                        if new_split_output_length != split_output_length:
                            split_output = sample.sample_sequence(
                                hparams=hparams,
                                length=new_split_output_length,
                                start_token=enc.encoder['<|endoftext|>'] if not question else None,
                                context=context if question else None,
                                batch_size=BATCH_SIZE,
                                temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P
                            )[:, 1:]
                        out = sess.run(split_output, feed_dict={
                            context: context_tokens
                        })

                    else:
                        out = sess.run(output, feed_dict={
                            context: context_tokens
                        })

                    total_tokens += num_tokens
                    for i in range(BATCH_SIZE):
                        text = out[i]
                        trunc_text = ""
                        if question:
                            text = np.append(context_tokens[i][:1], text)
                        if TRUNCATE or all(gen_text):
                            context_tokens[i] = out[i][(1023 - split_length - 1):]
                            if generated_once:
                                text = out[i][split_length:]

                            if TRUNCATE:
                                to_trunc = enc.decode(text)
                                truncate_esc = re.escape(TRUNCATE)
                                if question and not include_prefix:
                                    prefix_esc = re.escape(question)
                                    pattern = '(?:{})(.*?)(?:{})'.format(prefix_esc,
                                                                         truncate_esc)
                                else:
                                    pattern = '(.*?)(?:{})'.format(truncate_esc)

                                trunc_text = re.search(pattern, to_trunc, re.S)
                                if trunc_text:
                                    text = enc.encode(trunc_text.group(1))
                                    # better to re-encode here then decode every generation cycle, I think

                        if not truncated[i]:
                            gen_text[i] = np.concatenate((gen_text[i], text), axis=None)
                            if trunc_text or (length is not None and total_tokens >= length-1):
                                truncated[i] = True
                                gen = enc.decode(gen_text[i]).lstrip('\n')
                                '''
                                if destination_path:
                                    f.write("{}\n{}".format(gen, sample_delim))
                                if not return_as_list and not destination_path:
                                    print("{}\n{}".format(gen, sample_delim), end='')
                                '''
                                answers += gen
                    generated_once = True

                answers = ""
                for idx in range(BATCH_SIZE):
                    answers += enc.decode(out[idx])

                # Process the string (cleanup)
                clean_answers = cleaner.clean_additional(
                    " ".join(cleaner.clean_text(answers))
                )

                final_answers = cleaner.chunk_into_sentences(clean_answers)

                try:
                    #print(similarity.use_filter(question, answers, 5))
                    print(answers)

                except Exception:
                    print(" ".join(answers))
                    print("WARNING: Model cannot generate an answer using USE")
                    '''
                    for _ in range(NSAMPLES // BATCH_SIZE):
                        out = sess.run(
                            output,
                            feed_dict={
                                context: [context_tokens for _ in range(BATCH_SIZE)]
                            },
                        )[:, len(context_tokens) :]

                        # Build the answers string
                        answers = ""
                        for idx in range(BATCH_SIZE):
                            answers += enc.decode(out[idx])

                        # Process the string (cleanup)
                        clean_answers = cleaner.clean_additional(
                            " ".join(cleaner.clean_text(answers))
                        )

                        final_answers = cleaner.chunk_into_sentences(clean_answers)

                        try:
                            #print(similarity.use_filter(question, answers, 5))
                            print(answers)

                        except Exception:
                            print(" ".join(answers))
                            print("WARNING: Model cannot generate an answer using USE")
                    '''

            print()
            print("=" * 79)
            print()


if __name__ == "__main__":
    # Suppress (most) logging messages
    import absl
    import logging

    logger = logging.getLogger()
    logger.disabled = True
    absl.logging._warn_preinit_stderr = 0

    # Disable TensorFlow deprecation warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Run
    fire.Fire(main())
