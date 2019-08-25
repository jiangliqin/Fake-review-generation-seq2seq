# -*- coding:utf-8 -*-
from __future__ import print_function
from pprint import pprint

import tensorflow as tf
import numpy as np
import os
import sys
from collections import defaultdict
import random

random.seed(1994)

PAD = '<pad>'
UNK = '<unk>'
SOS = '<s>'
EOS = '</s>'

PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3

embed_shift = 4


def get_model_config(config):
    enc_num_layers = config["encoder"]["num_layers"]
    enc_num_units = config["encoder"]["num_units"]
    enc_cell_type = config["encoder"]["cell_type"]
    enc_bidir = config["encoder"]["bidirectional"]
    attn_num_units = config["decoder"]["attn_num_units"]
    dec_num_layers = config["decoder"]["num_layers"]
    dec_num_units = config["decoder"]["num_units"]
    dec_cell_type = config["decoder"]["cell_type"]
    state_pass = config["decoder"]["state_pass"]
    infer_max_iter = config["decoder"]["infer_max_iter"]
    l2_regularize = config["training"]["l2_regularize"]
    learning_rate = config["training"]["learning_rate"]

    return (enc_num_layers, enc_num_units, enc_cell_type, enc_bidir,
            attn_num_units, dec_num_layers, dec_num_units, dec_cell_type,
            state_pass, infer_max_iter, l2_regularize, learning_rate)


def get_training_config(config):
    train_s_file = config["training"]["train_source_file"]
    train_t_file = config["training"]["train_target_file"]
    dev_s_file = config["training"]["dev_source_file"]
    dev_t_file = config["training"]["dev_target_file"]
    max_length = config["training"]["max_length"]

    gpu_fraction = config["training"]["gpu_fraction"]
    gpu_id = config["training"]["gpu_id"]

    checkpoint_every = config["training"]["checkpoint_every"]  # 保存模型的步数
    max_checkpoints = config["training"]["max_checkpoints"]  # 最大保留模型的个数
    print_every = config["training"]["print_every"]  # 打印信息
    train_steps = config["training"]["train_steps"]  # 最大训练步数

    is_beam_search = False
    batch_size = config["training"]["batch_size"]
    beam_size = config["inference"]["beam_size"]

    return (train_s_file, train_t_file, dev_s_file, dev_t_file, max_length,
            gpu_fraction, gpu_id, checkpoint_every, max_checkpoints, print_every,
            train_steps, is_beam_search, batch_size, beam_size)


def get_infer_config(config):
    is_beam_search = config["inference"]["is_beam_search"]
    beam_size = config["inference"]["beam_size"]
    batch_size = config["inference"]["infer_batch_size"]
    infer_source_file = config["inference"]["infer_source_file"]
    infer_source_max_length = config["inference"]["infer_source_max_length"]
    output_path = config["inference"]["output_path"]

    gpu_fraction = config["training"]["gpu_fraction"]
    gpu_id = config["training"]["gpu_id"]

    return (is_beam_search, beam_size, batch_size,
            infer_source_file, infer_source_max_length,
            output_path, gpu_fraction, gpu_id)


def load(saver, sess, logdir, model_path=None):
    """
    Load the latest checkpoint
    """
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    if model_path:
        global_step = int(model_path.split("-")[-1])
        print("Global step was: {}".format(global_step))
        print("Restoring...",end="")
        saver.restore(sess, model_path)
        print("\tDone")
        return global_step
    else:
        ckpt = tf.train.get_checkpoint_state(logdir)
        if ckpt:
            print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
            global_step = int(ckpt.model_checkpoint_path
                              .split('/')[-1]
                              .split('-')[-1])
            print("  Global step was: {}".format(global_step))
            print("  Restoring...", end="")
            # vars = tf.train.list_variables(ckpt.model_checkpoint_path)
            # print('checkpoint_variables:')
            # pprint(vars)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(" Done.")
            return global_step
        else:
            print(" No checkpoint found.")
            return None


def save(saver, sess, logdir, step):
    """
    Save the checkpoint
    """
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def create_vocab_file(train_s_file, train_t_file,
                      dev_s_file, dev_t_file,
                      vocab_file, vocab_size):
    vocab = defaultdict(lambda: 0)
    for file_path in [train_s_file, train_t_file, dev_s_file, dev_t_file]:
        for _, line in enumerate(open(file_path, 'r')):
            words = line.strip().split(' ')
            for word in words:
                vocab[word] += 1
    vocab = list(sorted(vocab.items(), key=lambda x: x[1], reverse=True))
    vocab = [item[0] for item in vocab if item[0] != "<unk>"]
    vocab = vocab[:vocab_size - embed_shift]
    vocab = [PAD, UNK, SOS, EOS] + vocab
    with open(vocab_file, 'w') as f:
        for word in vocab:
            f.write("%s\n" % word)


def create_vocab_tables(vocab_file):
    def load_vocab(vocab_file):
        vocab = []
        for _, word in enumerate(open(vocab_file, 'r')):
            vocab.append(word.strip())
        return vocab
    vocab = load_vocab(vocab_file)
    vocab_table = {k: v for v, k in enumerate(vocab)}
    reverse_vocab_table = {k: v for k, v in enumerate(vocab)}
    return vocab_table, reverse_vocab_table


def prepare_train_dev_data(train_s_file, train_t_file,
                            dev_s_file, dev_t_file,
                            vocab_table, max_length, ):

    def readfile(file_path):
        data = []
        for _, line in enumerate(open(file_path, 'r')):
            word_list = line.strip().split(' ')
            data.append([vocab_table.get(word, UNK_ID) for word in word_list])
        return data

    train_s_dataset = readfile(train_s_file)
    train_t_dataset = readfile(train_t_file)
    dev_s_dataset = readfile(dev_s_file)
    dev_t_dataset = readfile(dev_t_file)

    train_dataset = []
    dev_dataset = []
    for idx, i in enumerate(train_s_dataset):
        if len(train_s_dataset[idx]) >= max_length or len(train_t_dataset[idx]) >= max_length:
            continue
        train_dataset.append((train_s_dataset[idx], train_t_dataset[idx]))

    for idx, i in enumerate(dev_s_dataset):
        if len(dev_s_dataset[idx]) >= max_length or len(dev_t_dataset[idx]) >= max_length:
            continue
        dev_dataset.append((dev_s_dataset[idx], dev_t_dataset[idx]))

    return train_dataset, dev_dataset


def prepare_infer_data(infer_data_file, vocab_table, max_length):
    src_dataset = []
    for _, line in enumerate(open(infer_data_file, 'r')):
        word_list = line.strip().split(' ')
        if len(word_list) >= max_length:
            continue
        id_list = [vocab_table.get(word, UNK_ID) for word in word_list]
        src_dataset.append(id_list)

    return np.array(src_dataset)


def get_train_batch(dataset, max_length, batch_size, max_src_num=None):

    encoder_inputs, decoder_inputs, decoder_targets, decoder_targets_masks,\
    encoder_length, decoder_length = [], [], [], [], [], []
    count = 0
    while count < batch_size:
        encoder, decoder = random.choice(dataset)

        enc_leng, dec_leng = len(encoder), len(decoder)
        if enc_leng >= max_length or dec_leng >= max_length:
            continue

        encoder_input = encoder
        e_length = len(encoder_input)
        if max_src_num is not None:  # 当输入是观点词的时候，最大长度为观点词的个数
            pads = PAD_ID * np.ones(max_src_num - e_length, dtype=np.int32)
            encoder_input = np.concatenate([encoder_input, pads])
            encoder_inputs.append(encoder_input)
            encoder_length.append(e_length)
        else:
            pads = PAD_ID * np.ones(max_length - e_length, dtype=np.int32)
            encoder_input = np.concatenate([encoder_input, pads])
            encoder_inputs.append(encoder_input)
            encoder_length.append(e_length)

        decoder_input = [SOS_ID] + decoder
        d_length = len(decoder_input)
        pads = PAD_ID * np.ones(max_length - d_length, dtype=np.int32)
        decoder_input = np.concatenate([decoder_input, pads])
        decoder_inputs.append(decoder_input)

        decoder_target = decoder + [EOS_ID]
        pads = PAD_ID * np.ones(max_length - d_length, dtype=np.int32)
        decoder_target = np.concatenate([decoder_target, pads])
        decoder_targets.append(decoder_target)
        decoder_length.append(d_length)

        count += 1

    target_max_length = max(decoder_length)
    encoder_inputs = np.array(encoder_inputs)
    decoder_inputs = np.array(decoder_inputs)
    decoder_targets = np.array(decoder_targets)
    decoder_targets = decoder_targets[:, :target_max_length]
    decoder_targets_masks = decoder_targets >= 3  # eos字符算在loss里
    decoder_targets_masks = decoder_targets_masks.astype(np.float32)
    encoder_length = np.array(encoder_length)
    decoder_length = np.array(decoder_length)

    return encoder_inputs, decoder_inputs, decoder_targets, decoder_targets_masks, encoder_length, decoder_length


def get_infer_batch(src_dataset, start, end, max_length, max_src_num=None):
    encoder_inputs = []
    encoder_length = []
    batch_data = src_dataset[start: end]
    for id_list in batch_data:
        length = len(id_list)
        if max_src_num is not None:  # 当输入是观点词的时候，最大长度为观点词的个数
            pads = PAD_ID * np.ones(max_src_num - length, dtype=np.int32)
            encoder_input = np.concatenate([id_list, pads])
            encoder_inputs.append(encoder_input)
            encoder_length.append(length)
        else:
            pads = PAD_ID * np.ones(max_length - length, dtype=np.int32)
            encoder_input = np.concatenate([id_list, pads])
            encoder_inputs.append(encoder_input)
            encoder_length.append(length)

    encoder_inputs = np.array(encoder_inputs)
    encoder_length = np.array(encoder_length)

    return encoder_inputs, encoder_length
