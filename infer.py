from utils import get_model_config
from utils import get_infer_batch, get_infer_config
from utils import load, prepare_infer_data, create_vocab_tables

from model.word2seq import Word2SeqModel
import argparse
import yaml
import tensorflow as tf
import numpy as np
import time
from pprint import pprint
import utils

def parse_args():
    '''
    Parse Seq2seq with attention arguments.
    '''
    parser = argparse.ArgumentParser(description="Run seq2seq inference.")

    parser.add_argument('--config', nargs='?',
                        default='./configs/basic_config.yaml',
                        help='Configuration file for model specifications')

    return parser.parse_args()


def token_to_str(tokens, reverse_vocab_table):
    tokens = list(tokens)
    word_list = [reverse_vocab_table[id] for id in tokens if id > 0]
    sentence = " ".join(word_list)
    return sentence


def main(args):
    # loading configurations
    with open(args.config) as f:
        config = yaml.safe_load(f)["configuration"]

    work_space = config["workspace"]
    name = config["Name"]

    # Construct or load embeddings
    print("Initializing embeddings ...")
    vocab_size = config["embeddings"]["vocab_size"]
    embed_size = config["embeddings"]["embed_size"]
    vocab_file = '%s/data/%s-%s' % (work_space, "vocab", vocab_size)
    print("\tDone.")

    (enc_num_layers, enc_num_units, enc_cell_type, enc_bidir,
     attn_num_units, dec_num_layers, dec_num_units, dec_cell_type,
     state_pass, infer_max_iter, l2_regularize, learning_rate
     ) = get_model_config(config)

    (is_beam_search, beam_size, batch_size,
     infer_source_file, infer_source_max_length,
     output_path, gpu_fraction, gpu_id) = get_infer_config(config)

    print("Building model architecture ...")
    infer_model = Word2SeqModel(
        mode='infer', model_name=name,
        vocab_size=vocab_size, embedding_size=embed_size,
        enc_num_layers=enc_num_layers, enc_num_units=enc_num_units, enc_cell_type=enc_cell_type, enc_bidir=enc_bidir,
        attn_num_units=attn_num_units,
        dec_num_layers=dec_num_layers, dec_num_units=dec_num_units, dec_cell_type=dec_cell_type,
        batch_size=batch_size, beam_search=is_beam_search, beam_size=beam_size, infer_max_iter=infer_max_iter,
        l2_regularize=l2_regularize, learning_rate=learning_rate, state_pass=state_pass)
    print("\tDone.")

    # Set up session
    restore_from = '%s/nn_models/' % work_space
    gpu_fraction = config["training"]["gpu_fraction"]
    gpu_id = config["training"]["gpu_id"]
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, visible_device_list=gpu_id)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                            gpu_options=gpu_options))

    init = tf.global_variables_initializer()
    sess.run(init)
    # print('global_variables:\n')
    # glob_var = tf.global_variables()
    # pprint(glob_var)

    try:
        saved_global_step = load(infer_model.saver, sess, restore_from)
        if saved_global_step is None:
            raise ValueError("Cannot find the checkpoint to restore from.")

    except Exception:
        print("Something went wrong while restoring checkpoint. ")
        raise

    # ##### Inference #####
    # Load data
    print("Loading inference data ...")

    # Load vocabularies.
    vocab_table, reverse_vocab_table = create_vocab_tables(vocab_file)

    src_dataset = prepare_infer_data(infer_source_file, vocab_table, max_length=infer_source_max_length)
    print("\tDone.")

    # Inference
    print("Start inferring ...")
    final_result = []

    for ith in range(int(len(src_dataset) / batch_size)):
        start = ith
        end = ith + 1
        batch = get_infer_batch(src_dataset, start, end, infer_source_max_length, max_src_num=5)
        sentence = token_to_str(batch[0][0], reverse_vocab_table)

        start_time = time.time()
        result = infer_model.infer(sess, batch)
        duration =round((time.time() - start_time), 3)
        print("sentence:%s, cost:%s s" % (ith, duration))

        res = "src:{}\n".format(sentence)
        if is_beam_search is True:
            for idx, i in enumerate(result[0][0]):
                reply = token_to_str(i, reverse_vocab_table)
                res += "\tpred %s:%s\n" % (idx, reply)
            res += "\n"
        else:
            reply = result[0][0]
            reply = token_to_str(reply, reverse_vocab_table)
            res += "\tpred:%s\n\n" % reply
        print(res)
        final_result.append(res)

    with open(config["inference"]["output_path"], 'w') as f:
        for i in final_result:
            f.write(i+'\n')
    print("\tDone.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
