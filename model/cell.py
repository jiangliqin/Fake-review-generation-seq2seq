import tensorflow as tf


def single_cell(num_units, cell_type, keep_prob=0.8):
    """
    Cell: build a recurrent cell
        num_units: number of hidden cell units
        cell_type: LSTM, GRU, LN_LSTM (layer_normalize)
    """
    if cell_type == "LSTM":
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, state_is_tuple=True)

    elif cell_type == "RES_LSTM":
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, state_is_tuple=True)
        # 如果使用ResidualWrapper 则必须保证word embedding和rnn cell的size一致
        cell = tf.contrib.rnn.ResidualWrapper(cell)
    else:
        raise ValueError("Unknown cell type %s" % cell_type)

    # cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell


def create_rnn_cell(num_layers, num_units, cell_type):
    """
    RNN_cell: build a multi-layer rnn cell
        num_layers: number of hidden layers
    """
    if num_layers > 1:
        rnn_cell = tf.nn.rnn_cell.MultiRNNCell(
            [single_cell(num_units, cell_type) for _ in range(num_layers)]
        )
    else:
        rnn_cell = single_cell(num_units, cell_type)

    return rnn_cell
