#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
sanity_check.py: sanity checks for assignment 5
Usage:
    sanity_check.py 1a
    sanity_check.py 1b
    sanity_check.py 1c
    sanity_check.py 1d
    sanity_check.py 1e
    sanity_check.py 1f
    sanity_check.py 2a
    sanity_check.py 2b
    sanity_check.py 2c
    sanity_check.py 2d
"""
import json
import pickle
import sys

import numpy as np
import torch
import torch.nn.utils
import torch.nn as nn
from docopt import docopt

from char_decoder import CharDecoder
from nmt_model import NMT
from utils import pad_sents_char
from vocab import Vocab, VocabEntry
from highway import Highway
from cnn import CNN

# ----------
# CONSTANTS
# ----------
BATCH_SIZE = 5
EMBED_SIZE = 3
HIDDEN_SIZE = 3
DROPOUT_RATE = 0.0


class DummyVocab():
    def __init__(self):
        self.char2id = json.load(open('./sanity_check_en_es_data/char_vocab_sanity_check.json', 'r'))
        self.id2char = {id: char for char, id in self.char2id.items()}
        self.char_unk = self.char2id['<unk>']
        self.start_of_word = self.char2id["{"]
        self.end_of_word = self.char2id["}"]


def question_1a_sanity_check():
    """ Sanity check for words2charindices function.
    """
    print("-" * 80)
    print("Running Sanity Check for Question 1a: words2charindices()")
    print("-" * 80)
    vocab = VocabEntry()

    print('Running test on small list of sentences')
    sentences = [["a", "b", "c?"], ["~d~", "c", "b", "a"]]
    small_ind = vocab.words2charindices(sentences)
    small_ind_gold = [[[1, 30, 2], [1, 31, 2], [1, 32, 70, 2]],
                      [[1, 85, 33, 85, 2], [1, 32, 2], [1, 31, 2], [1, 30, 2]]]
    assert (small_ind == small_ind_gold), \
        "small test resulted in indices list {:}, expected {:}".format(small_ind, small_ind_gold)

    print('Running test on large list of sentences')
    tgt_sents = [
        ['<s>', "Let's", 'start', 'by', 'thinking', 'about', 'the', 'member', 'countries', 'of', 'the', 'OECD,', 'or',
         'the', 'Organization', 'of', 'Economic', 'Cooperation', 'and', 'Development.', '</s>'],
        ['<s>', 'In', 'the', 'case', 'of', 'gun', 'control,', 'we', 'really', 'underestimated', 'our', 'opponents.',
         '</s>'],
        ['<s>', 'Let', 'me', 'share', 'with', 'those', 'of', 'you', 'here', 'in', 'the', 'first', 'row.', '</s>'],
        ['<s>', 'It', 'suggests', 'that', 'we', 'care', 'about', 'the', 'fight,', 'about', 'the', 'challenge.', '</s>'],
        ['<s>', 'A', 'lot', 'of', 'numbers', 'there.', 'A', 'lot', 'of', 'numbers.', '</s>']]
    tgt_ind = vocab.words2charindices(tgt_sents)
    tgt_ind_gold = pickle.load(open('./sanity_check_en_es_data/1a_tgt.pkl', 'rb'))
    assert (tgt_ind == tgt_ind_gold), "target vocab test resulted in indices list {:}, expected {:}".format(tgt_ind,
                                                                                                            tgt_ind_gold)

    print("All Sanity Checks Passed for Question 1a: words2charindices()!")
    print("-" * 80)


def question_1b_sanity_check():
    """ Sanity check for pad_sents_char() function.
    """
    print("-" * 80)
    print("Running Sanity Check for Question 1b: Padding")
    print("-" * 80)
    vocab = VocabEntry()

    print("Running test on a list of sentences")
    sentences = [['Human:', 'What', 'do', 'we', 'want?'], ['Computer:', 'Natural', 'language', 'processing!'],
                 ['Human:', 'When', 'do', 'we', 'want', 'it?'], ['Computer:', 'When', 'do', 'we', 'want', 'what?']]
    word_ids = vocab.words2charindices(sentences)

    padded_sentences = pad_sents_char(word_ids, 0)
    gold_padded_sentences = torch.load('./sanity_check_en_es_data/gold_padded_sentences.pkl')
    assert padded_sentences == gold_padded_sentences, "Sentence padding is incorrect: it should be:\n {} but is:\n{}".format(
        gold_padded_sentences, padded_sentences)

    print("Sanity Check Passed for Question 1b: Padding!")
    print("-" * 80)


def question_1c_sanity_check():
    vocab = VocabEntry()

    tgt_sents = [
        ['<s>', "Let's", 'start', 'by', 'thinking', 'about', 'the', 'member', 'countries', 'of', 'the', 'OECD,', 'or',
         'the', 'Organization', 'of', 'Economic', 'Cooperation', 'and', 'Development.', '</s>'],
        ['<s>', 'In', 'the', 'case', 'of', 'gun', 'control,', 'we', 'really', 'underestimated', 'our', 'opponents.',
         '</s>'],
        ['<s>', 'Let', 'me', 'share', 'with', 'those', 'of', 'you', 'here', 'in', 'the', 'first', 'row.', '</s>'],
        ['<s>', 'It', 'suggests', 'that', 'we', 'care', 'about', 'the', 'fight,', 'about', 'the', 'challenge.', '</s>'],
        ['<s>', 'A', 'lot', 'of', 'numbers', 'there.', 'A', 'lot', 'of', 'numbers.', '</s>']]
    tgt_ind = vocab.to_input_tensor_char(tgt_sents, device=torch.device('cpu'))
    assert [x for x in tgt_ind.shape] == [len(tgt_sents[0]), len(tgt_sents), 21]

    print("Sanity Check Passed for Question 1c: Tensor!")
    print("-" * 80)


def question_1d_sanity_check():
    conv_out = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    ex_out_proj = [
        [1.9, 1.9, 1.9],
        [4.6, 4.6, 4.6],
        [7.3, 7.3, 7.3],
    ]

    ex_out_gate = [
        [1.9, 1.9, 1.9],
        [4.6, 4.6, 4.6],
        [7.3, 7.3, 7.3],
    ]
    conv_out_tensor = torch.tensor(conv_out, dtype=torch.float, device=torch.device('cpu'))
    ex_out_proj_relu = nn.functional.relu(torch.tensor(ex_out_proj, dtype=torch.float, device=torch.device('cpu')))
    ex_out_gate_sigmoid = nn.functional.sigmoid(
        torch.tensor(ex_out_gate, dtype=torch.float, device=torch.device('cpu')))
    expected_final_output = torch.mul(ex_out_proj_relu, ex_out_gate_sigmoid) + torch.mul((1 - ex_out_gate_sigmoid),
                                                                                         conv_out_tensor)

    highway = Highway(filter_size=3)
    reinitialize_layers(highway)
    output = highway.forward(conv_out_tensor)

    assert np.allclose(output.data.numpy(),
                       expected_final_output.data.numpy())
    print(output)
    print("Here is expected value: ")
    print(expected_final_output)
    print("Sanity Check Passed for Question 1d: Highway!")
    print("-" * 80)


def question_1e_sanity_check():
    vocab = VocabEntry()

    input = [[
        [1, 2, 3], [4, 5, 6], [1, 2, 3], [0, 0, 0]]  # ata
    ]
    input_tensor = torch.tensor(input, dtype=torch.float, device=torch.device('cpu'))

    # m = nn.Conv1d(3, 2, 5)
    # input = torch.randn(1, 3, 10)
    # output = m(input)
    # print(output)

    cnn = CNN(char_embedded_size=EMBED_SIZE, filter_size=2, kernel_size=1)
    output = cnn(input_tensor)
    print(output)

    print("Sanity Check Passed for Question 1e: CNN!")
    print("-" * 80)


def reinitialize_layers(model):
    """ Reinitialize the Layer Weights for Sanity Checks.
    """

    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.3)
            if m.bias is not None:
                m.bias.data.fill_(0.1)
        # elif type(m) == nn.Embedding:
        #     m.weight.data.fill_(0.15)
        # elif type(m) == nn.Dropout:
        #     nn.Dropout(DROPOUT_RATE)

    with torch.no_grad():
        model.apply(init_weights)


def question_1f_sanity_check(model):
    """ Sanity check for model_embeddings.py
        basic shape check
    """
    print("-" * 80)
    print("Running Sanity Check for Question 1f: Model Embedding")
    print("-" * 80)
    sentence_length = 10
    max_word_length = 21
    inpt = torch.zeros(sentence_length, BATCH_SIZE, max_word_length, dtype=torch.long)
    ME_source = model.model_embeddings_source
    output = ME_source.forward(inpt)
    output_expected_size = [sentence_length, BATCH_SIZE, EMBED_SIZE]
    assert (list(
        output.size()) == output_expected_size), "output shape is incorrect: it should be:\n {} but is:\n{}".format(
        output_expected_size, list(output.size()))
    print("Sanity Check Passed for Question 1f: Model Embedding!")
    print("-" * 80)


def question_2a_sanity_check(decoder, char_vocab):
    """ Sanity check for CharDecoder.__init__()
        basic shape check
    """
    print("-" * 80)
    print("Running Sanity Check for Question 2a: CharDecoder.__init__()")
    print("-" * 80)
    assert (
            decoder.charDecoder.input_size == EMBED_SIZE), "Input dimension is incorrect:\n it should be {} but is: {}".format(
        EMBED_SIZE, decoder.charDecoder.input_size)
    assert (
            decoder.charDecoder.hidden_size == HIDDEN_SIZE), "Hidden dimension is incorrect:\n it should be {} but is: {}".format(
        HIDDEN_SIZE, decoder.charDecoder.hidden_size)
    assert (
            decoder.char_output_projection.in_features == HIDDEN_SIZE), "Input dimension is incorrect:\n it should be {} but is: {}".format(
        HIDDEN_SIZE, decoder.char_output_projection.in_features)
    assert (decoder.char_output_projection.out_features == len(
        char_vocab.char2id)), "Output dimension is incorrect:\n it should be {} but is: {}".format(
        len(char_vocab.char2id), decoder.char_output_projection.out_features)
    assert (decoder.decoderCharEmb.num_embeddings == len(
        char_vocab.char2id)), "Number of embeddings is incorrect:\n it should be {} but is: {}".format(
        len(char_vocab.char2id), decoder.decoderCharEmb.num_embeddings)
    assert (
            decoder.decoderCharEmb.embedding_dim == EMBED_SIZE), "Embedding dimension is incorrect:\n it should be {} but is: {}".format(
        EMBED_SIZE, decoder.decoderCharEmb.embedding_dim)
    print("Sanity Check Passed for Question 2a: CharDecoder.__init__()!")
    print("-" * 80)


def question_2b_sanity_check(decoder, char_vocab):
    """ Sanity check for CharDecoder.forward()
        basic shape check
    """
    print("-" * 80)
    print("Running Sanity Check for Question 2b: CharDecoder.forward()")
    print("-" * 80)
    sequence_length = 4
    inpt = torch.zeros(sequence_length, BATCH_SIZE, dtype=torch.long)
    logits, (dec_hidden1, dec_hidden2) = decoder.forward(inpt)
    logits_expected_size = [sequence_length, BATCH_SIZE, len(char_vocab.char2id)]
    dec_hidden_expected_size = [1, BATCH_SIZE, HIDDEN_SIZE]
    assert (list(
        logits.size()) == logits_expected_size), "Logits shape is incorrect:\n it should be {} but is:\n{}".format(
        logits_expected_size, list(logits.size()))
    assert (list(
        dec_hidden1.size()) == dec_hidden_expected_size), "Decoder hidden state shape is incorrect:\n it should be {} but is: {}".format(
        dec_hidden_expected_size, list(dec_hidden1.size()))
    assert (list(
        dec_hidden2.size()) == dec_hidden_expected_size), "Decoder hidden state shape is incorrect:\n it should be {} but is: {}".format(
        dec_hidden_expected_size, list(dec_hidden2.size()))
    print("Sanity Check Passed for Question 2b: CharDecoder.forward()!")
    print("-" * 80)


def question_2c_sanity_check(decoder):
    """ Sanity check for CharDecoder.train_forward()
        basic shape check
    """
    print("-" * 80)
    print("Running Sanity Check for Question 2c: CharDecoder.train_forward()")
    print("-" * 80)
    sequence_length = 4
    inpt = torch.zeros(sequence_length, BATCH_SIZE, dtype=torch.long)
    loss = decoder.train_forward(inpt)
    assert (list(loss.size()) == []), "Loss should be a scalar but its shape is: {}".format(list(loss.size()))
    print("Sanity Check Passed for Question 2c: CharDecoder.train_forward()!")
    print("-" * 80)


def question_2d_sanity_check(decoder):
    """ Sanity check for CharDecoder.decode_greedy()
        basic shape check
    """
    print("-" * 80)
    print("Running Sanity Check for Question 2d: CharDecoder.decode_greedy()")
    print("-" * 80)
    sequence_length = 4
    inpt = torch.zeros(1, BATCH_SIZE, HIDDEN_SIZE, dtype=torch.float)
    initialStates = (inpt, inpt)
    device = decoder.char_output_projection.weight.device
    decodedWords = decoder.decode_greedy(initialStates, device)
    assert (len(decodedWords) == BATCH_SIZE), "Length of decodedWords should be {} but is: {}".format(BATCH_SIZE,
                                                                                                      len(decodedWords))
    print("Sanity Check Passed for Question 2d: CharDecoder.decode_greedy()!")
    print("-" * 80)


def main():
    """ Main func.
    """
    args = docopt(__doc__)

    # Check Python & PyTorch Versions
    assert (sys.version_info >= (3, 5)), "Please update your installation of Python to version >= 3.5"
    assert (
            torch.__version__ >= "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0 or greater".format(
        torch.__version__)

    # Seed the Random Number Generators
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    vocab = Vocab.load('./sanity_check_en_es_data/vocab_sanity_check.json')

    # Create NMT Model
    model = NMT(
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        dropout_rate=DROPOUT_RATE,
        vocab=vocab)

    char_vocab = DummyVocab()

    # Initialize CharDecoder
    decoder = CharDecoder(
        hidden_size=HIDDEN_SIZE,
        char_embedding_size=EMBED_SIZE,
        target_vocab=char_vocab)

    if args['1a']:
        question_1a_sanity_check()
    elif args['1b']:
        question_1b_sanity_check()
    elif args['1c']:
        question_1c_sanity_check()
    elif args['1d']:
        question_1d_sanity_check()
    elif args['1e']:
        question_1e_sanity_check()
    elif args['1f']:
        question_1f_sanity_check(model)
    elif args['2a']:
        question_2a_sanity_check(decoder, char_vocab)
    elif args['2b']:
        question_2b_sanity_check(decoder, char_vocab)
    elif args['2c']:
        question_2c_sanity_check(decoder)
    elif args['2d']:
        question_2d_sanity_check(decoder)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
