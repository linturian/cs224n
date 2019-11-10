#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code


        ### YOUR CODE HERE for part 1j
        self.embed_size = embed_size
        pad_token_idx = vocab.char2id['<pad>']
        char_embedded_size = 50

        self.embeddings = nn.Embedding(len(vocab.char2id), char_embedded_size, padding_idx=pad_token_idx)
        self.cnn = CNN(char_embedded_size, filter_size=embed_size, kernel_size=5)
        self.highway = Highway(filter_size=embed_size)
        self.dropout = nn.Dropout(p=0.3)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        embed = self.embeddings(input)  # shape (num_words, num_sentences, max_characters_in_a_word, character_embedding_size)
        embed_shape = embed.shape
        embed_reshape = embed.reshape(-1, embed_shape[2], embed_shape[3])  # shape (num_words * num_sentences, max_characters_in_a_word, character_embedding_size)
        conv_out, _ = self.cnn(embed_reshape)  # shape (num_words * num_sentences, embed_size)
        highway_out = self.highway(conv_out)
        word_embed = self.dropout(highway_out)
        word_embed_reshape = word_embed.reshape(embed_shape[0], embed_shape[1], self.embed_size)  # shape (num_words, num_sentences, embed_size)
        return word_embed_reshape
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j


        ### END YOUR CODE

