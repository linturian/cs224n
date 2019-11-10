#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn.utils as utils
import torch.nn as nn
import torch.nn.functional as F
import torch


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab

        self.char_embedding_size = char_embedding_size
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size, bidirectional=False)
        self.char_output_projection = nn.Linear(hidden_size, len(target_vocab.char2id))

        self.pad_token_idx = target_vocab.char2id['<pad>']
        self.decoderCharEmb = nn.Embedding(len(target_vocab.char2id), char_embedding_size, padding_idx=self.pad_token_idx)

        ### END YOUR CODE

    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.

        # word_lengths = [len(w) for w in input]

        x_char_embed = self.decoderCharEmb(input)  # Shape (num_char, batch_size, char_embeding)
        # x_char_embed = utils.rnn.pack_padded_sequence(x_char_embed, word_lengths)
        char_dec_hiddens, (last_hidden, last_cell) = self.charDecoder(x_char_embed, dec_hidden)
        # char_dec_hiddens, _ = utils.rnn.pad_packed_sequence(char_dec_hiddens)
        s_t = self.char_output_projection(char_dec_hiddens)
        return s_t, (last_hidden, last_cell)


        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).

        input = char_sequence[:-1, :]   # Shape (char_len - 1, num_word) e.g. <START>,m,u,s,i,c
        target = char_sequence[1:, :]   # Shape (char_len - 1, num_word) e.g. m,u,s,i,c,<END>

        x_char_embed = self.decoderCharEmb(input)  # Shape (char_len - 1, num_word, emb_size)
        char_dec_hiddens, _ = self.charDecoder(x_char_embed, dec_hidden)
        s_t = self.char_output_projection(char_dec_hiddens)  # Shape (char_len - 1, num_word, all_char)

        # Reshape for CrossEntropyLoss
        s_t = s_t.reshape(-1, s_t.shape[2])  # Shape (char_len * num_word, all_char)
        target = target.reshape(-1)  # Shape (char_len - 1 * num_word)

        loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_idx, reduction='sum')
        output = loss(s_t, target)
        # output.backward()

        return output
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        dec_hidden, dec_cell = initialStates
        batch_size = dec_hidden.shape[1]
        start_words = [[self.target_vocab.start_of_word for b in range(batch_size)]]
        output_words = torch.tensor(start_words, dtype=torch.long, device=device)
        # shape (length=1, batch)
        curr_chars = torch.tensor(start_words, dtype=torch.long, device=device)
        for t in range(max_length):
            s_t, (dec_hidden, dec_cell) = self.forward(curr_chars, (dec_hidden, dec_cell))  # Shape (char_len, num_word, all_char)
            p_t = F.log_softmax(s_t, dim=2)
            curr_chars = torch.argmax(p_t, dim=2)
            output_words = torch.cat((output_words, curr_chars), 0)

        #Truncate the output after
        output_words = output_words[1:, :]
        output_words = output_words.permute(1, 0).tolist()

        decodedWords = []
        for word in output_words:
            decoded_word = ''
            for c_id in word:
                if c_id == self.target_vocab.end_of_word:
                    break
                decoded_word += self.target_vocab.id2char[c_id]
            decodedWords.append(decoded_word)
        return decodedWords

        ### END YOUR CODE

