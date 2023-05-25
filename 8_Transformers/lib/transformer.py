import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class TransformerModel(nn.Module):
    def __init__(
        self,
        en_vocab_size,
        de_vocab_size,
        max_len=20,
        embed_dim=64,
        enc_layers=6,
        dec_layers=6,
        num_heads=4,
    ):
        """
        Initializes the TransformerModel.

        Args:
            en_vocab_size (int): size of the English vocabulary used in this exercise.
            de_vocab_size (int): size of the German vocabulary used in this exercise.
            max_len (int): maximum number of words in the input sentence.
                Notice that during training the input to the decoder
                will be shifted one token to the right starting
                with <sos>, and will therefore have a maximum length of max_len + 1.
            embed_dim (int): the dimension of the model.
            enc_layers (int): number of encoder blocks.
            dec_layers (int): number of decoder blocks.
            num_heads (int): number of heads in Multi-head attention blocks.

        """
        super().__init__()

        self.en_vocab_size = en_vocab_size
        self.de_vocab_size = de_vocab_size
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.num_heads = num_heads

        # Linear transformation used for embedding the input to the encoder
        self.enc_input_dense = nn.Linear(self.de_vocab_size, self.embed_dim)

        # (Learned) positional encoding to be added to the embedded encoder input
        self.enc_pos_enc = nn.Parameter(torch.zeros((1, self.max_len, self.embed_dim)))

        # List of hidden layers in the feed-forward network of different encoder blocks
        self.enc_increase_hidden = nn.ModuleList(
            [nn.Linear(self.embed_dim, self.embed_dim * 2) for _ in range(self.enc_layers)]
        )

        # List of output layers used in the feed-forward network of different encoder blocks
        self.enc_decrease_hidden = nn.ModuleList(
            [nn.Linear(self.embed_dim * 2, self.embed_dim) for _ in range(self.enc_layers)]
        )

        # List of final layer normalizations used in different encoder blocks
        self.enc_layer_norm = nn.ModuleList(
            [nn.LayerNorm(self.embed_dim) for _ in range(self.enc_layers)]
        )

        # List of Multi-head attention blocks used in different encoder blocks
        self.enc_att = nn.ModuleList(
            [
                MultiHeadAttention(self.num_heads, self.embed_dim)
                for _ in range(self.enc_layers)
            ]
        )

        # Linear transformation used for embedding the input to the decoder
        self.dec_input_dense = nn.Linear(self.en_vocab_size, self.embed_dim)

        # (Learned) positional encoding to be added to the embedded decoder input
        self.dec_pos_enc = nn.Parameter(torch.zeros((1, self.max_len + 1, self.embed_dim)))

        # List of hidden layers in the feed-forward network of different decoder blocks
        self.dec_increase_hidden = nn.ModuleList(
            [nn.Linear(self.embed_dim, self.embed_dim * 2) for _ in range(self.dec_layers)]
        )

        # List of output layers in the feed-forward network of different decoder blocks
        self.dec_decrease_hidden = nn.ModuleList(
            [nn.Linear(self.embed_dim * 2, self.embed_dim) for _ in range(self.dec_layers)]
        )

        # List of final layer normalizations used in different decoder blocks
        self.dec_layer_norm = nn.ModuleList(
            [nn.LayerNorm(self.embed_dim) for _ in range(self.dec_layers)]
        )

        # List of (first) Multi-head attention blocks used in different decoder blocks
        self.dec_att = nn.ModuleList(
            [
                MultiHeadAttention(self.num_heads, self.embed_dim)
                for _ in range(self.dec_layers)
            ]
        )

        # List of (second) Multi-head attention blocks used in different decoder blocks
        self.enc_dec_att = nn.ModuleList(
            [
                MultiHeadAttention(self.num_heads, self.embed_dim)
                for _ in range(self.dec_layers)
            ]
        )

        # Final fully connected layer converting the final decoder output to predictions over the output vocabulary
        self.decoder_final_dense = nn.Linear(self.embed_dim, self.en_vocab_size)

    def forward(self, x1, x2):
        """Implement the forward pass of the Transformer model.

        Args:
            x1: with shape (batch_size, self.max_len, self.de_vocab_size)
            x2: with shape (batch_size, self.max_len + 1, self.en_vocab_size)

        Returns:
            Tuple of:
                decoding: final output of the model,
                          with shape (batch_size, self.max_len + 1, self.en_vocab_size)
                attention: attention_weights of the last multi-head attention block in decoder,
                           with shape (batch_size, self.max_len + 1, self.max_len)
        """

        # Embed inputs to embed dimension
        # START TODO #############
        enc_input_emb = self.enc_input_dense(x1)
        dec_input_emb = self.dec_input_dense(x2)
        # END TODO #############

        # Add positional encodings
        # START TODO #############
        encoding = enc_input_emb + self.enc_pos_enc
        decoding = dec_input_emb + self.dec_pos_enc
        # END TODO #############

        # Loop over the encoder blocks
        for i in range(self.enc_layers):
            # Encoder Self-Attention
            # In the ith encoder block:
            # 1) Pass encoding through the MultiHeadAttention block
            # 2) Pass the output through the FeedForward block
            # FeedForward block consists of:
            # 1 - a Linear layer doubling the embedding size
            # 2 - a ReLU activation function
            # 3 - a Linear layer halving the embedding size
            # 3) Sum up the outputs of the previous two steps
            # 4) pass the result through the encoder layer-normalization
            # START TODO #############
            multi_head = MultiHeadAttention(self.num_heads,self.embed_dim)
            output, attention = multi_head(encoding,encoding,encoding)

            ##feedforward block
            feedforward = nn.Sequential(nn.Linear(self.embed_dim,2*self.embed_dim),
                                         nn.ReLU(),
                                         nn.Linear(self.embed_dim*2,self.embed_dim)
                                        )
            feed_out = feedforward(output)
            output += feed_out
            layer_norm = nn.LayerNorm(self.embed_dim)
            encoding_output = layer_norm(output)
            # END TODO #############

        # Without decoder blocks, attention would be None
        attention = None

        # Loop over the decoder blocks
        for i in range(self.dec_layers):
            # Decoder Self-Attention
            # Repeat the steps in the encoder, but with three main differences:
            # 1) In each decoder block, there are two successive MultiHeadAttention blocks
            # 2) The first MultiHeadAttention is masked (you therefore need to set mask=True)
            # 3) The three inputs to the second MultiHeadAttention block are mixed:
            # two coming from the encoder, and
            # one coming from the previous MultiHeadAttention block in the decoder
            # START TODO #############

            # Masked Multi Head
            masked_multi_head = MultiHeadAttention(self.num_heads,self.embed_dim)
            layer_1_multi_head_output, attention = masked_multi_head(decoding,mask=True)
            
            # Feedforward 
            feedforward = nn.Sequential(nn.Linear(self.embed_dim,self.embed_dim*2),
                                        nn.ReLU(),
                                        nn.Linear(self.embed_dim*2,self.embed_dim))
            layer_1_feed_out = feedforward(layer_1_multi_head_output)
            
            # Add and norm
            layer_1_out = layer_1_feed_out + layer_1_multi_head_output
            layer_norm = nn.LayerNorm(self.embed_dim)
            layer_1_decoding_output = layer_norm(layer_1_out)

            # second multi head
            multi_head = MultiHeadAttention(self.num_heads,self.embed_dim)
            layer_2_multi_head_output, attention = multi_head(encoding_output,encoding_output,layer_1_decoding_output)
            
            # Feedforward
            layer_2_feed_out = feedforward(layer_2_multi_head_output)
            
            # Add and norm
            layer_2_out = layer_2_feed_out + layer_2_multi_head_output
            decoding = layer_norm(layer_2_out)

            # END TODO #############

        # Map the embedding dimension of the decoder output back to self.en_vocab_size
        # START TODO #############
        final_linear = nn.Linear(self.embed_dim,self.en_vocab_size)
        decoding = final_linear(decoding)
        # END TODO #############

        return decoding, attention


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, num_heads, embed_dim):
        """
        Initialize Multi-Head Attention module

        Args:
            num_heads (int): number of attention heads.
            embed_dim (int): the dimension of the model.
        """
        super().__init__()

        # Number of heads
        self.num_heads = num_heads

        # The dimension of the model
        self.embed_dim = embed_dim

        # W_Q of all heads stacked together for efficiency of dimensionality
        # (num_heads * d_head, num_heads * d_head) which is equal to (embed_dim, embed_dim)
        self.W_query = nn.Parameter(
            torch.normal(mean=torch.zeros((self.embed_dim, self.embed_dim)), std=1e-2)
        )

        # W_K of all heads stacked together for efficiency of dimensionality
        # (num_heads * d_head, num_heads * d_head) which is equal to (embed_dim, embed_dim)
        self.W_key = nn.Parameter(
            torch.normal(mean=torch.zeros((self.embed_dim, self.embed_dim)), std=1e-2)
        )
        # W_V of all heads stacked together for efficiency of dimensionality
        # (num_heads * d_head, num_heads * d_head) which is equal to (embed_dim, embed_dim)
        self.W_value = nn.Parameter(
            torch.normal(mean=torch.zeros((self.embed_dim, self.embed_dim)), std=1e-2)
        )
        # W_O of all heads stacked together for efficiency of dimensionality
        # (num_heads * d_head, num_heads * d_head) which is equal to (embed_dim, embed_dim)
        self.W_output = nn.Parameter(
            torch.normal(mean=torch.zeros((self.embed_dim, self.embed_dim)), std=1e-2)
        )

        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, query, key, value, mask=False):
        # Chunk size is equal to the dimensionality of each head (d_head)
        chunk_size = int(self.embed_dim / self.num_heads)

        # Tuple of length equal to the attention heads, each containing the query tensor of
        # the corresponding head with dimensionality (1, max_len, d_head)
        multi_query = torch.matmul(query, self.W_query).split(
            split_size=chunk_size, dim=-1
        )
        # Queries tensor from all attention heads stacked together resulting in a torch.tensor
        # of shape (num_heads, 1, max_len, d_head)
        multi_query = torch.stack(multi_query, dim=0)

        # Tuple of length equal to the attention heads, each containing the key tensor of
        # the corresponding head with dimensionality (1, max_len, d_head)
        multi_key = torch.matmul(key, self.W_key).split(split_size=chunk_size, dim=-1)
        # Keys tensor from all attention heads stacked together resulting in a torch.tensor
        # of shape (num_heads, 1, max_len, d_head)
        multi_key = torch.stack(multi_key, dim=0)

        # Tuple of length equal to the attention heads, each containing the value tensor of
        # the corresponding head with dimensionality (1, max_len, d_head)
        multi_value = torch.matmul(value, self.W_value).split(
            split_size=chunk_size, dim=-1
        )
        # Values tensor from all attention heads stacked together resulting in a torch.tensor
        # of shape (num_heads, 1, max_len, d_head)
        multi_value = torch.stack(multi_value, dim=0)

        # Scaling factor used in scaled dot product attention formula calculated as
        # square root of d_head
        scaling_factor = torch.tensor(np.sqrt(multi_query.shape[-1]))
        # Scaled dot product attention matrix for all attention heads
        # of dimensionality (num_heads, 1, max_len, max_len)
        dotp = torch.matmul(multi_query, multi_key.transpose(2, 3)) / scaling_factor
        # attention weights of all attention heads of dimensionality
        # (num_heads, 1, max_len, max_len)
        attention_weights = F.softmax(dotp, dim=-1)

        if mask:
            attention_weights = attention_weights.tril()
            attention_weights = attention_weights / attention_weights.sum(
                dim=3, keepdim=True
            )

        # sum of values weighted by the attention weights for all attention heads of
        # dimensionality (num_heads, 1, max_len, d_head)
        weighted_sum = torch.matmul(attention_weights, multi_value)
        # Tuple of weighted sums for each attention head (total length = num_heads)
        # each entry has dimensionality (1, 1, max_len, dim_head)
        weighted_sum = weighted_sum.split(1, dim=0)
        # All weighted sums stacked together for efficiency this gives a torch.tensor
        # with dimensionality (max_len, num_heads * dim_head) which is equal to (max_len, embed_dim)
        weighted_sum = torch.cat(weighted_sum, dim=-1).squeeze()
        # Calculate output of multi-head attention module before layer normalization
        multi_head = torch.matmul(weighted_sum, self.W_output)

        # Sum multi-head attention plus query
        output = multi_head + query
        # Pass through layer normalization
        output = self.layer_norm(output)
        # Return output and attention weights
        return output, attention_weights
