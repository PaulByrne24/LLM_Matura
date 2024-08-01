import torch
import torch.nn as nn

class textTokenizer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.text = self.read_file()
        self.chars = sorted(set(self.text)) # grabs every instance of a character and then puts it in a string

        self.string_to_int = {ch: i for i, ch in enumerate(self.chars)} # creates a dict with all the chars saved, sorts a number for each letter with a for loop
        self.int_to_string = {i: ch for i, ch in enumerate(self.chars)} # creates a dict with all the chars and nums saved, but reversed, sorts a number for each letter with a for loop

        self.data = torch.tensor(self.encode(self.text), dtype=torch.long)

    def read_file(self):
        with open(self.file_path, "r", encoding="utf-8") as f: # accesses the text file in a read format
            return f.read() # saves the text in a variable

    def tokenize(self, input_text):
        return [self.string_to_int[c] for c in input_text] # Creates a list, uses the string_to_int_func() function to convert the input to it's numerical value. 
                                                        # Iterates over that using the for loop until the whole string is complete
                                                        # x is the input, c is the individual characters in the input

    def detokenize(self, encoded_list):
        return ''.join(self.int_to_string[l] for l in encoded_list)

    def get_tokens(self):
        return self.data[:100]

encoder_decoder = textTokenizer("wizard_of_oz.txt") # saves the class into a variable with the wizard of oz text
print(encoder_decoder.get_tokens())

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size # the size of the embedding (of the words)
        self.heads = heads # the amount of attention heads
        self.head_dim = embed_size // heads # sets the dimension of each attention head

        assert(self.head_dim * heads == embed_size), "the embed size needs to be divisible by the amount of heads" # To create a 2d input array, checks if the amount of attention heads and the size of the embeds are divisible

        # creates the variables values, key and queries as linear layers with the value of head_dim as it's input dim and exit dim. 
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        # A linear layer that combines the outputs from all attention heads back into the embedding size
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, value, key, query, mask):
        N = query.shape[0] # the number of rows of query
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1] # each one is the number of columns of value, key and query

        # Splits the embedding into self.heads pieces (each value in the embedding will be passed through the amount of attention heads)
        values = value.reshape(N, value_len, self.heads, self.head_dim)
        keys = key.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, key_len, self.heads, self.head_dim)

        # Calculate the dot product between queries and keys for each head, this then gives the attention energy score
        energy_score = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Applies a mask to the energy scores, setting masked positions to a very low value to ignore them in the softmax
        if mask != None:
            energy_score = energy_score.masked_fill(mask == 0, float("-1e22"))

        attention = torch.softmax(energy_score / (self.embed_size**(1/2)), dim=3) # uses the softmax function and divides the energy score with the square root of the embed size and applies it on the 3rd dim

        # calculates the dot product of the attention and the values variables, then reshapes the result to combine the resulting heads
        output = torch.einsum("nhql,nlhd->nqhd",[attention, values]).reshape(N, query_len, self.heads*self.head_dim)

        # applies the fc_out linear neural network and uses the arrays of "output" as it's input. Then returns the output of the NN.
        output = self.fc_out(output)
        return output
    
class TransEncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransEncoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)

        # two layers that normalize the inputs to make it easier and quicker for the transformer to work with the embeds
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)


        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size), # expands embedding size in a linear nn
            nn.ReLU(), # makes the model non-linear by making any negative value 0
            nn.Linear(forward_expansion*embed_size, embed_size) # reduces embedding size to original
            )

        self.dropout = nn.Dropout(dropout) # randomly sets neurons/nodes to 0 to make the model find other alternatives that aren't reliant on specific nodes

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        dropped = self.dropout(self.norm1(attention + query)) # drops a random amount of neurons in the first normalization
        forward = self.feed_forward(dropped) # puts the var dropped in the feed_forward sequence
        output = self.dropout(self.norm2(forward + dropped)) # drops random neurons of the second normalization 

        return output
    
class Encoder(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, vocab_size, length, device, num_layers):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size) # embedds the words with a vector in 3d space
        self.posit_embedding = nn.Embedding(length, embed_size) # saves the positions of the words in the original sentence
        self.layers = nn.ModuleList([TransEncoderBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, sequence_length = x.shape
        positions = torch.arange(0, sequence_length).expand(N, sequence_length).to(self.device)

        output = self.dropout(self.word_embedding(x) + self.posit_embedding(positions))

        for layer in self.layers:
            output = layer(output, output, output, mask) # output three times because the value, key and query are all the same value

        return output
    
class TransDecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(TransDecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.encoder_block = TransEncoderBlock(embed_size, heads, dropout, forward_expansion)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, orgnl_mask, trgt_mask):
        attention = self.attention(x, x, x, trgt_mask)
        query = self.dropout(self.norm1(attention + x))
        output = self.encoder_block(value, key, query, orgnl_mask)
        return output
    
class Decoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout, length):
        pass
