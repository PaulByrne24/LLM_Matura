import torch
import torch.nn as nn

class EncoderDecoder:
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

    def encode(self, input_text):
        return [self.string_to_int[c] for c in input_text] # Creates a list, uses the string_to_int_func() function to convert the input to it's numerical value. 
                                                        # Iterates over that using the for loop until the whole string is complete
                                                        # x is the input, c is the individual characters in the input

    def decode(self, encoded_list):
        return ''.join(self.int_to_string[l] for l in encoded_list)

    def get_encoded_data(self):
        return self.data[:100]

file_path = "wizard_of_oz.txt"
encoder_decoder = EncoderDecoder(file_path) # saves the class into a variable with "file_path" as it's attribute
print(encoder_decoder.get_encoded_data())
