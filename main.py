with open("wizard_of_oz.txt", "r", encoding="utf-8") as f: #accesses the text file in a read format
    text = f.read() # saves the text in a variable

chars = sorted(set(text)) # grabs every instance of a character and then puts it in a string 

def string_to_int_func(): # function to enumerate/tokenize each character from chars
    string_to_int = {}
    for i, ch in enumerate(chars):
        string_to_int[ch] = i

    return string_to_int

def int_to_string_func(): # function to return enumerated characters back to regular letters
    int_to_string = {}
    for ch, i in enumerate(chars):
        int_to_string[i] = ch

    return int_to_string

encode = lambda x: [string_to_int_func()[c] for c in x] # Creates a list, uses the string_to_int_func() function to convert the input to it's numerical value. 
                                                        # Iterates over that using the for loop until the whole string is complete
                                                        # x is the input, c is the individual characters in the input

decode = lambda y: ''.join(int_to_string_func()[l] for l in y)

encoded = encode("Hello")

print(string_to_int_func())

print(encoded)

print(decode(encoded))
