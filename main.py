with open("wizard_of_oz.txt", "r", encoding="utf-8") as f: #accesses the text file in a read format
    text = f.read() # saves the text in a variable

chars = sorted(set(text)) # grabs every instance of a character and then puts it in a string 

string_to_int = { ch:i for i,ch in enumerate(chars) } # creates a dict with all the chars saved, sorts a number for each letter with a for loop
int_to_string = { i:ch for i,ch in enumerate(chars)} # creates a dict with all the chars and nums saved, but reversed, sorts a number for each letter with a for loop

encode = lambda x: [string_to_int[c] for c in x] # Creates a list, uses the string_to_int_func() function to convert the input to it's numerical value. 
                                                # Iterates over that using the for loop until the whole string is complete
                                                # x is the input, c is the individual characters in the input

decode = lambda y: ''.join(int_to_string[l] for l in y) # successful

encoded = encode("Hello")

print(string_to_int) # successful

print(encoded)

print(decode(encoded)) # successful
