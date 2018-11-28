#### Libraries
# Standard library
import random
import string
import binascii
import numpy as np
import backprop as bp    

class LetterParser:
    def __init__(self, N):
        self.keywords = self.generate_random_words(N)
        self.N = N

    def set_keywords(self, keywords):
        self.keywords = keywords

    def get_negative_keywords(self, keyword, N):
        set = []
        for i in range(N):
            edit = list(keyword)
            index = random.randint(0,6)
            edit[index] = random.choice(string.ascii_lowercase)
            input_vector = self.get_input_vector_from_word("".join(edit))
            output_vector = [0]*self.N
            output_vector = np.array(output_vector)
            output_vector.shape = (self.N, 1)
            set.append((input_vector, output_vector))
        return set

    def get_train_set(self):
        train_set=[]
        for j in range(len(self.keywords)):
            vector = self.get_input_vector_from_word(self.keywords[j])
            output_vector = [0]*self.N
            output_vector[j] = 1
            output_vector = np.array(output_vector)
            output_vector.shape = (self.N, 1)
            train_set.append((vector, output_vector))
        return train_set
       
    def get_input_vector_from_word(self, word):
        binary = bin(int(binascii.hexlify(word.encode()),16))
        bits35 = binary[2:37]
        output = []
        for c in bits35:
            output.append(int(c))
        output = np.array(output)
        output.shape = (35, 1)
        return output

    def generate_random_words(self, count):
        words = []
        for i in range(count):
            words.append(''.join(random.choice(string.ascii_lowercase) for _ in range(7)))
        return words    

    def test_word(self, word, nn):
        vector = self.get_input_vector_from_word(word)
        out = [str(round(num[0], 6)) for num in nn.feedforward(vector)]
        print("-------------------------------------")
        print("Testing word \"{0}\".".format(word))
        print("Raw output from neural network: {0}.".format(out))
        print("Word classified as keyword number: {0}".format(nn.test_against_threshold(vector, 0.9)))
        print("-------------------------------------")
        return out

    def train_network(self, nn, epochs, pos_iters, neg_words):
        train_set = self.get_train_set()
        for i in range(epochs):
            print("Epoch number {0} has begun.".format(i))
            j = 0
            for (x, y) in train_set:
                neg = self.get_negative_keywords(self.keywords[j], neg_words)
                print("Training keyword \"{0}\" and its negative variants...".format(self.keywords[j]))
                nn.SGD([(x, y)], pos_iters, 1, 0.5)
                nn.SGD(neg, 1, neg_words, 0.5)
                j = j + 1

### MAIN
l = LetterParser(5)
l.set_keywords(['aaaaaaa', 'bbbbbbb', 'ccccccc', 'ddddddd', 'eeeeeee'])
train_set = l.get_train_set()
nn = bp.NeuralNetwork([35, 17, 5])
l.train_network(nn, 100, 15, 20)
for word in l.keywords:
    l.test_word(word, nn)
inp = ''
while (inp != 'exit'):
    inp = input('>     Please enter word (of length 7) to classify by neural network.\n')
    if (len(inp) != 7):
        print('Given word has not length 7!')
    else:
        l.test_word(inp, nn)
