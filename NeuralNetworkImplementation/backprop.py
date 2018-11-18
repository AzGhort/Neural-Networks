#### Libraries
# Standard library
import random
import string
import binascii
import numpy as np

class NeuralNetwork:    

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta):
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        # (actual out - desired out)*(derivative of potential)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(self.test_against_threshold(x, 0.9), np.argmax(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def test_against_threshold(self, vector, threshold):
        out = [item for sublist in self.feedforward(vector).tolist() for item in sublist]
        s = sum(out)
        m = max(out)
        index = -1
        if (m > threshold):
            index = out.index(m)
        return index

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
            
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
        
# auxiliary math functions
def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
        return sigmoid(z)*(1-sigmoid(z))

### MAIN
l = LetterParser(5)
l.set_keywords(['aaaaaaa', 'bbbbbbb', 'ccccccc', 'ddddddd', 'eeeeeee'])
train_set = l.get_train_set()
nn = NeuralNetwork([35, 17, 5])
l.train_network(nn, 400, 15, 20)
for word in l.keywords:
    l.test_word(word, nn)
inp = ''
while (inp != 'exit'):
    inp = input('>     Please enter word (of length 7) to classify by neural network.\n')
    if (len(inp) != 7):
        print('Given word has not length 7!')
    else:
        l.test_word(inp, nn)
