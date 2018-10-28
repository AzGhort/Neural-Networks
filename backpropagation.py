## HEAVILY INSPIRED BY https://github.com/mattm/simple-neural-network

import random
import math
import binascii
import string

class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    def get_output(self, inputs):
        self.inputs = inputs
        self.output = self.transmit(self.get_total_net_input())
        return self.output

    def get_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    # yⱼ = φ = 1 / (1 + e^(-zⱼ))
    def transmit(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    # δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    def get_pd_error_wrt_total_net_input(self, target_output):
        return self.get_pd_error_wrt_output(target_output) * self.get_pd_total_net_wrt_input();

    # mean square error
    def get_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    # ∂E/∂yⱼ = -(tⱼ - yⱼ)
    def get_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    # dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
    def get_pd_total_net_wrt_input(self):
        return self.output * (1 - self.output)

    # ∂zⱼ/∂wᵢ =  0 + 1 * xᵢw₁^(1-0) + 0 ... = xᵢ
    def get_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]

class NeuronLayer:
    def __init__(self, num_neurons, bias):
        # every bias is the same or random
        self.bias = bias if bias else random.random()
        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def to_string(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def push_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.get_output(inputs))
        return outputs

    def get_all_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs


class NeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hidden, num_hidden_layers, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.hidden_layers = []
        for i in range(num_hidden_layers):
            self.hidden_layers.append(NeuronLayer(num_hidden, hidden_layer_bias))
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)

        self.init_weights_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_to_output_layer_neurons(output_layer_weights)

    def init_weights_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layers[0].neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layers[0].neurons[h].weights.append(random.random())
                else:
                    self.hidden_layers[0].neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1
        for l in range(1, len(self.hidden_layers)):
            for h in range(len(self.hidden_layers[l].neurons)):
                for i in range(self.num_hidden):
                    if not hidden_layer_weights:
                        self.hidden_layers[l].neurons[h].weights.append(random.random())
                    else:
                        self.hidden_layers[l].neurons[h].weights.append(hidden_layer_weights[weight_num])
                    weight_num += 1

    def init_weights_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(self.num_hidden):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def to_string(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        for i in range(len(self.hidden_layers)):
            print('Hidden Layer')
            self.hidden_layer.to_string()
            print('------')
        print('* Output Layer')
        self.output_layer.to_string()
        print('------')

    def push_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layers[0].push_forward(inputs)
        for i in range(1, len(self.hidden_layers)):
            out = self.hidden_layers[i].push_forward(hidden_layer_outputs)
            hidden_layer_outputs = out
        return self.output_layer.push_forward(hidden_layer_outputs)

    def train(self, training_inputs, training_outputs):
        self.push_forward(training_inputs)

        # 1. Output neuron deltas
        pd_errors_wrt_output_neuron = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):
            pd_errors_wrt_output_neuron[o] = self.output_layer.neurons[o].get_pd_error_wrt_total_net_input(training_outputs[o])

        # 2. Hidden neuron deltas and update them
        pd_errors_wrt_hidden_neuron = [0] * self.num_hidden
        
        # last hidden layer
        d_error_wrt_hidden_neuron_output = 0
        for h in range(self.num_hidden):
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron[o] * self.output_layer.neurons[o].weights[h]
            pd_errors_wrt_hidden_neuron[h] = d_error_wrt_hidden_neuron_output * self.hidden_layers[-1].neurons[h].get_pd_total_net_wrt_input()
        for h in range(self.num_hidden):
            for w_ih in range(len(self.hidden_layers[-1].neurons[h].weights)):
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron[h] * self.hidden_layers[-1].neurons[h].get_pd_total_net_input_wrt_weight(w_ih)
                self.hidden_layers[-1].neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight
        # other hidden layers
        for l in range(len(self.hidden_layers)-1, 0, -1): 
            for h in range(self.num_hidden):
                d_error_wrt_hidden_neuron_output = 0
                for o in range(self.num_hidden):
                    d_error_wrt_hidden_neuron_output += pd_errors_wrt_hidden_neuron[o] * self.hidden_layers[l+1].neurons[o].weights[h]
                pd_errors_wrt_hidden_neuron[h] = d_error_wrt_hidden_neuron_output * self.hidden_layers[l].neurons[h].get_pd_total_net_wrt_input()
            for h in range(self.num_hidden):
                for w_ih in range(len(self.hidden_layers[l].neurons[h].weights)):
                    pd_error_wrt_weight = pd_errors_wrt_hidden_neuron[h] * self.hidden_layers[l].neurons[h].get_pd_total_net_input_wrt_weight(w_ih)
                    self.hidden_layers[l].neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

        # 3. Update output neuron weights
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                pd_error_wrt_weight = pd_errors_wrt_output_neuron[o] * self.output_layer.neurons[o].get_pd_total_net_input_wrt_weight(w_ho)
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight       
                
    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.push_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].get_error(training_outputs[o])
        return total_error

# word recognition, not used
class WordsParser:
    def __init__(self, output_classes, corpus):
        self.output_classes = output_classes
        self.corpus = corpus

    def train_network(input_text):
        pass

    def get_input_vector_from_text(in_text):
        vector = []
        for i in range(len(in_text)):
            if in_text[i] in corpus:
                vector.append(1)
            else:
                vector.append(0);
        return vector
            
class LetterParser:
    def __init__(self, N):
        self.keywords = self.generate_random_words(N)

    def train_network(self, network, iterations):
        for i in range(iterations):
            # train
            for j in range(len(self.keywords)):
                vector = self.get_input_vector_from_word(self.keywords[j])
                output_vector = [0]*len(self.keywords)
                output_vector[j] = 1;
                train_set = [[vector, output_vector]]
                network.train(vector, output_vector)
                print(round(network.calculate_total_error(train_set), 10))
        # check the trained one
        for j in range(len(self.keywords)):
            vector = self.get_input_vector_from_word(self.keywords[j])
            print([round(c, 4) for c in network.push_forward(vector)])
       
    def get_input_vector_from_word(self, word):
        binary = bin(int(binascii.hexlify(word.encode()),16))
        # print(binary[2:37])
        # print(len(binary))
        bits35 = binary[2:37]
        output = []
        for c in bits35:
            output.append(int(c))
        return output

    def generate_random_words(self, count):
        words = []
        for i in range(count):
            words.append(''.join(random.choice(string.ascii_lowercase) for _ in range(7)))
        return words    


### MAIN

'''
l = LetterParser(10)
nn = NeuralNetwork(35, 4, 1, 10) 
l.train_network(nn, 100);
   

'''
#nn = NeuralNetwork(2, 2, 1, 2, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35, output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)
nn = NeuralNetwork(2, 2, 3, 2)
for i in range(100):
    nn.train([0.05, 0.1], [0.01, 0.99])
    print(round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9))

