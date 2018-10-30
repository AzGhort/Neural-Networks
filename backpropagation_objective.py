## HEAVILY INSPIRED BY https://github.com/mattm/simple-neural-network

import random
import math
import binascii
import string
import enum

class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    def init_weights(self, num, layer_type):
        if (layer_type == NeuronLayerType.INPUT):
            self.weights = [1 for i in range(num)]
        else:
            self.weights = [random.random() for i in range(num)]

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
    def __init__(self, num_neurons, bias, type):
        # every bias is the same or random
        self.bias = bias if bias else random.random()
        self.neurons = []
        self.layer_type = type
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def connect_layer(self, last_layer, next_layer, num_last_layer):
        count = len(last_layer.neurons) if last_layer else num_last_layer
        for i in range(len(self.neurons)):
            self.neurons[i].init_weights(count, self.layer_type)
        self.last_layer = last_layer
        self.next_layer = next_layer

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
        if (self.layer_type != NeuronLayerType.OUTPUT):
            return self.next_layer.push_forward(outputs)
        else:
            return outputs

    def get_all_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs

class NeuronLayerType:
    INPUT = 1,
    HIDDEN = 2,
    OUTPUT = 3

class NeuronLayerConfiguration:

    def __init__(self, num_neurons, bias):
        self.neurons = num_neurons
        self.bias = bias

class NeuralNetwork:
    LEARNING_RATE = 0.99

    def __init__(self, num_inputs, hidden_layer_configs, output_layer_config):
        self.num_inputs = num_inputs

        # initialize the layers
        # first layer
        first_layer_config = hidden_layer_configs[0]
        # (the actual input layer is not implemented, as it only outputs the original input vector)
        self.input_layer = NeuronLayer(first_layer_config.neurons, first_layer_config.bias, NeuronLayerType.HIDDEN)
        self.hidden_layers = [self.input_layer]
        last_hidden_layer = self.input_layer
        
        for i in range(1, len(hidden_layer_configs)):
            self.hidden_layers.append(NeuronLayer(hidden_layer_configs[i].neurons, hidden_layer_configs[i].bias, NeuronLayerType.HIDDEN))
            last_hidden_layer = self.hidden_layers[i]

        print(len(self.hidden_layers))
        # last layer
        self.output_layer = NeuronLayer(output_layer_config.neurons, output_layer_config.bias, NeuronLayerType.OUTPUT)

        # connect them    
        # special case - two layer network 
        if (len(self.hidden_layers) > 1):
            self.input_layer.connect_layer(None, self.hidden_layers[0], num_inputs)
            self.hidden_layers[len(self.hidden_layers)-1].connect_layer(self.hidden_layers[len(self.hidden_layers)-2], self.output_layer, hidden_layer_configs[len(self.hidden_layers)-2].neurons)
        else:
            self.input_layer.connect_layer(None, self.output_layer, num_inputs)
        # all other hidden layers
        for i in range(1, len(hidden_layer_configs)-1):
            self.hidden_layers[i].connect_layer(self.hidden_layers[i-1], self.hidden_layers[i+1], hidden_layer_configs[i-1].neurons)
        # output layer
        self.output_layer.connect_layer(last_hidden_layer, None, None)
               
    def to_string(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.input_layer.to_string()
        print('------')
        print('* Output Layer')
        self.output_layer.to_string()
        print('------')

    # recursive
    def push_forward(self, inputs):
        return self.input_layer.push_forward(inputs)

    def train(self, training_inputs, training_outputs):
        self.push_forward(training_inputs);

        # 1. Output neuron deltas
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):

            # ∂E/∂zⱼ
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].get_pd_error_wrt_total_net_input(training_outputs[o])

        # 2. Hidden neuron deltas
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.input_layer.neurons)
        for h in range(len(self.input_layer.neurons)):

            # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]

            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.input_layer.neurons[h].get_pd_total_net_wrt_input()

        # 3. Update output neuron weights
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):

                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].get_pd_total_net_input_wrt_weight(w_ho)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        # 4. Update hidden neuron weights
        for h in range(len(self.input_layer.neurons)):
            for w_ih in range(len(self.input_layer.neurons[h].weights)):

                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.input_layer.neurons[h].get_pd_total_net_input_wrt_weight(w_ih)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.input_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight      
                
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
                err = round(network.calculate_total_error(train_set), 10)
                print(err)
            #print('Iteration number ' + str(i))
            for j in range(len(self.keywords)):
                vector = self.get_input_vector_from_word(self.keywords[j])
                print([round(c, 4) for c in network.push_forward(vector)])
       
    def get_input_vector_from_word(self, word):
        binary = bin(int(binascii.hexlify(word.encode()),16))
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


l = LetterParser(3)
nn = NeuralNetwork(35, [NeuronLayerConfiguration(7, None), NeuronLayerConfiguration(7, None)], NeuronLayerConfiguration(3, None)) 
l.train_network(nn, 1000);


'''
nn = NeuralNetwork(2, [NeuronLayerConfiguration(2, 0.35)], NeuronLayerConfiguration(2, 0.6))
for i in range(100):
    nn.train([0.05, 0.1], [0.01, 0.99])
    print(round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9))
'''
