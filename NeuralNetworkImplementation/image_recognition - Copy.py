#### Libraries
import random
import string
import math
import backprop as bp
import numpy as np
from PIL import Image

class ImageRecognizer:
    def __init__(self, imagename, tiles_vert, tiles_horz):
        self.image = Image.open(imagename)
        self.image = self.image.convert('L')
        self.basic_inputs = []
        self.basic_outputs = []
        self.inputs = []
        self.outputs = []
        self.tiles_vert = tiles_vert
        self.tiles_horz = tiles_horz
        self.prepare_data()
        self.initialize_neural_network()

    # region Neural network

    # set a new neural network
    def initialize_neural_network(self):
        # length of input/output vector
        vec_len = len(self.inputs[0])
        self.NN = bp.NeuralNetwork([vec_len, 2500, vec_len])

    # train network for given number of epochs
    # orig_tile_iters = iterations to train the original non-transformed tile
    def train_network(self, epochs, orig_tile_iters, learning_rate):
        print("Learning of neural network has started...")
        print("---------------------------------------")
        err = self.get_mean_squared_error(self.inputs, self.outputs)
        print("Mean squared error: {0}".format(err))
        j = 0
        for (x, y) in zip(self.inputs, self.outputs):
            iters = 1
            # every 36th vector is the original tile
            if (j % 36 == 0):
                print("Training tile number {0} and its transformations...".format(int(j/36)))
                iters = orig_tile_iters
            j = j + 1
            for i in range(0, epochs):
                print("Epoch number {0} has begun.".format(i))
                self.NN.SGD([(x, y)], iters, 1, learning_rate)
            err = self.get_mean_squared_error(self.inputs, self.outputs)
            print("Mean squared error: {0}".format(err))
            print("---------------------------------------")
        print("Learning of neural network completed.")
        print("---------------------------------------")
   
    # train network on basic inputs
    def train_network_basic(self, epochs, pos_iters, learning_rate):
        print("Learning of neural network has started...")
        print("---------------------------------------")
        err = self.get_mean_squared_error(self.basic_inputs, self.basic_outputs)
        print("Mean squared error: {0}".format(err))
        for i in range(0, epochs):
            print("Epoch number {0} has begun.".format(i))
            j = 0
            for (x, y) in zip(self.basic_inputs, self.outputs):
                #print("Training tile number {0}...".format(j))
                j = j + 1
                self.NN.SGD([(x, y)], pos_iters, 1, learning_rate)
            err = self.get_mean_squared_error(self.basic_inputs, self.basic_outputs)
            print("Mean squared error: {0}".format(err))
            print("---------------------------------------")
        print("Learning of neural network completed.")
        print("---------------------------------------")
    
    # get tiles from outputs of NN for original tiles
    def get_network_output_tiles(self):
        outs = []
        i = 0
        for vec in self.basic_inputs:
            out = self.NN.feedforward(vec)
            print("Raw output of NN for original tile number {0}".format(i))
            print(out.tolist())
            outs.append(self.get_image_from_vector(out))
            i = i + 1
        return outs

    # get mean squared error of the network
    def get_mean_squared_error(self, ins, outs):
        N = len(self.inputs)
        s = 0
        for (input, des_out) in zip(ins, outs):
            actual_out = np.array([(1 if (o > 0.5) else 0) for o in self.NN.feedforward(input)])
            squares = np.square((des_out - actual_out))
            s = s + np.sum(squares)
        return s*1.0/N

    # tests image identity (before and after NN)
    def test_image_identity(self):
        self.show_image()
        out_tiles = self.get_network_output_tiles()
        im = self.reconstruct_image(out_tiles)
        im.show()
        #im.save("nn_out.jpg")

    # endregion
    
    # region Image handling and data preparation

    # show input image
    def show_image(self):
        self.image.show()

    # prepare data for the network
    def prepare_data(self):
        self.tiles = []
        width, height = self.image.size
        tile_height = int(height/self.tiles_vert)
        tile_width = int(width/self.tiles_horz)
        for i in range(0, height, tile_height):
            for j in range(0, width, tile_width):
                ins = []
                # basic
                tile = (self.image.crop((i, j, i+tile_height, j+tile_width)))
                ins.append(tile)
                self.tiles.append(tile)
                self.basic_inputs.append(self.get_vector_from_image(tile))
                self.basic_outputs.append(self.get_vector_from_image(tile))
                # shift up
                ins.append(self.image.crop((i+1, j, i+tile_height+1, j+tile_width)))
                ins.append(self.image.crop((i+2, j, i+tile_height+2, j+tile_width)))
                # shift down
                ins.append(self.image.crop((i-1, j, i+tile_height-1, j+tile_width)))
                ins.append(self.image.crop((i-2, j, i+tile_height-2, j+tile_width)))
                # shift right
                ins.append(self.image.crop((i, j+1, i+tile_height, j+tile_width+1)))
                ins.append(self.image.crop((i, j+2, i+tile_height, j+tile_width+2)))
                # shift left
                ins.append(self.image.crop((i, j-1, i+tile_height, j+tile_width-1)))
                ins.append(self.image.crop((i, j-2, i+tile_height, j+tile_width-2))) 
                self.set_inputs_outputs(ins, tile)
       
    # sets input/output vectors from given tiles
    def set_inputs_outputs(self, input_tiles, output_tile):
        # output tile
        out = self.get_vector_from_image(output_tile)
        # input tiles
        for tile in input_tiles:
            rotated = self.get_all_rotated_transforms(tile)
            self.inputs.append(self.get_vector_from_image(tile))
            self.outputs.append(out)
            for rot in rotated:
                self.inputs.append(self.get_vector_from_image(rot))
                self.outputs.append(out)

    # get three rotation transformed tiles
    def get_all_rotated_transforms(self, tile):
        transforms = []
        transforms.append(tile.rotate(90))
        transforms.append(tile.rotate(180))
        transforms.append(tile.rotate(270))
        return transforms
    
    # get a numpy column vector from image
    def get_vector_from_image(self, image):
        array = np.array(image)
        for i in range(len(array)):
            for j in range(len(array[0])):
                if array[i][j] > 200:
                   array[i][j] = 1
                else:
                   array[i][j] = 0
        # we need a column vector
        array.shape = (len(array)*len(array[0]), 1)
        return array

    # get a PIL image from numpy column vector
    def get_image_from_vector(self, vector):
        # get a square matrix from vector
        side = int(math.sqrt(len(vector)))
        vector.shape = (side, side)
        for i in range(side):
            for j in range(side):
                if vector[i][j] > 0.5:
                   vector[i][j] = 255
                else:
                   vector[i][j] = 0
        # convert back to image
        im = Image.fromarray(vector)
        return im

    # reconstructs image
    def reconstruct_image(self, tiles):
        width, height = self.image.size
        im = Image.new('L', (width, height))
        tile_height = int(height/self.tiles_vert)
        tile_width = int(width/self.tiles_horz)
        i = 0
        j = 0
        for tile in tiles:
            im.paste(tile, (i, j))
            j = j + tile_width
            # next row
            if (j == width):
                j = 0
                i = i + tile_height
        pass
        return im

    # endregion
    
### MAIN
parser = ImageRecognizer("house.jpg", 50, 50)
parser.train_network_basic(10, 5, 1)
parser.test_image_identity()