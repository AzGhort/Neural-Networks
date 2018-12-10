#### Libraries
import random
import string
import math
import backprop as bp
import numpy as np
from PIL import Image

class ImageRecognizer:
    def __init__(self, imagename, tiles_vert, tiles_horz, threshold):
        # image
        self.image = Image.open(imagename)
        self.image = self.image.convert('L')
        self.image_name = imagename
        # data
        self.basic_inputs = []
        self.basic_outputs = []
        self.inputs = []
        self.outputs = []
        # constants
        self.tiles_vert = tiles_vert
        self.tiles_horz = tiles_horz
        self.threshold = threshold
        # prepare data and initialize NN
        width, height = self.image.size
        tile_height = int(height/self.tiles_vert)
        tile_width = int(width/self.tiles_horz)
        self.prepare_data(tile_height, tile_width)
        self.initialize_neural_network(tile_height*tile_width)

    # region Neural network

    # set a new neural network
    def initialize_neural_network(self, N):
        self.NN = bp.NeuralNetwork([N, 100, N])   
   
    # train network
    def train_network(self, epochs, pos_iters, learning_rate):
        print("Learning of neural network has started...")
        print("---------------------------------------")
        err = self.get_total_mean_squared_error()
        print("Total mean squared error: {0}".format(err))
        for i in range(0, epochs):
            print("Epoch number {0} has begun.".format(i))
            j = 0
            for (in_batch, out_batch) in zip(self.inputs, self.outputs):
                # find the best version (e.g. the one with the lowest MSE)
                (x, y) = self.get_tile_representation(in_batch, out_batch)
                if (self.tile_empty(x)):
                    continue
                j = j + 1
                self.NN.SGD([(x, y)], pos_iters, 1, learning_rate)
            err = self.get_total_mean_squared_error()
            print("Total mean squared error: {0}".format(err))
            print("---------------------------------------")
        print("Learning of neural network completed.")
        print("---------------------------------------")
    
    # get tiles from outputs of NN for original tiles
    def get_network_output_tiles(self):
        outs = []
        for (in_batch, out_batch) in zip(self.inputs, self.outputs):
            # find the best version (e.g. the one with the lowest MSE)
            (x, y) = self.get_tile_representation(in_batch, out_batch)
            out = self.NN.feedforward(x)
            outs.append(self.get_image_from_vector(out))
        return outs

    # get MSE of original image
    def get_total_mean_squared_error(self):
        retval = 0
        for (in_batch, out_batch) in zip(self.inputs, self.outputs):
            MSEs = [self.get_vector_mean_squared_error(i, o) for (i, o) in zip(in_batch, out_batch)]
            retval = retval + min(MSEs)
        return retval

    # tests image identity (before and after NN)
    def test_image_identity(self):
        #self.show_image()
        out_tiles = self.get_network_output_tiles()
        im = self.reconstruct_image(out_tiles)
        im.show()
        im.save("nn_out_" + self.image_name)

    # get MSE of one vector
    def get_vector_mean_squared_error(self, in_vec, out_vec):
        out = self.NN.feedforward(in_vec)
        return np.sum(np.square(np.subtract(out_vec, out)))
    
    # returns the tile with the lowest MSE
    def get_tile_representation(self, in_b, out_b):
        MSEs = [self.get_vector_mean_squared_error(i, o) for (i, o) in zip(in_b, out_b)]
        min_ind = MSEs.index(min(MSEs))
        return (in_b[min_ind], out_b[min_ind])
    # endregion
    
    # region Image handling and data preparation

    # show input image
    def show_image(self):
        self.image.show()

    # is the tile empty (white)?
    def tile_empty(self, tile):
        vec = self.get_vector_from_image(tile)
        if (np.sum(vec) == 0):
            return True
        else: 
            return False

    # prepare data for the network
    def prepare_data(self, tile_height, tile_width):
        self.tiles = []
        width, height = self.image.size
        for i in range(0, height, tile_height):
            for j in range(0, width, tile_width):
                ins = []
                tile = (self.image.crop((i, j, i+tile_height, j+tile_width)))
                self.tiles.append(tile)
                self.basic_inputs.append(self.get_vector_from_image(tile))
                self.basic_outputs.append(self.get_vector_from_image(tile))
                ins.append(tile)
                self.append_shifted_tiles(ins, i, j, tile_height, tile_width) 
                #self.append_rotated_tiles(ins, tile)
                self.set_inputs_outputs(ins)
    
    # append shifted transforms of tiles
    def append_shifted_tiles(self, ins, i, j, tile_height, tile_width):
        # shift up
        ins.append(self.image.crop((i+1, j, i+tile_height+1, j+tile_width)))
        ins.append(self.image.crop((i+2, j, i+tile_height+2, j+tile_width)))
        # upper-right corner
        ins.append(self.image.crop((i+1, j+1, i+tile_height+1, j+tile_width+1)))
        ins.append(self.image.crop((i+2, j+1, i+tile_height+2, j+tile_width+1)))
        ins.append(self.image.crop((i+1, j+2, i+tile_height+1, j+tile_width+2)))
        ins.append(self.image.crop((i+2, j+2, i+tile_height+2, j+tile_width+2)))
        # upper-left corner
        ins.append(self.image.crop((i+1, j-1, i+tile_height+1, j+tile_width-1)))
        ins.append(self.image.crop((i+2, j-1, i+tile_height+2, j+tile_width-1)))
        ins.append(self.image.crop((i+1, j-2, i+tile_height+1, j+tile_width-2)))
        ins.append(self.image.crop((i+2, j-2, i+tile_height+2, j+tile_width-2)))
        # shift down
        ins.append(self.image.crop((i-1, j, i+tile_height-1, j+tile_width)))
        ins.append(self.image.crop((i-2, j, i+tile_height-2, j+tile_width)))
        # lower-right corner
        ins.append(self.image.crop((i-1, j+1, i+tile_height-1, j+tile_width+1)))
        ins.append(self.image.crop((i-2, j+1, i+tile_height-2, j+tile_width+1)))
        ins.append(self.image.crop((i-1, j+2, i+tile_height-1, j+tile_width+2)))
        ins.append(self.image.crop((i-2, j+2, i+tile_height-2, j+tile_width+2)))
        # lower-left corner
        ins.append(self.image.crop((i-1, j-1, i+tile_height-1, j+tile_width-1)))
        ins.append(self.image.crop((i-2, j-1, i+tile_height-2, j+tile_width-1)))
        ins.append(self.image.crop((i-1, j-2, i+tile_height-1, j+tile_width-2)))
        ins.append(self.image.crop((i-2, j-2, i+tile_height-2, j+tile_width-2)))
        # shift right
        ins.append(self.image.crop((i, j+1, i+tile_height, j+tile_width+1)))
        ins.append(self.image.crop((i, j+2, i+tile_height, j+tile_width+2)))
        # shift left
        ins.append(self.image.crop((i, j-1, i+tile_height, j+tile_width-1)))
        ins.append(self.image.crop((i, j-2, i+tile_height, j+tile_width-2)))    

    # append rotated transforms
    def append_rotated_tiles(self, ins, tile):
        ins.append(tile.rotate(90))
        ins.append(tile.rotate(180))
        ins.append(tile.rotate(270))
   
    # sets input/output vectors from given tiles
    def set_inputs_outputs(self, input_tiles):
        # input tiles
        in_batch = []
        out_batch = []
        for tile in input_tiles:
            #rotated = self.get_all_rotated_transforms(tile)
            in_batch.append(self.get_vector_from_image(tile))
            out_batch.append(self.get_vector_from_image(tile))
        self.inputs.append(in_batch)
        self.outputs.append(out_batch)

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
                if vector[i][j] > self.threshold:
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
name = "house.jpg"
parser = ImageRecognizer(name, 25, 25, 0.7)
parser.train_network(50, 10, 0.7)
parser.test_image_identity()