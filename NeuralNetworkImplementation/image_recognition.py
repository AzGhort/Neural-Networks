#### Libraries
import random
import string
import math
import backprop as bp
import numpy as np
from PIL import Image, ImageChops

class ImageRecognizer:


    def __init__(self, imagename, tiles_vert, tiles_horz, threshold, rotate):
        # image
        self.image = Image.open(imagename)
        self.image = self.image.convert('L')
        self.image_name = imagename
        # data
        self.inputs = []
        self.outputs = []
        # constants
        self.tiles_vert = tiles_vert
        self.tiles_horz = tiles_horz
        self.threshold = threshold
        # prepare data and initialize NN
        self.rotate = rotate
        width, height = self.image.size
        tile_height = int(height/self.tiles_vert)
        tile_width = int(width/self.tiles_horz)
        self.prepare_data(tile_height, tile_width)
        self.NN = bp.NeuralNetwork([tile_height*tile_width, tile_height*tile_width, tile_height*tile_width])

    # region Neural network
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
                ((x, i1), (y, i2)) = self.get_tile_representation(in_batch, out_batch)
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
        diff = 0
        for (in_batch, out_batch) in zip(self.inputs, self.outputs):
            # find the best version (e.g. the one with the lowest MSE)
            ((x, i1), (y, i2)) = self.get_tile_representation(in_batch, out_batch)
            out = self.NN.feedforward(x)
            im = self.get_image_from_vector(out)
            if (self.rotate):
                rot = i1 % 4
                im = im.rotate(-90*rot)
            outs.append(im)
            #dif = sum([1 if (np.array_equal(im, v)) else 0 for v in outs])
            #if (dif == 0):
            #    diff = diff + 1
        return (outs, diff)

    # get MSE of original image
    def get_total_mean_squared_error(self):
        retval = 0
        for (in_batch, out_batch) in zip(self.inputs, self.outputs):
            MSEs = [self.get_vector_mean_squared_error(i, o) for (i, o) in zip(in_batch, out_batch)]
            retval = retval + min(MSEs)
        return retval

    # tests image identity (before and after NN)
    def test_image_identity(self):
        (out_tiles, diff) = self.get_network_output_tiles()
        #print(diff)
        im = self.reconstruct_image(out_tiles)
        im.show()
        im.save("nn_out_" + self.image_name)

    # get MSE of one vector
    def get_vector_mean_squared_error(self, in_vec, out_vec):
        (vec1, index1) = in_vec
        (vec2, index2) = out_vec
        out = self.NN.feedforward(vec1)
        return np.sum(np.square(np.subtract(vec2, out)))
    
    # returns the tile with the lowest MSE
    def get_tile_representation(self, in_b, out_b):
        MSEs = [self.get_vector_mean_squared_error(i, o) for (i, o) in zip(in_b, out_b)]
        min_ind = MSEs.index(min(MSEs))
        return (in_b[min_ind], out_b[min_ind])
    # endregion
    
    # region Image handling and data preparation
    # test image equality
    def images_equal(self, im1, im2):
        return ImageChops.difference(im1, im2).getbbox() is None
    
    # show input image
    def show_image(self):
        self.image.show()

    # is the tile empty (white)?
    def tile_empty(self, tile):
        vec = self.get_vector_from_image(tile)
        if (np.sum(vec) == 0):
            return False
        else: 
            return True

    # prepare data for the network
    def prepare_data(self, tile_height, tile_width):
        width, height = self.image.size
        counter = 0
        for i in range(0, height, tile_height):
            for j in range(0, width, tile_width):
                ins = []
                tile = (self.image.crop((i, j, i+tile_height, j+tile_width)))
                ins.append(tile)
                if (self.rotate):
                    self.append_rotated_tiles(ins, tile)
                    self.set_inputs_outputs(ins, counter)
                    counter = counter + 4
                else:
                    self.append_shifted_tiles(ins, i, j, tile_height, tile_width)
                    self.set_inputs_outputs(ins, counter)
                    counter = counter + 25
    
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
    def set_inputs_outputs(self, input_tiles, counter):
        # input tiles
        in_batch = []
        out_batch = []
        inner_counter = counter
        for tile in input_tiles:
            vec = self.get_vector_from_image(tile)
            in_batch.append((vec, inner_counter))
            out_batch.append((vec, inner_counter))
            inner_counter = inner_counter + 1
            #dif = sum([1 if (np.array_equal(v, vec)) else 0 for v in self.all_ins])
            #if (dif > 0):
            #    self.diff_tiles = self.diff_tiles + 1
            #self.all_ins.append(vec)
        self.inputs.append(in_batch)
        self.outputs.append(out_batch)
        pass
    
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
name = "pig.jpg"
parser = ImageRecognizer(name, 25, 25, 0.7, False)
parser.train_network(30, 10, 0.7)
parser.test_image_identity()