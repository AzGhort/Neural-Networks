#### Libraries
import random
import string
import backprop as bp
import numpy as np
from PIL import Image

class ImageParser:
    def __init__(self, imagename, tiles_vert, tiles_horz):
        self.image = Image.open(imagename)
        self.tiles = self.get_basic_tiles(tiles_vert, tiles_horz)

    def show_image(self):
        self.image.show()

    def get_all_tiles(self, tiles_vert, tiles_horz):
        tiles = []
        width, height = self.image.size
        tile_height = int(height/tiles_vert)
        tile_width = int(width/tiles_horz)
        for i in range(0, height, tile_height):
            for j in range(0, width, tile_width):
                # basic
                tiles.append(self.image.crop((i, j, i+tile_height, j+tile_width)))
                # shift up
                tiles.append(self.image.crop((i+1, j, i+tile_height+1, j+tile_width)))
                tiles.append(self.image.crop((i+2, j, i+tile_height+2, j+tile_width)))
                # shift down
                tiles.append(self.image.crop((i-1, j, i+tile_height-1, j+tile_width)))
                tiles.append(self.image.crop((i-2, j, i+tile_height-2, j+tile_width)))
                # shift right
                tiles.append(self.image.crop((i, j+1, i+tile_height, j+tile_width+1)))
                tiles.append(self.image.crop((i, j+2, i+tile_height, j+tile_width+2)))
                # shift left
                tiles.append(self.image.crop((i, j-1, i+tile_height, j+tile_width-1)))
                tiles.append(self.image.crop((i, j-2, i+tile_height, j+tile_width-2)))
                
   def get_all_rotated_transforms(self, tile):
       tile = []
       pass
    
### MAIN
parser = ImageParser("house.jpg", 10, 10)
parser.show_image()
