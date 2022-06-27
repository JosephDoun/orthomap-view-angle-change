#!/usr/bin/python3

from src.rasters import RasterIn, RasterOut
from scipy.ndimage import rotate;
from tqdm import tqdm

import numpy as np


angles = np.arange(0, 90, 5)
inimg = RasterIn("London/London_LandCoverMap_2012_1m.tif")

for angle in tqdm(angles):
    outimg = RasterOut(inimg, (angle, 10))
    rotation = 270 - angle
    print("Out:", outimg.XSize, outimg.YSize, outimg.tile_size,
                  outimg.XSize / outimg.tile_size,
                  outimg.YSize / outimg.tile_size)
    print("In:", inimg.XSize, inimg.YSize, inimg.tile_size,
                  inimg.XSize / inimg.tile_size,
                  inimg.YSize / inimg.tile_size)
    print("Diff XY:", outimg.XSize-inimg.XSize, outimg.YSize - inimg.YSize)
    print(outimg.overlaps_xy,
                  inimg.stride,
                  outimg.stride,
                  inimg.length,
                  outimg.length,
                  "padding per tile:", outimg.padded_space_xy)
    
    # for i in tqdm(range(len(inimg))):
        
    #     rot = rotate(inimg[i], rotation, order=0, reshape=True)
    #     unrot = rotate(rot, -rotation, order=0, reshape=False)
        
    #     try:
    #         outimg.write(i, unrot)
            
    #     except Exception as e:
    #         print("Out:", outimg.XSize, outimg.YSize, outimg.tile_size,
    #               outimg.XSize / outimg.tile_size,
    #               outimg.YSize / outimg.tile_size)
    #         print("In:", inimg.XSize, inimg.YSize, inimg.tile_size,
    #               inimg.XSize / inimg.tile_size,
    #               inimg.YSize / inimg.tile_size)
    #         print(outimg.overlaps_xy,
    #               inimg.stride,
    #               outimg.stride,
    #               inimg.length,
    #               outimg.length,
    #               "padding per tile:", outimg.padded_space_xy)
    #         raise Exception(e)
            
    