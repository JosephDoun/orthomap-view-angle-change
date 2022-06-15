from email.mime import image
from typing import Tuple
from osgeo import gdal, gdal_array
from scipy.ndimage import rotate

import matplotlib.pyplot as plt
import numpy as np
import os


class RasterIn:
    """
    Class to be reading large geo-images
    tile by tile through subscription.
    
    Class implementation also allows for
    block by block writing of processed image.
    """
    
    __slots__ = ['out_x',
                 'out_y',
                 'handle',
                 'path',
                 'YSize',
                 'XSize',
                 'stride',
                 'length',
                 'tile_size',
                 '__out_handle',
                 'name',
                 'dir',
                 '__xpad',
                 '__ypad',
                 'origin']
    
    def __init__(self,
                 path: str,
                 block_size=1024):
        
        self.handle = gdal.Open(path, gdal.GA_ReadOnly)
        
        self.path = path
        self.name = os.path.split(path)[-1].split('.')[0]
        
        self.YSize = self.handle.RasterYSize
        self.XSize = self.handle.RasterXSize
        
        self.stride = -(-self.XSize // block_size)
        self.length  = self.stride * -(-self.YSize // block_size)
        
        self.tile_size = block_size
                
    def __len__(self):
        return self.length
    
    def __get_tile(self, idx, off_pad=(0, 0)):
        row = idx // self.stride
        col = idx % self.stride
        
        offx = self.tile_size*col
        offy = self.tile_size*row
        
        xsize = min(self.tile_size, self.XSize - col*self.tile_size)
        ysize = min(self.tile_size, self.YSize - row*self.tile_size)
        return offx, offy, xsize, ysize
    
    def __getitem__(self, idx):
        offx, offy, xsize, ysize = self.__get_tile(idx)
        array = gdal_array.LoadFile(self.path, offx, offy, xsize, ysize)
        array[array < 0] = 0
        return array 
        
    def show(self, idx):
        array = self.__getitem__(idx)
        
        if array.shape[0] > 3:
            array = np.moveaxis(array[[0, 1, 2]], 0, -1) / array.max()
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.imshow(array)
        plt.show()
        plt.close('all')
        
        return self
    
    
class RasterOut(RasterIn):
    def __init__(self, image: RasterIn, angles: Tuple, res: Tuple=(0, 0)) -> None:
        self.image  = image
        self.handle = image.handle
        self.angles = angles
        self.length = image.length
        self.stride = image.stride
        
        self.tile_size = self.__probe_tile_size()
        
        # (xpad_total, ypad_total)
        self.padded_space_xy = self.__calc_padding()
        self.overlaps_xy     = (self.padded_space_xy[0] // 2,
                                self.padded_space_xy[1] // 2)
        
        # extract name of file
        self.name = os.path.split(image.path)[-1].split('.')[0]
        
        self.res  = res
        
        # Compute new geotransformation
        self.geotrans = self.__get_geotransform(self.overlaps_xy[0],
                                                self.overlaps_xy[1])
        
        os.makedirs("Results", exist_ok=True)

        self.XSize, self.YSize = self.__get_output_shape()
        
        self.out_handle = self.__get_out_handle(
            f"Results/{self.name}_{angles[0]}_{angles[1]}.tif"
            )
        
        self.path = self.out_handle.GetDescription()
        
        self.x_overlap_area = None
        self.y_overlap_area = None
        self.blocks_written = []
        
    def __len__(self):
        return super().__len__()
    
    def __get_tile(self, idx):
        row = idx // self.stride
        col = idx % self.stride
        
        off_pad = self.__get_offset_shift(idx)
                
        offx = self.tile_size*col - off_pad[0]
        offy = self.tile_size*row - off_pad[1]
        
        xsize = min(self.tile_size, self.XSize - col*self.tile_size)
        ysize = min(self.tile_size, self.YSize - row*self.tile_size)
        
        return offx, offy, xsize, ysize
    
    def __get_offset_shift(self, idx):
        return (
            # X
            (idx % self.stride) and self.overlaps_xy[0],
            # Y
            (idx // self.stride) and self.overlaps_xy[1]
        )
    
    def __probe_tile_size(self):
        """
        Rotate top left corner block to azimuth
        and retrieve tile dimensions after padding.
        """
        tile_size = rotate(self.image[0],
                           self.angles[0],
                           # Rotation place is last 2 dimensions
                           axes=(-1, -2)).shape[-1]
        return tile_size
    
    def __pad(self, block: np.ndarray):
        shape = block.shape
        diff  = (self.tile_size - shape[0],
                 self.tile_size - shape[1])
        pad   = (
            (diff[0] // 2, diff[0] - diff[0] // 2),
            (diff[1] // 2, diff[1] - diff[1] // 2)
        )

        assert shape[0] + pad[0][0] + pad[0][1] == self.tile_size[0]
        return np.pad(block, pad, constant_values=-1)
    
    def __calc_padding(self):
        """
        Calculate padded space.
        """
        return (
            self.tile_size - self.image.tile_size,
            self.tile_size - self.image.tile_size
        )
    
    def __get_output_shape(self):
        """
        # TODO
        # Make XSize, YSize more precise.
        """
        tile_size = self.tile_size
        orig_tile = self.image.tile_size
        length = len(self.image)
        stride = self.image.stride
        
        xsize = tile_size * stride
        ysize = tile_size * (int(length / stride))
        
        return ysize, xsize
    
    def __get_geotransform(self, xpad, ypad):
        geotrans = self.handle.GetGeoTransform()
        geotrans = (
            geotrans[0] - geotrans[1] * xpad,
            float(self.res[0]) or geotrans[1],
            geotrans[2],
            geotrans[3] - geotrans[5] * ypad,
            geotrans[4],
            -float(self.res[1]) or geotrans[5]
        )
        return geotrans
    
    def __get_out_handle(self, rel_path: str):
        """
        Return output handle for writing.
        """
        driver = self.__get_driver()
        handle = driver.Create(
            rel_path,
            self.XSize,
            self.YSize,
            1,
            gdal.GDT_Int16
        )
        return handle
    
    def __get_driver(self) -> gdal.Driver:
        return gdal.GetDriverByName("GTiff")
    
    def write(self, idx, block: np.ndarray):
        """
        Write to file block by block.
        """
        if block.shape != self.tile_size:
            block = self.__pad(block)
        
        band = self.__out_handle.GetRasterBand(1)
        
        xoff, yoff, _, _ = self.__get_tile(idx)
        
        block = self.__handle_overlap(idx, block) 
                
        band.WriteArray(
            block,
            xoff,
            yoff,
            callback=self.__register_block,
            callback_data=(idx,)
        )
        
    def __register_block(self, idx):
        self.blocks_written.append(idx)
        
    def __handle_overlap(self, idx, block):
        overlap = self.__get_offset_shift(idx)
        if overlap[0]:
            """
            Retrieve west tile and overwrite
            overlapping part.
            """
        
        if overlap[1]:
            """
            Retrieve north tile and overwrite
            overlapping part.
            """
        
        return block
    
    def __west_overlap(self, idx, block):
        pass
    
    def __north_overlap(self, idx, block):
        pass