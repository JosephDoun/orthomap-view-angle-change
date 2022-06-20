from email.mime import image
from typing import Any, List, Tuple, Union
from osgeo import gdal, gdal_array
from scipy.ndimage import rotate
from scipy.signal import medfilt2d

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
                 'origin',
                 'res',
                 'geotrans']
    
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
        self.geotrans  = self.handle.GetGeoTransform()
        self.res = (self.geotrans[1], -self.geotrans[-1])
                
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
        array = self.__pad_corners(array)
        return array 
    
    def __pad_corners(self, array):
        if array.shape == (self.tile_size, self.tile_size):
            return array
        else:
            diff = (
                self.tile_size - array.shape[-2],
                self.tile_size - array.shape[-1]
            )
            return np.pad(array, (
                (0, diff[0]),
                (0, diff[1])
                )
                          )
    
    def show(self, idx):
        array = self.__getitem__(idx)
        
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
        
        # Desired output resolution
        self.res  = res
        
        # Compute new geotransformation
        self.geotrans = self.__get_geotransform(self.overlaps_xy[0],
                                                self.overlaps_xy[1])
        
        os.makedirs("Results", exist_ok=True)

        self.XSize, self.YSize = self.__get_output_shape()
        
        self.__out_handle = self.__get_out_handle(
            f"Results/{self.name}_{angles[0]}_{angles[1]}.tif"
            )
        self.band = self.__out_handle.GetRasterBand(1)
        self.path = self.__out_handle.GetDescription()

        self.registered_blocks = []
        self.nodata = 0
        
    def __len__(self):
        return super().__len__()
    
    def __get_tile(self, idx):
        """
        Calculate and return tile coordinates
        and dimensions.
        """
        row = idx // self.stride
        col = idx % self.stride
        
        off_pad = self.__get_offset_shift(idx)
        
        offx = self.tile_size*col - col*off_pad[0]
        offy = self.tile_size*row - row*off_pad[1]

        tile_size = self.tile_size
        
        xsize = min(tile_size, self.XSize - offx)
        ysize = min(tile_size, self.YSize - offy)
        
        print("Total cols:", self.XSize,
              "Total rows:", self.YSize,
              "tile_size:", tile_size,
              "offx, offy:", (offx, offy),
              "size: ", (xsize, ysize),
              "off_size: ", (self.XSize - offx, self.YSize - offy))
        
        return offx, offy, xsize, ysize
    
    def __getitem__(self, idx):
        offx, offy, xsize, ysize = self.__get_tile(idx)
        array = gdal_array.LoadFile(self.path, offx, offy, xsize, ysize)
        return array 
    
    def __get_offset_shift(self, idx):
        return (
            # X
            2 * self.overlaps_xy[0],
            # Y
            2 * self.overlaps_xy[1]
        )
    
    def __probe_tile_size(self):
        """
        Rotate top left corner block to azimuth
        and retrieve tile dimensions after padding.
        """
        tile_size = rotate(self.image[0],
                           self.angles[0],
                           # Rotation place is last 2 dimensions
                           axes=(-1, -2),
                           order=0).shape[-1]
        return tile_size
    
    def __pad(self, block: np.ndarray, idx, cv=0):
        shape = block.shape
        orig  = self.image[idx].shape
        
        shape_diff = (-(-(shape[0] - orig[0]) // 2),
                      -(-(shape[1] - orig[1]) // 2)) 
        
        diff  = (self.overlaps_xy[0] - shape_diff[0],
                 self.overlaps_xy[1] - shape_diff[1])

        """
        Shape_diff : Cols // 2 (605) gives left padding.
        This has to be equal to self.overlaps_xy[0], for
        offset calculations to work properly.
        
        Shape_diff : Rows // 2 (-11) gives top padding.
        This has to be equal to self.overlaps_xy[1], for
        offset calculations to work properly.
        """
        # (1013, 1013) (1024, 408) (-11, 605) (223, -393)
        
        pad   = (
            ((diff[0])*(diff[0] > 0), 0),
            ((diff[1])*(diff[1] > 0), 0)
        )
        
        return np.pad(block, pad, constant_values=cv)[-(diff[0])*(diff[0] < 0):,
                                                      -(diff[1])*(diff[1] < 0):]
    
    def __calc_padding(self):
        """
        Calculate padded space for full block.
        """
        return (
            self.tile_size - self.image.tile_size,
            self.tile_size - self.image.tile_size
        )
    
    def __get_output_shape(self):
        """
        Calculate and return output size
        based on measurements of 1 block.
        """
        tile_size = self.tile_size
        orig_tile = self.image.tile_size
        length = self.__len__()
        stride = self.image.stride
        height = length // stride
        
        xsize = self.image.XSize + 5*self.overlaps_xy[0]
        ysize = self.image.YSize + 5*self.overlaps_xy[1]
        return xsize, ysize
    
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
            # gdal.GDT_Float32
        )
        handle.SetGeoTransform(self.geotrans)
        handle.SetProjection(self.handle.GetProjection())
        return handle
    
    def __get_driver(self) -> gdal.Driver:
        return gdal.GetDriverByName("GTiff")
    
    def write(self, idx, block: np.ndarray):
        """
        Write to file block by block.
        """
        if block.shape[-1] != self.tile_size:
            print("::"*100)
            block = self.__pad(block, idx)
                     
        band = self.band
        xoff, yoff, _, _ = self.__get_tile(idx)
        
        block = self.__handle_overlap(idx, block)
        
        # block = np.where(block == 0, self.image.length - idx, block)
        block = medfilt2d(block, kernel_size=3)
        
        band.WriteArray(
            block,
            xoff,
            yoff,
            callback=self.__register_block,
            callback_data=(idx, self.registered_blocks)
            # Ensure NEAREST NEIGHTBOR
        )
        
        self.__out_handle.FlushCache()
    
    def __register_block(self, *args):
        progress      = args[0]
        idx, register = args[-1]
        if progress == 1.0 and not idx in register:
            register.append(idx)
        
    def __handle_overlap(self, idx, block):

        xoff, yoff, _, _ = self.__get_tile(idx)
        print(self.__get_tile(idx))
        if xoff:
            """
            Retrieve west tile and overwrite
            overlapping part.
            """
            block = self.__west_overlap(idx, block,
                                        self.overlaps_xy[0])
        
        if yoff:
            """
            Retrieve north tile and overwrite
            overlapping part.
            """
            block = self.__north_overlap(idx, block, self.overlaps_xy[1])
            
        return block
    
    def __west_overlap(self, idx, block, overlap):
        west_block = idx - 1
        
        assert west_block in self.registered_blocks, "Oops, west."
        
        """TESTING SNIPPET"""
        oblock = self.__getitem__(west_block)
        
        s = {432 - 24 * i for i in range(2)}
        
        if idx in s:
            plt.imshow(oblock)
            plt.show()
            plt.imshow(block)
            plt.show()
        
        # print("*****WEST Reading Shape: ", oblock.shape)
        # print("*****WEST Writing Shape: ", block.shape)
                
        overlap_1 = oblock[
            :, -2*overlap:
        ]
        overlap_2 = block[:, :2*overlap]
        
        # print("*****WEST OVERLAP SHAPES:", overlap_1.shape, overlap_2.shape)
        
        if idx in s:
            plt.imshow(overlap_1);
            plt.show();
            plt.imshow(overlap_2);
            plt.show()
            
        overlap_2[overlap_2 == self.nodata] = overlap_1[overlap_2 == self.nodata]
        
        if idx in s:
            plt.imshow(block)
            plt.show()
        
        return block
    
    def __north_overlap(self, idx, block, overlap):
        north_block = idx - self.stride
        assert north_block in self.registered_blocks, "Opps, north."
        s = {433 - 24 * i for i in range(2)}
        oblock = self.__getitem__(north_block)
        # print("*****NORTH Reading Shape: ", oblock.shape)
        # print("*****NORTH Writing Shape: ", block.shape)
        
        if idx in s:
            plt.imshow(oblock)
            plt.show()
            plt.imshow(block)
            plt.show()
            
        oblock = oblock[
            :, :
        ]
        # print("*****NORTH DIFFERENCES:", difference)
        overlap_1 = oblock[
            -2*overlap:, :
        ]
        # overlap_1 = overlap_1[:, :block.shape[1]]
        overlap_2 = block[:2*overlap, :]
        # print("*****NORTH OVERLAP SHAPES:", overlap_1.shape, overlap_2.shape)
        
        if idx in s:
            plt.imshow(overlap_1);
            plt.show();
            plt.imshow(overlap_2);
            plt.show()
            
        overlap_2[overlap_2 == self.nodata] = overlap_1[overlap_2 == self.nodata]
        
        if idx in s:
            plt.imshow(block)
            plt.show()
            
        return block


class DownSampler:
    def __init__(self, image: RasterIn, res: List[Union[float, float]]) -> None:
        pass
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
    