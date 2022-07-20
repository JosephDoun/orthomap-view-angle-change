from typing        import Any, Tuple, Union
from osgeo         import gdal, gdal_array
from scipy.ndimage import rotate, median_filter
from argparser     import args
from logger        import logger
from legend        import *

import matplotlib.pyplot as plt
import numpy             as np
import os

"""

# BUG TODO FIX

# RasterOut class:
#
# Output size has shifts of 
# 1 pixel vertically and/or
# horizontally depending on
# rotation angle.
#
# Must fix to be reliable.

"""


MAPFOLDER  = "Results"
PRODFOLDER = "Products"


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
                 'geotrans',
                 'nodata',
                 '_Resampler']
    
    def __init__(self,
                 path: str,
                 block_size=1024):
        
        self.handle = gdal.Open(path, gdal.GA_ReadOnly)
        
        self.path   = path
        self.name   = os.path.split(path)[-1].split('.')[0]
        
        self.YSize  = self.handle.RasterYSize
        self.XSize  = self.handle.RasterXSize
        
        self.stride    = -(-self.XSize // block_size)
        self.length    = self.stride * -(-self.YSize // block_size)
        
        self.tile_size = block_size
        self.geotrans  = self.handle.GetGeoTransform()
        
        self.res       = (self.geotrans[1], -self.geotrans[-1])
        self.nodata    = self.handle.GetRasterBand(1).GetNoDataValue()
        
        self._Resampler = None
                
    def __len__(self):
        return self.length
    
    def __get_tile(self, idx):
        row   = idx // self.stride
        col   = idx % self.stride
        
        offx  = self.tile_size*col
        offy  = self.tile_size*row
        
        xsize = min(self.tile_size, self.XSize - col*self.tile_size)
        ysize = min(self.tile_size, self.YSize - row*self.tile_size)
        return offx, offy, xsize, ysize
    
    def __getitem__(self, idx):
        """
        Retrieve block through subscription.
        """
        (offx,
         offy,
         xsize,
         ysize) = self.__get_tile(idx)
        
        array   = gdal_array.LoadFile(self.path, offx, offy, xsize, ysize)
        array   = self.__pad_corners(array)
            
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
                ), constant_values=0
                          )
    
    def SetResampler(self, resampler):
        """
        NOT IMPLEMENTED/APPLICABLE.
        """
        self._Resampler = resampler
        self.geotrans   = (
            self.geotrans[0],
            self.geotrans[1] * resampler.ratio,
            self.geotrans[2],
            self.geotrans[3],
            self.geotrans[4],
            self.geotrans[5] * resampler.ratio
        )
    
    def show(self, idx):
        """
        Visualize corresponding tile/block.
        """
        array   = self.__getitem__(idx)
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.imshow(array)
        plt.show()
        plt.close('all')
        
        return self


class RasterOut(RasterIn):
    def __init__(self,
                 image: RasterIn,
                 angles: Tuple,
                 res: Tuple=(0, 0)) -> None:
        
        self.image     = image
        self.handle    = image.handle
        self.angles    = angles
        self.rotation  = 90 + angles[0]
        self.length    = image.length
        self.stride    = image.stride
        self.tile_size = self.__probe_tile_size()
        
        # (xpad_total, ypad_total)
        "# BUG could also be here -- most likely."
        self.padded_space_xy = self.__calc_padding()
        self.overlaps_xy     = (self.padded_space_xy[0] // 2,
                                self.padded_space_xy[1] // 2)
        
        # extract name of file
        self.name = os.path.split(image.path)[-1].split('.')[0]
        
        # Desired output resolution
        # To be removed. DEPRECATED.
        self.res  = res
        
        # Compute new geotransformation
        self.geotrans = self.__get_geotransform(self.overlaps_xy[0],
                                                self.overlaps_xy[1])
        
        os.makedirs("Results", exist_ok=True)

        self.XSize, self.YSize = self.__get_output_shape()
        self.__out_handle      = self.__get_out_handle(
            f"{MAPFOLDER}/{self.name}_{angles[0]}_{angles[1]}.tif"
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
        row       = idx // self.stride
        col       = idx % self.stride
        off_pad   = self.__get_offset_shift(idx)
        offx      = self.tile_size*col - col*off_pad[0]
        offy      = self.tile_size*row - row*off_pad[1]
        tile_size = self.tile_size
        
        xsize = min(tile_size, self.XSize - offx)
        ysize = min(tile_size, self.YSize - offy)
        
        """
        # BUG either in here
        # or in __handle_overlap()
        """
        
        return offx, offy, xsize, ysize
    
    def __getitem__(self, idx):
        (
            offx,
            offy,
            xsize,
            ysize
        
        ) = self.__get_tile(idx)
        
        array = gdal_array.LoadFile(self.path,
                                    offx, offy,
                                    xsize,
                                    ysize)
        return array 
    
    def __get_offset_shift(self, idx):
        """
        # TODO
        # NOT NEEDED -- TEMPORARY
        """
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
                           self.rotation,
                           # Rotation place is last 2 dimensions
                           axes=(-1, -2),
                           order=0).shape[-1]
        return tile_size
    
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
        
        # TODO
        # FIX UNUSED VARIABLES
        # TEST WITH ARBITRARY INPUT SIZES.
        
        """
        tile_size = self.tile_size
        orig_tile = self.image.tile_size
        
        length    = self.__len__()
        stride    = self.image.stride
        height    = length // stride
                
        xsize     = (tile_size - self.overlaps_xy[0]) * stride
        ysize     = (tile_size - self.overlaps_xy[1]) * height
        
        return xsize, ysize
    
    def __get_geotransform(self, xpad, ypad):
        geotrans = self.handle.GetGeoTransform()
        geotrans = (
            # Translate X
            geotrans[0] - geotrans[1] * xpad,
            # self.res is DEPRECATED and will be removed.
            float(self.res[0]) or geotrans[1],
            geotrans[2],
            # Translate Y
            geotrans[3] - geotrans[5] * ypad,
            geotrans[4],
            # self.res is DEPRECATED and will be removed.
            -float(self.res[1]) or geotrans[5]
        )
        return geotrans
    
    def __get_out_handle(self, rel_path: str) -> gdal.Dataset:
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
            options=['COMPRESS=LZW']
        )
        handle.SetGeoTransform(self.geotrans)
        handle.SetProjection(self.handle.GetProjection())
        handle.FlushCache()
        return handle
    
    def __get_driver(self) -> gdal.Driver:
        return gdal.GetDriverByName("GTiff")
    
    def write(self, idx, block: np.ndarray):
        """
        Write to file block by block.
        """

        band = self.band
        
        (xoff,
         yoff,
         _,
         _)   = self.__get_tile(idx)
        
        block = self.__handle_overlap(idx, block)
        block = self.__fill_holes    (block)
        block = self.__wall_norm     (block)
        
        band.WriteArray(
            block,
            xoff,
            yoff,
            callback=self.__register_block,
            callback_data=(idx, self.registered_blocks)
            # Ensure NEAREST NEIGHTBOR
        )
        self.__out_handle.FlushCache()
        
        """
        BOTTOM LEFT CORNER NOT WRITTEN PROPERLY WITH 1 TRY
        REWRITING FIXES THIS ISSUE. POTENTIAL GDAL BUG.
        
        WRITING AND FLUSHING TWICE SEEMS TO BE FIXING IT.
        """
        band.WriteArray(
            block,
            xoff,
            yoff,
            callback=self.__register_block,
            callback_data=(str(idx)+"_", self.registered_blocks)
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
        
        while not str(west_block)+"_" in self.registered_blocks:
            # Wait for tile dependencies.
            continue
        
        """TESTING SNIPPET"""
        oblock = self.__getitem__(west_block)
        
        # s = {433 - 24 * i for i in range(2)}
        # if idx in s:
        #     plt.imshow(oblock)
        #     plt.show()
        #     plt.imshow(block)
        #     plt.show()
                
        overlap_1 = oblock[
            :, -2*overlap:
        ]
        overlap_2 = block[:, :2*overlap]
                
        # if idx in s:
        #     plt.imshow(overlap_1);
        #     plt.show();
        #     plt.imshow(overlap_2);
        #     plt.show()
            
        overlap_2[overlap_2 == self.nodata] = overlap_1[overlap_2 == self.nodata]
        
        # if idx in s:
        #     plt.imshow(block)
        #     plt.show()
        
        return block
    
    def __north_overlap(self, idx, block, overlap):
        north_block = idx - self.stride
        
        while not str(north_block)+"_" in self.registered_blocks:
            # Wait for tile dependencies.
            continue
        
        oblock = self.__getitem__(north_block)

        # s = {432 - 24 * i for i in range(2)}        
        # if idx in s:
        #     plt.imshow(oblock)
        #     plt.show()
        #     plt.imshow(block)
        #     plt.show()
        
        overlap_1 = oblock[
            -2*overlap:, :
        ]
        overlap_2 = block[:2*overlap, :]
        
        # if idx in s:
        #     plt.imshow(overlap_1);
        #     plt.show();
        #     plt.imshow(overlap_2);
        #     plt.show()
            
        overlap_2[overlap_2 == self.nodata] = overlap_1[overlap_2 == self.nodata]
        
        # if idx in s:
        #     plt.imshow(block)
        #     plt.show()
        
        return block

    def __fill_holes(self, block: np.ndarray):
        """
        Handle hole-filling.
        """
        for i in {1, -1}:
            block[
            
            block == self.nodata
            
            ] = median_filter(
                
                block,
                size=(5, 5),
                origin=(i, i),
                mode='reflect'
                
            )[block == self.nodata]
        
        return block
    
    def __wall_norm(self, block: np.ndarray):
        """
        Remove building pixel discrepancies.
        """
        block[
        
            block == BUILDINGS
            
        ] = median_filter(
            
            block,
            (3, 3)
            
        )[block == BUILDINGS]
        return block
        
    def __cleanup(self):
        """
        Crop resulting image according to the initial
        bounding box coordinates of the input image.
        """        
        options = gdal.TranslateOptions(
            projWin=[self.image.geotrans[0],
                     self.image.geotrans[3],
                     self.image.geotrans[0] + self.image.XSize * self.image.geotrans[1],
                     self.image.geotrans[3] + self.image.YSize * self.image.geotrans[5]]
        )
        newname = f"{self.name}_{self.angles[0]}_{self.angles[1]}_.tif"
        
        gdal.Translate(
            f"{MAPFOLDER}/{newname}",
            self.__out_handle.GetDescription(),
            options=options
            )
        
        os.remove(self.__out_handle.GetDescription())
        return newname
        
    def __del__(self):
        "# TODO: Check if this works."
        name = self.__cleanup()
        ProductFormatter(name)
        

class ProductFormatter:
    
    """
    Class designed to produce
    the final product based on
    RasterOut instance.
    """
    
    os.makedirs(PRODFOLDER, exist_ok=True)
    
    def __init__(self, name: str) -> None:
        self.__name  : str = name
        self.__source: str = f"{MAPFOLDER}/{name}"
        self.__destin: str = f"{PRODFOLDER}/{name}"
        
        self.__handle: gdal.Dataset = gdal.Open(self.__source)
        self.__array : np.ndarray   = gdal_array.LoadFile(self.__source)
        self.__out_handle           = self.__get_product_handle()
        
        self.__populate_product()
        
    def __get_product_handle(self) -> gdal.Dataset:
        driver: gdal.Driver = gdal.GetDriverByName("GTiff")
        
        handle = driver.Create(
            self.__destin,
            self.__handle.RasterXSize,
            self.__handle.RasterYSize,
            len(np.unique(self.__array)) - 1,
            gdal.GDT_Float32,
            options=['COMPRESS=LZW']
        )
        
        handle.SetProjection  (self.__handle.GetProjection(  ))
        handle.SetGeoTransform(self.__handle.GetGeoTransform())
        handle.FlushCache     (                               )
        
        return handle
    
    def __populate_product(self):
        for i, v in enumerate(np.unique(self.__array)):
            if v == 0 :
                assert i == 0
                continue
            
            band: gdal.Band = self.__out_handle.GetRasterBand(i)
            band.SetDescription(LEGEND[v])
            band.WriteArray(
                (self.__array == v).astype(np.float32),
                0,
                0
            )
            
        self.__out_handle.FlushCache()
    
    def __rescale(self):
        options = gdal.WarpOptions(xRes=args.tr,
                                   yRes=args.tr,
                                   resampleAlg="average")
        gdal.Warp(self.__destin.replace("_", ""),
                  self.__destin,
                  options=options)
        os.remove(self.__destin)
    
    def __del__(self):
        self.__rescale()
    

class LandCoverCleaner:
    
    """
    Class to discard uncertainties and disagreements.
    Assumes identical metadata between pairs.
    """
    
    def __init__(self,
                 lcm: RasterIn,
                 dsm: RasterIn) -> None:
        self.nodata_lcm = lcm.nodata
        self.nodata_dsm = dsm.nodata
    
    def __call__(self,
                 lcm: np.ndarray,
                 dsm: np.ndarray,
                 **kwds: Any) -> Tuple[Union[np.ndarray,
                                             np.ndarray]]:
    
        """
        BEHAVIOR: Convert buildings w.o. height info
                  to paved surfaces class.
                  
        Get the intersection between LCM & DSM.
        
        REVERSELY:
        If non-elevated surfaces have elevation
        assume lidar error.
        """
        
        dsm                                  = median_filter(dsm, (5, 5))
        dsm[(dsm < 2)]                       = 0
        lcm[(dsm > 0) & (lcm != BUILDINGS)]  = BUILDINGS
        lcm[(lcm == BUILDINGS) & (dsm == 0)] = PAVED
                
        if self.nodata_dsm:
            lcm[dsm == self.nodata_dsm] = 0
            dsm[dsm == self.nodata_dsm] = 0
        
        if (dsm < 0).any():
            dsm[dsm < 0] = 0
        
        return lcm, dsm
