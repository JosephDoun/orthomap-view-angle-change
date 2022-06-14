
import logging
from unicodedata import name
import numpy as np
import os

from osgeo import gdal, gdal_array
from scipy.ndimage import rotate
from scipy.signal  import medfilt2d

# TODO temp
import torch
import matplotlib.pyplot as plt
# TODO temp

from multiprocessing import Process, RawArray
from tqdm import tqdm
from typing import Tuple


if __name__ == 'main':
    from argparser import args

logger = logging.getLogger("ProjectionTools.py");
logger.setLevel(logging.INFO);

logging.basicConfig(
    format='%(asctime)s %(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S %b%d'
)

"""

# TODO

Things to solve:

1. Output is handled by Image class
2. Same dimensions approximately as input Image.
3. Same GeoTransform and Projection of input Image.
4. Exception constitutes rounding due to resolution change.

"""


class Projector:
    def __init__(self) -> None:
        self.tile_size = args.ts;
    
    def __get_shared_array__(self, ref_array):
        shape = ref_array.shape
        pass
    
    def __do_line__(self, line):
        pass
    
    def __line_multiprocessing__(self, tile):
        pass
    
    def __do_tile__(self, tile, angle):
        """
        Tile handling process.
        
        # TODO
        """
        
        azim, zen      = angle
        rotation_angle = azim 
        
        "This is a copy"
        tile = rotate(tile, rotation_angle)
        
        "Do stuff with it"
        tile = self.__line_multiprocessing__(tile)
        
        "Rotate back to origin && avoid copy"
        tile[:, :] = rotate(tile, -rotation_angle)
        
        pass
    
    def __get_overlap(self, image, angle):
        """
        Use this to get the full output dimensions.
        Accounts for padded tiles and overlapping.
        
        # TODO
        # ORIGIN SHOULD NOT CHANGE.
        # THE ROTATION PADDING MUST BE UNDONE.
        # AT LEAST FROM THE WEST AND NORTH SIDES
        # OF THE IMAGE.
        # TODO
        # THIS METHOD WILL BE PASSED TO IMAGE CLASS.
        
        """

        # Get dimensions of key tiles
        fshape = rotate(image[0], angle=angle).shape
        pshape = rotate(
            image[image.length - image.vstride], angle=angle
        ).shape
        
        # Get the estimated total padding from rotation
        fpad = (fshape[0] - image.tile_size,
                fshape[1] - image.tile_size)
        ppad = (
                pshape[0] - image.tile_size,
                pshape[1] - image.tile_size
            )
        
        # Get left, right pad estimates for each dimension.
        fpad_h = (
             (fpad[0] // 2, fpad[0] - fpad[0] // 2),
             (fpad[1] // 2, fpad[1] - fpad[1] // 2)
            )

        xsize = (
            # Full tile shape minus right padding
            fshape[1] - fpad_h[1][1]
            # Times number of tiles per row
            ) * (image.vstride - 1) + pshape[1]
        
        ysize = (
            # Full tile shape minus right padding
            fshape[0] - fpad_h[0][1]
            # Times number of tiles per column
            ) * (image.length / image.vstride - 1) + pshape[0]

        return (ysize, xsize)
    
    def __get_padded_area__(self, image):
        pass
    
    def __move_origin__(self, image):
        pass
    
    def __rotate__(self, image, angle, cv=-1):
        return rotate(image, angle=angle, cval=cv)

    def __do_angle__(self, angle, tiles):
        for tile in tiles:
            self.__do_tile__(tile, angle)
        
    def main(self):
        
        dsm = Image(args.dsm)
        dsm = self.__format_array__(dsm)
        
        for angle in args.angles:
            self.__do_angle__(angle, dsm)
        
        return 0;


class Image:
    """
    Class to be reading large geo-images
    tile by tile through subscription.
    """
    
    __slots__ = ['out_x',
                 'out_y',
                 'handle',
                 'path',
                 'YSize',
                 'XSize',
                 'vstride',
                 'length',
                 'tile_size',
                 '__out_handle',
                 'name',
                 'dir',
                 '__xpad',
                 '__ypad']
    
    def __init__(self,
                 path: str,
                 block_size=1024):
        
        self.handle = gdal.Open(path, gdal.GA_ReadOnly)
        
        self.path = path
        self.name = os.path.split(path)[-1].split('.')[0]
        self.dir  = 'Results'
        
        os.makedirs(self.dir, exist_ok=True)
        
        self.YSize = self.handle.RasterYSize
        self.XSize = self.handle.RasterXSize
        
        self.vstride = -(-self.XSize // block_size)
        self.length  = self.vstride * -(-self.YSize // block_size)
        
        self.tile_size = block_size
        
        self.__out_handle = None
        
    def __len__(self):
        return self.length
    
    def set_out_handle(self, rel_path: str):
        self.__out_handle = self.__get_out_handle(rel_path)
    
    def __get_out_handle(self, rel_path: str):
        """
        In WRITE mode, it would be convenient if overlapping
        and output sizes could be figured out internally.
        
        Let the Projector class handle other more relevant things.
        
        It should be inferred using the cval from the processed
        array to be written, probably standardized to -1.
        
        That would make much more sense.
        """
        
        driver = self.__get_driver()
        params = self.__output_metadata()
        handle = driver.Create(
            rel_path,
            self.XSize,
            self.YSize,
            1,
            gdal.GDT_Int16
        )
        return handle
        
    def __output_metadata(self):
        overlap = None
        return overlap
    
    def __get_tile(self, idx, off_pad=(0, 0)):
        row = idx // self.vstride
        col = idx % self.vstride
        
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
    
    def write(self, idx, block: np.ndarray):
        band = self.__out_handle.GetRasterBand(1)
        xoff, yoff, _, _ = self.__get_tile(idx)
        band.WriteArray(
            block,
            xoff,
            yoff,
            callback=None,
            callback_data=None
        )
        
    def show(self, idx):
        array = self.__getitem__(idx)
        
        if array.shape[0] > 3:
            array = np.moveaxis(array[[0, 1, 2]], 0, -1) / array.max()
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.imshow(array)
        plt.show()
        plt.close('all')
        
        return self
    
    def __get_driver(self) -> gdal.Driver:
        return gdal.GetDriverByName("GTiff")


class Shadow_Projector:
    def cast_shadows(self, num_p: int):
        processes = []
        
        shared = self.shadows
        # This consumes memory uneccessarily
        # It will need to be substituted in the future.
        heights = self.azimuth_shift(self.bh_array)
        
        for i in range(num_p):
            p = Process(
                target=self.casting_process,
                args=(shared[:, i::num_p], heights[:, i::num_p]),
                name=str(i)
            )
            p.start()
            processes.append(p)
            
        for p in tqdm(processes,
                      f"Processing date {self.i}/{self.dates_length} ~ {self.date} "):
            p.join()
            
        return shared
    
    def casting_process(self, cast, bh):
        """
        Intermediate shadow-casting process.
        """
        
        logger.debug("""Angles (%f, %f)""" % (self.sun_angles['azimuth'],
                                              self.sun_angles['zenith']))

        for line in zip(cast.T, bh.T):
            self.shadow_algorithm(line)
    
    def shadow_algorithm(self, line_pair):
        """
        Cast a shadow line as if the
        azimuth angle is 0 degrees.
        """
        line, bh = line_pair
        mask = bh > 0
        theta = (90 - self.sun_angles['zenith']) * np.pi / 180
        
        while mask.any():
            
            # Jump to the index with value.
            idx = torch.where(mask)[0][0]
            
            # Get height at location.
            height = bh[idx]
            
            if height < line[idx]:
                """
                # TODO
                Move if location is already
                written successfully.
                
                This needs to be improved
                by jumping more than one location.
                
                # TODO
                # Room for improvement.
                """
                jump = 1
                line, bh = line[jump:], bh[jump:]
                mask = bh > 0
                continue
                        
            # Jump rows to location.
            line = line[idx:]
            bh = bh[idx:]
            
            # Calculate dislocation
            d = height / np.tan(theta)
            d = d.round().to(int)
            
            """
            The roof ends where the height changes ahead.
            """
            # Check where the rooftop ends.
            roof_top_end = torch.where(bh != height)[0]
            roof_top_end = roof_top_end[0] if roof_top_end.any() else 0
            
            # Add rooftop length to dislocation
            d += roof_top_end + 1
            """
            Check if anything is taller ahead.
            If there is an obstacle, adjust d.
            """
            # Check for obstacles within dislocation range.
            check_obstacle = torch.where(bh[:d] > height)[0]
            # Shorten dislocation if obstacle is found.
            d = d if not check_obstacle.any() else check_obstacle[0]
            
            """
            Adjust d incase shadow lands on higher ground (another building).
            """
            
            try:
                new_height = bh[d]
                if new_height < height:
                    d2 = new_height / np.tan(theta)
                    d2 = d2.round().to(int)
                    d -= d2
            except IndexError:
                pass
            
            # Write shadow + rooftop    
            line[:d] = height
            
            # Next location should be the edge of the rooftop plus one.
            end = roof_top_end + 1
            line, bh = line[end:], bh[end:]
            
            # Get a new mask.
            mask = bh > 0
        
        logger.debug("Finished a line: %d" % np.random.randint(0, 100000))
    
    def save_result(self):
        driver = gdal.GetDriverByName('GTiff')
        output = driver.Create(
                os.path.join(
                        self.out_folder,
                        f"Shadows_{self.date.strftime('%Y%m%dT%H%M%S')}_1m.tif"
                    ),
                self.file_handle.RasterXSize,
                self.file_handle.RasterYSize,
                1,
                gdal.GDT_Float32
            )
        raster = output.GetRasterBand(1)
        raster.WriteArray(
                self.shadows,
                0,
                0
            )
        raster.SetNoDataValue(0)
        raster.FlushCache()
        output.SetGeoTransform(self.file_handle.GetGeoTransform())
        output.SetProjection(self.file_handle.GetProjection())
    
    def get_shared_array(self):
        """
        Instantiate a RawArray object,
        copy the building heights on it
        and return a tensor view.
        """
        shape = self.bh_array.shape
        shared = torch.frombuffer(
                    RawArray("f", shape[0]*shape[1]),
                    dtype=torch.float32,
                ).reshape(shape)
        print("We are inside the shared array process")
        shared[:, :] = self.bh_array
        return shared
        
    # def azimuth_shift(self, img: torch.Tensor):
    #     """
    #     Turn the image as if the azimuth were zero.
    #     """
    #     return F.affine(
    #         img.unsqueeze(0),
    #         -self.sun_angles['azimuth'].item(),
    #         [0., 0.],
    #         1.,
    #         [0., 0.],
    #     ).squeeze(0)
    
    # def azimuth_unshift(self, img: torch.Tensor):
    #     """
    #     Turn the image back to the original azimuth.
    #     """
    #     return F.affine(
    #         img.unsqueeze(0),
    #         self.sun_angles['azimuth'].item(),
    #         [0., 0.],
    #         1.,
    #         [0., 0.],
    #     ).squeeze(0)
    
    def remove_footprints(self):
        """
        Remove the building footprints
        from the shadow array if they are
        taller than the shadows.
        """
        idx = torch.where(self.shadows <= self.bh_array)
        self.shadows[idx] = 0
        
    
    def binarize_shadows(self):
        """
        Binarize the shadows array,
        and remove padding.
        """
        self.shadows[self.shadows > 0] = 1
        self.shadows = torch.from_numpy(self.shadows)
        self.shadows = self.unpad(self.shadows).numpy()
    
    def clean_output(self, passes):
        for i in range(passes):
            self.shadows = medfilt2d(self.shadows)
    
    def batch_rotation(self):
        ...
    
    def main(self):
        logger.info(
            f"""
            
            Starting process for raster '{os.path.basename(self.args.i)}' 
            """
        )
        
        for i, date in enumerate(self.dates):
            self.i = i + 1
            self.date = date
            print("WE DON'T HAVE A SHARED ARRAY")
            self.shadows = self.get_shared_array()
            print("WE HAVE A SHARED ARRAY")
            self.sun_angles = self.sun_position(
                    date,
                    self.location
                )
            print("WE HAVE SUN ANGLES")
            print(self.shadows.shape)
            self.shadows[:, :] = self.azimuth_shift(self.shadows)
            print("WE SHIFTED THE SHARED ARRAY")
            self.cast_shadows(20)
            self.shadows[:, :] = self.azimuth_unshift(self.shadows)
            self.remove_footprints()
            self.clean_output(5)
            self.binarize_shadows()
            self.save_result()
            
            if self.args.visualize:
                self.inspect()
            
            del self.shadows
        
        logger.info("Finished.")

    def inspect(self):
        """
        Plot height raster and
        shadow raster for inspection.
        """
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        ax[0].imshow(self.unpad(self.bh_array), vmin=0, vmax=20)
        ax[0].set_title("Heights")
        ax[1].imshow(self.shadows, vmin=0, vmax=1)
        ax[1].set_title("Projected Shadows")
        plt.show()
        del fig, ax