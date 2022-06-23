
import logging
import numpy as np
import os

from osgeo import gdal, gdal_array
from scipy.ndimage import rotate
from scipy.signal  import medfilt2d

# TODO temp
import torch
import matplotlib.pyplot as plt
# TODO temp

from multiprocessing import Process, Queue as pQueue, RawArray
from threading import Thread
from queue import Queue as tQueue, deque
from tqdm import tqdm
from typing import Any, Tuple
from rasters import LandCoverCleaner, RasterIn, RasterOut
from ctypes import c_uint8

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
        self.tile_size  = args.ts
        self.num_p      = args.c
        self.num_t      = args.threads
        self.threads    = []
        self.processes  = []
        self.p_queue    = pQueue()
        self.t_queue    = tQueue()
        self.lcmviewer  = LCMView()
        self.shadowcast = Shadow()
        self.prog_queue = tQueue()
        # tqdm iterable should come from parser
        self.progress   = tqdm(range(100),
                               desc="Progress:",
                               unit="No idea yet",
                               colour='RED')
        
        self.prog_queue.put(self.progress)
        
        for i in range(self.num_p):
            p = Process(target=self.__process,
                        name=f"Process_{i}",
                        args=(self.p_queue,))
            self.processes.append(
                p
            )
            p.start()
        
        for i in range(self.num_t):
            t = Thread(target=self.__thread,
                       name=f"Thread_{i}",
                       args=(self.t_queue,))
            self.threads.append(
                t
            )
            t.start()
        
    def __make_shared(self, ref_array: np.ndarray):
        """
        Pass array to shared memory before multiprocessing.
        """
        shape  = ref_array.shape
        shared = np.frombuffer(
            RawArray(c_uint8, shape[0]*shape[1]),
            dtype=ref_array.dtype
        ).reshape(shape)
        shared[:, :] = ref_array
        return shared
    
    def __thread(self, queue: tQueue):
        """
        
        # TODO
        # Implement None as termination flag?
        
        # Data loading should happen
        # within each thread.
        
        # Therefore tQueue should probably
        # only receive an index to a tile.
        
        """
        payload = queue.get()
        while not payload == None:
            """
            Load tiles and process.
            """
            
            idx, azim, zen, out = payload
            
            lcm_tile = self.lcm[idx]
            dsm_tile = self.dsm[idx]
            
            result = self.__do_tile(
                (lcm_tile, dsm_tile),
                (azim, zen)
            )
            
            "Write block to RasterOut instance"
            out.write(idx, result)
            
            queue.task_done()
            payload = queue.get()
        
        queue.task_done()
        queue.put(None)    
        
    def __process(self, queue: pQueue):
        while not queue.empty():
            pass
        queue.put(None)
        
    def __do_tile(self, tiles, angles):
        """
        Tile handling process.
        
        # TODO
        # Verify solidity of process.
        """
        lcm, dsm  = tiles
        azim, zen = angles
        
        """
        Calculate desired rotation angle,
        so as to bring azimuth to 270 degrees.
        """
        rotation  = 270 - azim
        
        "This is a copy"
        lcm, dsm = self.__rotate((lcm, dsm), rotation)
        lcm      = self.__make_shared(lcm)
        
        "Feed the multiprocessing queue"
        self.__feed_pQueue_n_wait((lcm, dsm), zen)
        
        "Rotate back to origin"
        lcm = self.__rotate((lcm,), -rotation, reshape=False)
        return lcm
    
    def __feed_pQueue_n_wait(self, tiles: Tuple[np.ndarray], zen):
        """
        Launch multiprocessing in here.
        But wait for processes before proceeding.
        
        Need to ensure processes return before
        thread goes on to write to disk.
        
        Need to signal the tile is done somehow.
        
        # TODO
        # Implement signaling that the tile is done.
        # Implement wait until signal.
        """
        for lines in zip(tiles):
            self.p_queue.put(lines, zen)
        self.p_queue.put(None)
    
    def __rotate(self, blocks: Tuple[RasterIn], angle, cv=0, reshape=True):
        rotated = []
        for block in blocks:
            rotated.append(
                rotate(block, angle=angle, cval=cv, reshape=reshape)
            )
        return rotated

    def __do_angles(self, angles):
        """
        Each angle has to produce its
        own RasterOut object for the map.
        
        I guess this should happen in here.
        
        Eventually, every running thread
        is desired to be doing its own writing.
        
        So it needs access to current
        tile's RasterOut object.
        
        # TODO
        # Implement RasterOut creation.
        """
        out = RasterOut(self.lcm, angles)
        for idx in range(len(self.lcm)):
            """
            Perhaps the RasterOut object
            should be fed through the Queue.
            
            # TODO
            # The above.
            """
            self.t_queue.put((idx, *angles, out))
    
    def main(self):
        
        self.dsm      = RasterIn(args.dsm)
        self.lcm      = RasterIn(args.lcm)
        self.cleaner  = LandCoverCleaner(self.lcm,
                                         self.dsm)
        dsm = self.dsm
        for angles in args.angles:
            self.__do_angles(angles, dsm)
        
        return 0;


class LCMView:
    def __init__(self) -> None:
        pass
    
    def __call__(self, line_pair, **kwds: Any) -> Any:
        """
        LCM = 0
        DSM = 1
        """
        lcm, dsm = line_pair
        mask = None
        while mask.any():
            pass


class Shadow:
    """
    
    Shadow casting algorithm:
    
    Cast shadows according to Sun Location and DSM.
    Reproject scene according to Observer Location and DSM.
    
    """
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