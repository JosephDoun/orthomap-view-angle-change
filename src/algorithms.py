
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import logging
logger = logging.getLogger()
from multiprocessing import RawArray
from scipy.signal import medfilt2d
from typing import Any
from osgeo import gdal


class LCMView:
    def __init__(self, res=1) -> None:
        self.res = res
    
    def __call__(self, lcm, dsm, zen, **kwds: Any) -> Any:
        """
        LCM = 0
        DSM = 1
        """
        elev = 90 - zen
        tan  = np.tan(elev * np.pi / 180)
        
        mask = dsm > 0
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
        
        # Why am I computing tan(theta) each time?
        # Compute it once here.
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