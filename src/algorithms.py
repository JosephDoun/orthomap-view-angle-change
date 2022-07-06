import numpy as np

from legend import *
from multiprocessing.dummy import current_process
from logger import logger

torch = None


class LandCover:
    """
        Land-cover view shifting algorithm.
    """
    
    """
    Relative elevation in meters
    to use as a threshold for individual
    objects that have the concept of walls.
    
    Meant to define individual roof parts.
    """
    UNITDIFF = 3
    
    def __init__(self,
                 res=1) -> None:
        self.res = res
        
    def __call__(self, lcm: np.ndarray, dsm: np.ndarray, zen: float) -> None:
        """
        Land-cover view shifting algorithm.
        """
        
        elev = 90 - zen
        tan  = np.tan(elev * np.pi / 180)
        mask = dsm > 0
        idx  = 0
        
        # logger.debug("hello.")

        while mask[idx:].any():
            
            "Jump ahead to point of interest."
            idx    += np.where(mask[idx:])[0][0]
            
            "Get info"
            (   
                # Height before.
                height_,
                # Current height.
                height,
                # Height after.
                _height
             )          = dsm[idx-1:idx+2]
            rel_height  = height - height_
            rel_height *= rel_height > 0
            higher      = np.where(dsm[idx:] > height + self.UNITDIFF)[0]
            cover       = lcm[idx]
            
            "Calculate dislocation."
            d = min(
                
                    # The geometric disclocation.
                    int(
                        round(
                            
                            # Relative height
                            rel_height / tan, 0
                            
                            )
                        ),
                    
                    # Or until the higher object.
                    higher[0] if higher.any() else 0 
                    
                )
            
            "Temp: Debugging purposes."
            d = 10
            
            if all([cover == BUILDINGS,
                    height - dsm[idx-1] > self.UNITDIFF,
                    not lcm[idx-1] == WALLS]):
                """
                First occurrence sufficient
                for wall definition.
                
                Assume idx<d> is the start
                of the remaining roof, for now.
                """
                lcm[idx:idx+d] = WALLS
                
                
            "Update mask"
            d    = d or 1
            idx += d
            
            # logger.error(f"{idx}")
        
        # logger.info("Line Done.")
            
    def __do_roof(self):
        """
        Handle the extent of 1 contiguous roof.
        """
    
    def __do_tree(self):
        """
        Handle the extent of 1 contiguous tree.
        """


class Shadow:
    """
    
    Shadow casting algorithm:
    
    Cast shadows according to Sun Location and DSM.
    Reproject scene according to viewing angle and DSM.
    
    """
    
    def __call__(self, line_pair):
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
    