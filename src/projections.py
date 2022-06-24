
import logging
import multiprocessing
import threading
import numpy as np
import os

from scipy.ndimage import rotate

from multiprocessing import Process, Queue as pQueue, RawArray
from threading import Thread, current_thread
from queue import Queue as tQueue, deque
from tqdm import tqdm
from typing import Any, Dict, List, Tuple
from rasters import LandCoverCleaner, RasterIn, RasterOut
from algorithms import LCMView, Shadow
from ctypes import c_uint8

if __name__ == 'projections':
    from argparser import args

logger = logging.getLogger(__file__);
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
        self.t_queue    = tQueue()
        self.lcmviewer  = LCMView()
        self.shadowcast = Shadow()
        
        self.tile_completion_Qs = {}
        self.p_queues           = {}
        
        for i in range(self.num_t):
            
            name    = f"Thread_{i}"
            
            p_queue = pQueue()
            c_queue = pQueue()
            c_queue.put([])
            
            t = Thread(target=self.__thread,
                       name=name,
                       args=(self.t_queue, p_queue, c_queue,))
            
            self.threads.append(
                t
            )
            
            t.start()
    
    def __make_progress(self, image: RasterIn, angles: List):
        # tqdm iterable should come from parser
        todo                  = len(image) * len(angles)
        self.__progress_queue = tQueue()
        self.__progress_bar   = tqdm(range(todo),
                                     desc="Progress",
                                     unit="blocks",
                                     colour='RED')        
        self.__progress_queue.put(self.__progress_bar)
    
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
    
    def __thread(self, thread_queue: tQueue, p_queue: pQueue, c_queue: pQueue):
        """
        
        # TODO
        # Implement None as termination flag?
        
        # Data loading should happen
        # within each thread.
        
        # Therefore tQueue should probably
        # only receive an index to a tile.
        
        """

        for i in range(self.num_p):
            """
            If each thread has dedicated processes
            it must also have dedicated processing Queues.
            
            # TODO
            # Implement thread specific processing Queues.
            """

            p = Process(target=self.__process,
                        name=f"Process_{i}_{threading.current_thread().name}",
                        args=(
                            # Feed the dedicated queues
                            p_queue, c_queue,
                        )
                        )
            self.processes.append(
                p
            )
            p.start()
        
        payload = thread_queue.get()
        while not payload == None:
            if isinstance(payload, tuple):
                """
                Load tiles and process.
                """
                
                idx, azim, zen, out = payload
                
                lcm_tile = self.lcm[idx]
                dsm_tile = self.dsm[idx]
                
                result = self.__do_tile(
                    (lcm_tile, dsm_tile),
                    (azim, zen),
                    p_queue,
                    c_queue
                )
                
                "Write block to RasterOut instance."
                out.write(idx, result)
                
                self.__update_progress()
                
            else:
                logger.error("Invalid thread payload type.")
                
            payload = thread_queue.get()
        
        """
        If signaled to stop (payload is None)
        then replace the signal in the queue
        for the rest of the threads.
        """
        thread_queue.task_done()
        thread_queue.put(None)    
    
    def __update_progress(self):
        "Count block."
        progress = self.__progress_queue.get()
        progress.update(1)
        self.__progress_queue.put(progress)
    
    def __process(self, p_queue: pQueue, c_queue: pQueue):
        """
        Each process handles 1 line at a time.
        """
        
        payload = p_queue.get()
        
        while not payload == None:
            if isinstance(payload, tuple):
                
                lcm, dsm, zen = payload
                self.__do_line(lcm, dsm, zen)
                
                if p_queue.empty():
                    
                    """
                    If Queue is empty, assume no remaining tasks.
                    
                    Write on bucket that process is done.
                    """
                    
                    bucket = c_queue.get()
                    
                    bucket.append(multiprocessing.current_process().name)
                    
                    c_queue.put(bucket)
            else:
                logger.error(f"Invalid process payload type.")
                                    
            payload = p_queue.get()
            
        """
        If signaled to stop (payload is None)
        then replace the signal in the queue
        for the rest of the processes.
        """
        p_queue.task_done()
        p_queue.put(None)
    
    def __feed_pQueue_n_wait(self,
                             tiles: Tuple[np.ndarray],
                             zen: float,
                             p_queue: pQueue,
                             c_queue: pQueue):
        """
        Feed for multiprocessing in here.
        But wait for processes to finish
        before proceeding.
        
        Need to ensure processes return before
        thread goes on to write to disk.
        
        Need to signal the tile is done somehow.
        
        # TODO
        # Implement signaling that the tile is done.
        # Implement wait until signal.
        # Done. Verify integrity.
        
        <bucket>: Expected to be of type List.
        
        """
        
        for lines in zip(*tiles):
            """
            Feed tasks to the processes.
            """
            p_queue.put((*lines, zen))
        
        self.__check_thread_completion(c_queue)
        return
    
    def __check_thread_completion(self, c_queue: pQueue):
        """
        Unique to each thread.
        
        Purpose is to check whether all number
        of processes within each thread are 
        out of tasks.
        
        Once this condition is true, then return.
        """
        
        "Retrieve the bucket for variable definition"
        bucket = c_queue.get()
        
        while len(bucket) < self.num_p:
            
            "Return the bucket to processes"
            c_queue.put(bucket)
            
            """
            
            This is dead space during which the running
            processes are expected to be operating.
            
            # TODO
            # Verify concept.
            
            """
            logger.warn(f"Deadlock {current_thread().name}")
            print("hello")
            
            "Reclaim the bucket for inspection"
            bucket = c_queue.get()
        
        "Logic assertion"
        assert c_queue.empty(), "Queue not empty -- logic error"
        
        "Clear the bucket for use by the next thread task"
        bucket.clear()
        
        "Reload the bucket for use by the next thread task"
        c_queue.put(bucket)
        
        return
    
    
    def __do_tile(self,
                  tiles: Tuple,
                  angles: Tuple,
                  p_queue: pQueue,
                  c_queue: pQueue) -> np.ndarray:
        """
        Tile handling process.
        
        # TODO
        # Verify solidity of process.
        """
        lcm,  dsm  = tiles
        azim, zen  = angles
        
        """
        Calculate desired rotation angle,
        so as to bring azimuth to 270 degrees.
        """
        rotation  = 270 - azim
        
        "This is a copy"
        lcm, dsm = self.__rotate((lcm, dsm), rotation)
        lcm      = self.__make_shared(lcm)
        
        "Feed the multiprocessing queue"
        self.__feed_pQueue_n_wait((lcm, dsm), zen, p_queue, c_queue)
        
        "Rotate back to origin"
        lcm = self.__rotate((lcm,), -rotation, reshape=False)
        return lcm
    
    def __do_line(self, lcm, dsm, zen):
        print(f"""
              -----------------------  Dummy Line Doer
              Angle  : {zen}
              Lines  : {lcm.shape} {dsm.shape}
              Thread : {threading.current_thread().name}
              Process: {multiprocessing.current_process().name} 
              -----------------------  Dummy Line Doer ++++++++++++++++++++++++
              """)
        return 0
    
    def __rotate(self, blocks: Tuple[RasterIn], angle, cv=0, reshape=True):
        rotated = []
        for block in blocks:
            rotated.append(
                rotate(block, angle=angle, cval=cv, reshape=reshape)
            )
        return rotated

    def __do_angles(self, angles: Tuple):
        """
        Each angle has to produce its
        own RasterOut object for the map.
        
        I guess this should happen in here.
        
        Eventually, every running thread
        is desired to be doing its own writing.
        
        So it needs access to current
        tile's RasterOut object.
        
        angles: (Azimuth, Zenith)
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
        
        self.__make_progress(self.lcm, args.angles)
        
        for angles in args.angles:
            self.__do_angles(angles)
        
        return 0;
