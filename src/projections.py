
from concurrent.futures import thread
import logging
import multiprocessing
import threading
from time import sleep
import traceback
import numpy as np
import os

from scipy.ndimage import rotate

from multiprocessing import JoinableQueue, Process, Queue as pQueue, RawArray, current_process
from threading import Thread, current_thread
from queue import Empty, Queue as tQueue, deque
from tqdm import tqdm
from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple
from rasters import LandCoverCleaner, RasterIn, RasterOut
from algorithms import LCMView, Shadow
from ctypes import c_uint8

if __name__ == 'projections':
    from argparser import args

logger = logging.getLogger(__file__);
logger.setLevel(logging.DEBUG);

logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(filename)s:%(processName)s:%(funcName)s:%(lineno)d: %(message)s',
    level=logging.DEBUG,
    datefmt='%H:%M:%S %b%d'
)

"""

# TODO

Things to solve:

1) # BUG RasterOut Class is deadlocking when writing right-side edges.
   # Probably because north or west tile is never written.
   # This needs to be figured out.
    
2) Methods should be rearranged in better logical order for better maintenance.
"""


class Projector:
    """
    
    # TODO
    # Movie multiprocessing from spawning processes
    # To daemons. Keep constant workers up.
    
    """
    def __init__(self) -> None:
        self.tile_size  = args.ts
        self.num_p      = args.c
        self.num_t      = args.threads
        self.threads    = []
        self.processes  = []
        self.deamons    = []
        self.t_queue    = tQueue()
        self.d_queue    = pQueue()
        self.t_comple_Q = pQueue()
                
        self.algorithms = {
            "lcmv": LCMView,
            "shad": Shadow
        }
        self.lcmviewer  = LCMView()
        self.shadowcast = Shadow()
        
        self.tile_completion_Qs = {}
        self.p_queues           = {}
        
        for i in range(self.num_t):
            
            name    = f"Thread_{i}"
            
            # p_queue = JoinableQueue()
            # c_queue = pQueue()
            # c_queue.put([])
            
            t = Thread(target=self.__thread,
                       name=name,
                       args=(self.t_queue, self.d_queue, self.t_comple_Q,))
            
            self.threads.append(
                t
            )
            
            t.start()
            
        for i in range(self.num_p):
            name = f"Daemon_{i}"
            d    = LineProcess(
                target=self.__process,
                args=(self.d_queue,
                      self.t_comple_Q),
                name=name
            )
            self.deamons.append(d)
            d.start()
    
    def __make_progress(self, image: RasterIn, angles: List):
        # tqdm iterable should come from parser
        todo                  = len(image) * len(angles)
        self.__progress_queue = tQueue()
        self.__progress_bar   = tqdm(range(todo),
                                     desc="Progress",
                                     unit="blocks",
                                     colour='RED')        
        self.__progress_queue.put(self.__progress_bar)

        self.t_comple_Q.put(np.zeros(todo))
        
    def __update_progress(self):
        "Count block."
        progress = self.__progress_queue.get()
        progress.update(1)
        self.__progress_queue.put(progress)
    
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
        
        for payload in iter(thread_queue.get, None):

            if isinstance(payload, tuple):
                """
                Load tiles and process.
                """
                
                idx, azim, zen, out = payload
                
                lcm_tile   = self.lcm[idx]
                dsm_tile   = self.dsm[idx]
                
                (lcm_tile,
                 dsm_tile) = self.cleaner(lcm_tile, dsm_tile)
                
                result = self.__do_tile(
                    (lcm_tile, dsm_tile),
                    (azim, zen),
                    p_queue,
                    c_queue,
                    idx
                )
                
                "Write block to RasterOut instance."
                out.write(idx, result)
                                
                self.__update_progress()
                
            else:
                
                logger.error("Invalid thread payload type.")
                
            # payload = thread_queue.get()
        
        """
        If signaled to stop (payload is None)
        then replace the signal in the queue
        for the rest of the threads.
        """
        # thread_queue.task_done()
        thread_queue.put(None)    
    
    def __process(self, p_queue: pQueue, c_queue: pQueue):
        """
        Each process handles 1 line at a time.
        """
        for payload in iter(p_queue.get, None):
                            
            lcm, dsm, zen, idx = payload
            self.__do_line(lcm, dsm, zen)
            
            "Keep count of lines written."
            """
            # TODO
            # I should change this to running sums.
            # An array of size No. Tiles. filled with zeros.
            # Append 1 for every line processed at the correct
            # position.
            """
            prog       = c_queue.get()
            prog[idx] += 1
            c_queue.put(prog)
                
        logger.error("FINISHED")
        return 0
        
    def __feed_pQueue_n_wait(self,
                             tiles: Tuple[np.ndarray],
                             zen: float,
                             p_queue: pQueue,
                             c_queue: pQueue,
                             idx: int):
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
            p_queue.put((*lines, zen, idx))
        
        self.__check_thread_completion(c_queue, idx)
        return
    
    def __check_thread_completion(self, c_queue: pQueue, idx: int):
        """
        Shared between threads.
        
        Purpose is to check whether all number
        of processes within each thread are 
        out of tasks.
        
        Once this condition is true, then return.
        """
        
        flag = True
        while flag:
            
            bucket = c_queue.get()
            flag   = bucket[idx] < self.tile_size
            c_queue.put(bucket)
            
            sleep(.3)
            
        return
    
    def __do_tile(self,
                  tiles: Tuple,
                  angles: Tuple,
                  p_queue: pQueue,
                  c_queue: pQueue,
                  idx: int) -> np.ndarray:
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
        rotation = 270 - azim
        
        "This is a copy"
        lcm, dsm = self.__rotate((lcm, dsm), rotation)
        lcm      = self.__make_shared(lcm)
        
        self.__feed_pQueue_n_wait((lcm, dsm), zen, p_queue, c_queue, idx)
        # logger.debug(f"{p_queue.qsize()}")
        # self.__start_thread_processes(p_queue, c_queue)
        # logger.debug(f"{p_queue.qsize()}")
        "Wait for the processes to return."
        
        "Rotate back to origin"
        lcm = self.__rotate((lcm,), -rotation, reshape=False)
        return lcm[0]
    
    def __do_line(self, lcm, dsm, zen):
        # print(f"""
        #       -----------------------  Dummy Line Doer
        #       Angle  : {zen}
        #       Lines  : {lcm.shape} {dsm.shape}
        #       Thread : {current_thread().name}
        #       Process: {current_process().name} 
        #       -----------------------  Dummy Line Doer ++++++++++++++++++++++++
        #       """)
        self.lcmviewer(lcm, dsm, zen)
        return
    
    def __rotate(self, blocks: Tuple[RasterIn], angle, cv=0, reshape=True):
        rotated = []
        for block in blocks:
            rotated.append(
                rotate(block, angle=angle, cval=cv, reshape=reshape, order=0)
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
        
        out    = RasterOut(self.lcm, angles)
           
        for idx in range(len(self.lcm)):
            """
            Perhaps the RasterOut object
            should be fed through the Queue.
            
            # TODO
            # DONE.
            """
            self.t_queue.put((idx, *angles, out))
        
        "Plant sentinel"
        self.t_queue.put(None)
        
    def main(self):
        
        self.dsm      = RasterIn(args.dsm)
        self.lcm      = RasterIn(args.lcm)
        self.cleaner  = LandCoverCleaner(self.lcm,
                                         self.dsm)
        
        self.__make_progress(self.lcm, args.angles)
        
        for angles in args.angles:
            self.__do_angles(angles)
        
        return 0;


class LineProcess(Process):
    """
    Process subclass to throw Exceptions
    in the Main Process.
    """
    def __init__(self,
                 group: None = None,
                 target: Callable[..., Any] | None = ...,
                 name: str | None = ...,
                 args: Iterable[Any] = ...,
                 kwargs: Mapping[str, Any] = {},
                 *,
                 daemon: bool | None = ...) -> None:
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception = None
    
    def run(self) -> None:
        try:
            super().run()
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            raise e
        
    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


class TileThread(Thread):
    """
    Probably not needed.
    
    To be removed.
    """
    def __init__(self,
                 group: None = None,
                 target: Callable[..., Any] | None = ...,
                 name: str | None = ...,
                 args: Iterable[Any] = ...,
                 kwargs: Mapping[str, Any] | None = {},
                 *,
                 daemon: bool | None = ...) -> None:
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
    
    def run(self) -> None:
        try:
            super().run()
        except Exception as e:
            raise e
