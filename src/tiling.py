#!/usr/bin/python3
import numpy as np


def to_tiles(X, tile_size):
    
    if X.ndim == 2:
        
        X = X.reshape(1, *X.shape)
        
    assert X.ndim == 3, "Array is not 3D"
    
    if X.ndim == 3:
        """
        Handle 3-dimensional arrays.
        """

        # Pad X to be divisible by tile size
        X = np.pad(X, ((0, 0),
                    (0, int((tile_size - X.shape[-2] % tile_size))*bool(X.shape[-2]%tile_size)),
                    (0, int((tile_size - X.shape[-1] % tile_size))*bool(X.shape[-2]%tile_size))),
                constant_values=0)

        channel_axis = X.shape.index(min(X.shape))
        
        assert channel_axis == 0, "channel axis is not at position -3"

        # Move channels axis to the end for convenience
        X = np.moveaxis(X, channel_axis, -1)
        arr_height, arr_width, channels = X.shape

        X = X.reshape(arr_height // tile_size,
                      tile_size,
                      arr_width // tile_size,
                      tile_size, channels)
        X = X.swapaxes(1, 2).reshape(-1,
                                     tile_size,
                                     tile_size,
                                     channels)

        # Return channels to axis -3 as expected by PyTorch
        X = np.moveaxis(X, -1, 1)
        
    if X.shape[1] == 1:
        """
        If there's only one channel, squeeze it.
        """
        X = X.reshape(X.shape[0], tile_size, tile_size)

    return X

def from_tiles(X: np.array, orig_dims: tuple):
    assert X.ndim == 4, f'Expected 4d array, but got {X.ndim}d'    
    if X.ndim == 4:
        tiles, channels, tiles_size, tiles_size = X.shape
        X = np.moveaxis(X, 1, -1)
        X = X.reshape(orig_dims[0] // tiles_size,
                      orig_dims[1] // tiles_size,
                      tiles_size, tiles_size, channels)
        X = X.swapaxes(2, 1)
        X = X.reshape(*orig_dims, channels)
    return X
