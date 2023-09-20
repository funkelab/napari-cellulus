from typing import Tuple


class NapariDatasetMetaData:
    def __init__(self, shape, axis_names):
        self.num_dims = len(axis_names)
        self.num_spatial_dims: int = 0
        self.num_samples: int = 0
        self.num_channels: int = 0
        self.sample_dim = None
        self.channel_dim = None
        self.time_dim = None
        self.spatial_array: Tuple[int, ...] = ()
        for dim, axis_name in enumerate(axis_names):
            if axis_name == "s":
                self.sample_dim = dim
                self.num_samples = shape[dim]
            elif axis_name == "c":
                self.channel_dim = dim
                self.num_channels = shape[dim]
            elif axis_name == "t":
                self.num_spatial_dims += 1
                self.time_dim = dim
            elif axis_name == "z":
                self.num_spatial_dims += 1
                self.spatial_array += (shape[dim],)
            elif axis_name == "y":
                self.num_spatial_dims += 1
                self.spatial_array += (shape[dim],)
            elif axis_name == "x":
                self.num_spatial_dims += 1
                self.spatial_array += (shape[dim],)
