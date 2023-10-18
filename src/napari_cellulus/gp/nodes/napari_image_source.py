import gunpowder as gp
import numpy as np
from csbdeep.utils import normalize
from gunpowder.array_spec import ArraySpec
from napari.layers import Image


class NapariImageSource(gp.BatchProvider):
    """
    A gunpowder interface to a napari Image
    Args:
        image (Image):
            The napari image layer to pull data from
        key (``gp.ArrayKey``):
            The key to provide data into
    """

    def __init__(
        self, image: Image, key: gp.ArrayKey, spec: ArraySpec, spatial_dims
    ):
        self.array_spec = spec
        self.image = gp.Array(
            normalize(
                image.data.astype(np.float32),
                pmin=1,
                pmax=99.8,
                axis=spatial_dims,
            ),
            self.array_spec,
        )
        self.spatial_dims = spatial_dims
        self.key = key

    def setup(self):
        self.provides(self.key, self.array_spec.copy())

    def provide(self, request):
        output = gp.Batch()
        output[self.key] = self.image.crop(request[self.key].roi)
        return output
