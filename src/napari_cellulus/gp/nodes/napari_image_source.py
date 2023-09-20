from typing import Optional

import gunpowder as gp
from gunpowder.array_spec import ArraySpec
from napari.layers import Image


class NapariImageSource(gp.BatchProvider):
    """
    A gunpowder interface to a napari Image
    Args:
        image (Image):
            The napari Image to pull data from
        key (``gp.ArrayKey``):
            The key to provide data into
    """

    def __init__(
        self, image: Image, key: gp.ArrayKey, spec: Optional[ArraySpec] = None
    ):
        if spec is None:
            self.array_spec = self._read_metadata(image)
        else:
            self.array_spec = spec
        self.image = gp.Array(image.data.astype(float), self.array_spec)
        self.key = key

    def setup(self):
        self.provides(self.key, self.array_spec.copy())

    def provide(self, request):
        output = gp.Batch()
        output[self.key] = self.image.crop(request[self.key].roi)
        return output
