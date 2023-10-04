from typing import Optional

import gunpowder as gp
import numpy as np
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
        self,
        image: Image,
        key: gp.ArrayKey,
        spec: Optional[ArraySpec],
        spatial_dims,
    ):
        if spec is None:
            self.array_spec = self._read_metadata(image)
        else:
            self.array_spec = spec
        self.spatial_dims = spatial_dims
        self.image = gp.Array(
            self.normalize_min_max_percentile(
                image.data.astype(np.float32),
                pmin=1,
                pmax=99.8,
                axis=self.spatial_dims,
            ),
            self.array_spec,
        )
        self.key = key

    def normalize_min_max_percentile(
        self,
        x,
        pmin=3,
        pmax=99.8,
        axis=None,
        clip=False,
        eps=1e-20,
        dtype=np.float32,
    ):
        """
        Percentile-based image normalization.
        Function taken from StarDist repository  https://github.com/stardist/stardist
        """
        mi = np.percentile(x, pmin, axis=axis, keepdims=True)
        ma = np.percentile(x, pmax, axis=axis, keepdims=True)
        return self.normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)

    def normalize_mi_ma(
        self, x, mi, ma, clip=False, eps=1e-20, dtype=np.float32
    ):
        """
        Percentile-based image normalization.
        Function taken from StarDist repository  https://github.com/stardist/stardist
        """
        if dtype is not None:
            x = x.astype(dtype, copy=False)
            mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
            ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
            eps = dtype(eps)

        try:
            import numexpr

            x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
        except ImportError:
            x = (x - mi) / (ma - mi + eps)

        if clip:
            x = np.clip(x, 0, 1)

        return x

    def setup(self):
        self.provides(self.key, self.array_spec.copy())

    def provide(self, request):
        output = gp.Batch()
        output[self.key] = self.image.crop(request[self.key].roi)
        return output
