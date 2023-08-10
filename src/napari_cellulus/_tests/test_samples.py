def test_open(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.open_sample(plugin="napari-cellulus", sample="tissuenet_sample")
