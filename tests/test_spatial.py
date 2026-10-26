import numpy as np
import SimpleITK as sitk

from pulmognosis.preprocessing.spatial import resample_around, resample_isotropic, resample_to_reference, to_array


def _image(array_zyx, spacing_xyz, origin=(0.0, 0.0, 0.0)):
    img = sitk.GetImageFromArray(array_zyx.astype(np.float32))
    img.SetSpacing(spacing_xyz)
    img.SetOrigin(origin)
    return img


def test_resample_isotropic_preserves_physical_extent():
    img = _image(np.zeros((20, 50, 40)), spacing_xyz=(0.5, 0.5, 2.5))
    out = resample_isotropic(img, 1.0)
    assert out.GetSpacing() == (1.0, 1.0, 1.0)
    assert out.GetSize() == (20, 25, 50)  # x: 40*0.5, y: 50*0.5, z: 20*2.5
    assert out.GetOrigin() == img.GetOrigin()


def test_resample_to_reference_keeps_a_point_at_the_same_physical_location():
    # Coarse "PET": one hot voxel at physical (8, 8, 8) mm with 4 mm voxels.
    pet = np.zeros((5, 5, 5))
    pet[2, 2, 2] = 10.0
    moving = _image(pet, (4.0, 4.0, 4.0))
    reference = _image(np.zeros((20, 20, 20)), (1.0, 1.0, 1.0))

    out = to_array(resample_to_reference(moving, reference, sitk.sitkNearestNeighbor))
    assert out.shape == (20, 20, 20)
    assert out[8, 8, 8] == 10.0  # (z, y, x) index 8 == 8 mm on a 1 mm grid at origin 0


def test_resample_around_centres_the_cube_on_a_physical_point():
    vol = np.zeros((40, 40, 40))
    vol[20, 10, 30] = 100.0  # (z, y, x) → physical (30, 10, 20) mm on a 1 mm grid at origin 0
    img = _image(vol, (1.0, 1.0, 1.0))

    cube = resample_around(img, (30.0, 10.0, 20.0), size=9, spacing_mm=0.5, interpolator=sitk.sitkNearestNeighbor)
    assert cube.shape == (9, 9, 9)
    assert cube[4, 4, 4] == 100.0  # the point sits at the cube's centre


def test_resample_around_fills_outside_the_image():
    img = _image(np.zeros((10, 10, 10)), (1.0, 1.0, 1.0))
    cube = resample_around(img, (0.0, 0.0, 0.0), size=8, spacing_mm=1.0, default_value=-1000.0)
    assert cube[0, 0, 0] == -1000.0  # 3.5 mm before the origin → outside
