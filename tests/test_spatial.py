import numpy as np
import SimpleITK as sitk

from pulmognosis.preprocessing.spatial import crop_patch, resample_isotropic, resample_to_reference, to_array


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


def test_crop_patch_centred_and_padded():
    vol = np.arange(10 * 10 * 10, dtype=float).reshape(10, 10, 10)
    patch = crop_patch(vol, (5, 5, 5), size=4)
    assert patch.shape == (4, 4, 4)
    assert patch[2, 2, 2] == vol[5, 5, 5]

    corner = crop_patch(vol, (0, 0, 0), size=4, fill=-1.0)
    assert corner[0, 0, 0] == -1.0  # outside the volume → fill
    assert corner[2, 2, 2] == vol[0, 0, 0]
