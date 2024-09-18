# -*- coding: utf-8 -*-
"""
**MBIRJAX: FDK (filtered back projection for cone-beam) Basic Demo**

This is basically a modified version of the demo_1_shepp_logan.py file that its more streamlined and uses filtered back projection insetad of the standard recon.

"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install mbirjax

import numpy as np
import time
import jax.numpy as jnp
import mbirjax

"""**Set the geometry parameters**"""

# set parameters
geometry_type = "cone"

# Set parameters for the problem size - you can vary these, but if you make num_det_rows 
# very small relative to channels, then the generated phantom may not have an interior.
num_views = 256
num_det_rows = 40
num_det_channels = 128

# For cone beam geometry, we need to describe the distances source to detector and source to rotation axis.
# np.Inf is an allowable value, in which case this is essentially parallel beam
source_detector_dist = 4 * num_det_channels
source_iso_dist = source_detector_dist / 2

detector_cone_angle = 2 * np.arctan2(num_det_channels / 2, source_detector_dist)
start_angle = -(np.pi + detector_cone_angle) * (1/2)
end_angle = (np.pi + detector_cone_angle) * (1/2)

# Initialize sinogram
sinogram_shape = (num_views, num_det_rows, num_det_channels)
angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

ct_model_for_generation = mbirjax.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)

# Generate 3D Shepp Logan phantom
print('Creating phantom')
phantom = ct_model_for_generation.gen_modified_3d_sl_phantom()

# Generate synthetic sinogram data
print('Creating sinogram')
sinogram = ct_model_for_generation.forward_project(phantom)
sinogram = np.array(sinogram)

# View sinogram
title = 'Original sinogram \nUse the sliders to change the view or adjust the intensity range.'
mbirjax.slice_viewer(sinogram, slice_axis=0, title=title, slice_label='View')

"""**Initialize for the reconstruction**"""

# Initialize the model for reconstruction.

ct_model_for_recon = mbirjax.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)

# Generate weights array - for an initial reconstruction, use weights = None, then modify if needed.
weights = None
# weights = ct_model_for_recon.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')


# Print out model parameters
ct_model_for_recon.print_params()

"""**Do the reconstruction and display the results.**"""

# Perform VCD reconstruction
print('Starting recon')
time0 = time.time()
recon = ct_model_for_recon.fdk_recon(sinogram)

recon.block_until_ready()
elapsed = time.time() - time0
##########################

def scale_to_unit_range(array):
    min_val = np.min(array)
    max_val = np.max(array)
    return (array - min_val) / (max_val - min_val)

# THIS NEEDS TO BE FIXED AND REMOVED.
recon = scale_to_unit_range(recon)

max_diff = np.amax(np.abs(phantom - recon))
print('Geometry = {}'.format(geometry_type))
nrmse = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
pct_95 = np.percentile(np.abs(recon - phantom), 95)
print('NRMSE between recon and phantom = {}'.format(nrmse))
print('Maximum pixel difference between phantom and recon = {}'.format(max_diff))
print('95% of recon pixels are within {} of phantom'.format(pct_95))

mbirjax.get_memory_stats()
print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))

# Display results
title = 'Phantom (left) vs filtered back projection (right) \nUse the sliders to change the slice or adjust the intensity range.'
mbirjax.slice_viewer(phantom, recon, title=title)
print(np.min(recon))
