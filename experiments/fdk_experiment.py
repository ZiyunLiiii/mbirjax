# -*- coding: utf-8 -*-
"""
**MBIRJAX: Feldkamp, Davis, and Kress (FDK) cone-beam reconstruction algorithm basic demo**

This is a modified version of the demo_1_shepp_logan.py file that its more streamlined and uses FDK reconstruction insetad of the standard recon.

"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install mbirjax

import numpy as np
import time
import jax.numpy as jnp
import mbirjax

# Set geometry parameters
geometry_type = "cone"  # FDK is necessarily cone, but I still define it as a "parameter"
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
print("Creating phantom", end="\n\n")
phantom = ct_model_for_generation.gen_modified_3d_sl_phantom()

# Generate sinogram from phantom
print("Creating sinogram", end="\n\n")
sinogram = ct_model_for_generation.forward_project(phantom)

# View sinogram
title = "Original sinogram \nUse the sliders to change the view or adjust the intensity range."
mbirjax.slice_viewer(sinogram, slice_axis=0, title=title, slice_label="View")

# Initialize the model for reconstruction.
ct_model_for_recon = mbirjax.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)

# Generate weights array - for an initial reconstruction, use weights = None, then modify if needed.
weights = None
# weights = ct_model_for_recon.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

# Set reconstruction parameter values
# Increase sharpness by 1 or 2 to get clearer edges, possibly with more high-frequency artifacts. ....
# ... Decrease by 1 or 2 to get softer edges and smoother interiors.
sharpness = 0.0
ct_model_for_recon.set_params(sharpness=sharpness)

# Print out model parameters
ct_model_for_recon.print_params()

################################################################################
# Reconstruction starts here
################################################################################

# Perform VCD reconstruction
print("Starting recon")
time0 = time.time()
recon = ct_model_for_recon.fdk_recon(sinogram)

recon.block_until_ready()
elapsed = time.time() - time0
print(f"Elapsed time for recon is {elapsed} seconds", end="\n\n")

# TODO: Solve the scaling issues of fbp_recon
def scale_to_unit_range(array):
    min_val = np.min(array)
    max_val = np.max(array)
    return (array - min_val) / (max_val - min_val)

recon = scale_to_unit_range(recon)

# Compute descriptive statistics about recon result
max_diff = np.amax(np.abs(phantom - recon))
nrmse = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
pct_95 = np.percentile(np.abs(recon - phantom), 95)
print(f"NRMSE between recon and phantom = {nrmse}")
print(f"Maximum pixel difference between phantom and recon = {max_diff}")
print(f"95% of recon pixels are within {pct_95} of phantom")

# Get stats on memory usage
mbirjax.get_memory_stats()

# Display results
title = 'Phantom (left) vs filtered back projection (right) \nUse the sliders to change the slice or adjust the intensity range.'
mbirjax.slice_viewer(phantom, recon, title=title)
