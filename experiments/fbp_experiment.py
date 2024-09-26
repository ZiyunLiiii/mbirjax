"""
**MBIRJAX: Filtered back projection FBP basic demo**

This is a modified version of the demo_1_shepp_logan.py file that is more streamlined and uses filtered back projection insetad of the standard recon.

"""

import numpy as np
import time
import jax.numpy as jnp
import mbirjax

# Set geometry parameters
geometry_type = "parallel"  # FBP is necessarily parallel, but I still define it as a "parameter"
num_views = 256
num_det_rows = 40
num_det_channels = 128
detector_cone_angle = 0
start_angle = -(np.pi + detector_cone_angle) * (1/2)
end_angle = (np.pi + detector_cone_angle) * (1/2)

# Initialize sinogram
sinogram_shape = (num_views, num_det_rows, num_det_channels)
angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)
ct_model_for_generation = mbirjax.ParallelBeamModel(sinogram_shape, angles)

# Generate 3D Shepp Logan phantom
print("Creating phantom", end="\n\n")
phantom = ct_model_for_generation.gen_modified_3d_sl_phantom()

# Generate sinogram from phantom
print("Creating sinogram", end="\n\n")
sinogram = ct_model_for_generation.forward_project(phantom)

# View sinogram
title = "Original sinogram \nUse the sliders to change the view or adjust the intensity range."
# mbirjax.slice_viewer(sinogram, slice_axis=0, title=title, slice_label="View")

# Initialize the model for reconstruction.
ct_model_for_recon = mbirjax.ParallelBeamModel(sinogram_shape, angles)

# Generate weights array - for an initial reconstruction, use weights = None, then modify if needed.
weights = None
# weights = ct_model_for_recon.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

# Set reconstruction parameter values
# Increase sharpness by 1 or 2 to get clearer edges, possibly with more high-frequency ...
# ... artifacts. Decrease by 1 or 2 to get softer edges and smoother interiors.
sharpness = 0.0
ct_model_for_recon.set_params(sharpness=sharpness)

# Print out model parameters
ct_model_for_recon.print_params()

################################################################################
# Reconstruction starts here
################################################################################

# Perform VCD reconstruction
print("Starting recon", end="\n\n")
time0 = time.time()
filter = "Ram-Lak" 
recon = ct_model_for_recon.fbp_recon_vmap(sinogram, filter=filter)

recon.block_until_ready()
elapsed = time.time() - time0
print(f"Elapsed time for recon is {elapsed} seconds", end="\n\n")

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
title = f"Phantom (left) vs filtered back projection (right). Filter used: {filter}. \nUse the sliders to change the slice or adjust the intensity range."
mbirjax.slice_viewer(phantom, recon, title=title)
