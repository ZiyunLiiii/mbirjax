import numpy as np
import jax.numpy as jnp
import mbirjax
import unittest


class TestFBPReconstruction(unittest.TestCase):
    """
    Unit test for FBP reconstruction.
    Tests the accuracy of the reconstruction against the 3 metrics: NRMSE, max_diff, and pct_95.
    """

    def setUp(self):
        """Set up parameters and initialize models before each test."""
        # Geometry parameters
        self.geometry_type = "parallel"  # FBP is parallel
        self.num_views = 256
        self.num_det_rows = 128
        self.num_det_channels = 128
        detector_cone_angle = 0
        start_angle = -(np.pi + detector_cone_angle) * (1/2)
        end_angle = (np.pi + detector_cone_angle) * (1/2)
        
        # Sinogram parameters
        self.sinogram_shape = (self.num_views, self.num_det_rows, self.num_det_channels)
        self.angles = jnp.linspace(start_angle, end_angle, self.num_views, endpoint=False)
        
        # Initialize the model for phantom generation and reconstruction
        self.ct_model_for_generation = mbirjax.ParallelBeamModel(self.sinogram_shape, self.angles)
        self.ct_model_for_recon = mbirjax.ParallelBeamModel(self.sinogram_shape, self.angles)

        # Generate 3D Shepp-Logan phantom and sinogram
        self.phantom = self.ct_model_for_generation.gen_modified_3d_sl_phantom()
        self.sinogram = self.ct_model_for_generation.forward_project(self.phantom)

        # Set tolerances for the metrics
        self.tolerances = {'nrmse': 0.20, 'max_diff': 0.40, 'pct_95': 0.05}

    def test_fbp_reconstruction(self):
        """Test the FBP reconstruction against the defined tolerances."""
        # Perform FBP reconstruction
        filter = "Ram-Lak"
        recon = self.ct_model_for_recon.fbp_recon_reshape_jit(self.sinogram, filter=filter)
        recon.block_until_ready()

        # Compute the statistics
        max_diff = np.amax(np.abs(self.phantom - recon))
        nrmse = np.linalg.norm(recon - self.phantom) / np.linalg.norm(self.phantom)
        pct_95 = np.percentile(np.abs(recon - self.phantom), 95)

        # Verify that the computed stats are within tolerances
        self.assertTrue(max_diff < self.tolerances['max_diff'], f"Max difference too high: {max_diff}")
        self.assertTrue(nrmse < self.tolerances['nrmse'], f"NRMSE too high: {nrmse}")
        self.assertTrue(pct_95 < self.tolerances['pct_95'], f"95th percentile difference too high: {pct_95}")


if __name__ == '__main__':
    unittest.main()