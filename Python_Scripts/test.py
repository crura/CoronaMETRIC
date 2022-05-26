from integrate import arrDens_central, params
from scipy.io import readsav
import unittest

class TestDataConsistency(unittest.TestCase):

    def setUp(self):
        self.idl_save_output = readsav('/Users/crura/Desktop/Research/github/Image-Coalignment/Data/{}.sav'.format(params))
        self.output_dens_2d_center = self.idl_save_output['dens_2d_center']
        self.output_dens_integrated_2d = self.idl_save_output['dens_integrated_2d']
        self.output_forward_pb_image = self.idl_save_output['forward_pb_image']
        self.output_bx_2d_center = self.idl_save_output['bx_2d_center']
        self.output_by_2d_center = self.idl_save_output['by_2d_center']
        self.output_bz_2d_center = self.idl_save_output['bz_2d_center']
        self.output_bx_2d_integrated = self.idl_save_output['bx_2d_integrated']
        self.output_bx_2d_integrated = self.idl_save_output['by_2d_integrated']
        self.output_bx_2d_integrated = self.idl_save_output['bz_2d_integrated']

    def test_electron_density_center(self):
        for i in range(arrDens_central.shape[0]):
            for j in range(arrDens_central.shape[0]):
                self.assertAlmostEqual(self.output_dens_2d_center[i][j],arrDens_central[i][j],6)

if __name__ == '__main__':
    unittest.main()
