from integrate import arrDens_central, params, image_sum, bx_central_image_data, by_central_image_data, bz_central_image_data, image_sum_bx, image_sum_by, image_sum_bz
from scipy.io import readsav
import unittest
import matplotlib.pyplot as plt
import matplotlib

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
        self.output_by_2d_integrated = self.idl_save_output['by_2d_integrated']
        self.output_bz_2d_integrated = self.idl_save_output['bz_2d_integrated']

    def test_electron_density_center(self):
        for i in range(arrDens_central.shape[0]):
            for j in range(arrDens_central.shape[0]):
                # assert arrays are equivalent to within six decimal places
                self.assertAlmostEqual(self.output_dens_2d_center[i][j],arrDens_central[i][j],6)

    def test_electron_density_integrated(self):
        for i in range(image_sum.shape[0]):
            for j in range(image_sum.shape[0]):
                # assert arrays are equivalent to within six decimal places
                self.assertAlmostEqual(self.output_dens_integrated_2d[i][j],image_sum[i][j],6)

    def test_bx_central(self):
        for i in range(bx_central_image_data.shape[0]):
            for j in range(bx_central_image_data.shape[0]):
                # assert arrays are equivalent to within six decimal places
                self.assertAlmostEqual(self.output_bx_2d_center[i][j],bx_central_image_data[i][j],6)

    def test_by_central(self):
        for i in range(by_central_image_data.shape[0]):
            for j in range(by_central_image_data.shape[0]):
                # assert arrays are equivalent to within six decimal places
                self.assertAlmostEqual(self.output_by_2d_center[i][j],by_central_image_data[i][j],6)

    def test_bz_central(self):
        for i in range(bz_central_image_data.shape[0]):
            for j in range(bz_central_image_data.shape[0]):
                # assert arrays are equivalent to within six decimal places
                self.assertAlmostEqual(self.output_bz_2d_center[i][j],bz_central_image_data[i][j],6)

    def test_bx_integrated(self):
        for i in range(image_sum_bx.shape[0]):
            for j in range(image_sum_bx.shape[0]):
                # assert arrays are equivalent to within six decimal places
                self.assertAlmostEqual(self.output_bx_2d_integrated[i][j],image_sum_bx[i][j],6)

    def test_by_integrated(self):
        for i in range(image_sum_by.shape[0]):
            for j in range(image_sum_by.shape[0]):
                # assert arrays are equivalent to within six decimal places
                self.assertAlmostEqual(self.output_by_2d_integrated[i][j],image_sum_by[i][j],6)

    def test_bz_integrated(self):
        for i in range(image_sum_bz.shape[0]):
            for j in range(image_sum_bz.shape[0]):
                # assert arrays are equivalent to within six decimal places
                self.assertAlmostEqual(self.output_bz_2d_integrated[i][j],image_sum_bz[i][j],6)

if __name__ == '__main__':
    unittest.main()
