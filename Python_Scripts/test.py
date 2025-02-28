# Copyright 2025 Christopher Rura

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from integrate import arrDens_central, params, image_sum, bx_central_image_data, by_central_image_data, bz_central_image_data, image_sum_bx, image_sum_by, image_sum_bz, arrDens_forward
# from coalign import bz_integrated_coaligned, by_integrated_coaligned, bx_integrated_coaligned, bx_central_coaligned, by_central_coaligned, bz_central_coaligned,
from scipy.io import readsav
import unittest
import matplotlib.pyplot as plt
import matplotlib
import git
import os
import pandas as pd

class TestDataConsistency(unittest.TestCase):

    def setUp(self):
        repo = git.Repo('.', search_parent_directories=True)
        self.repo_path = repo.working_tree_dir
        self.idl_save_output = readsav(os.path.join(self.repo_path,'Output/{}.sav'.format(params)))
        self.output_dens_2d_center = self.idl_save_output['dens_2d_center']
        self.output_dens_integrated_2d = self.idl_save_output['dens_integrated_2d']
        self.output_forward_pb_image = self.idl_save_output['forward_pb_image']
        self.output_bx_2d_center = self.idl_save_output['bx_2d_center']
        self.output_by_2d_center = self.idl_save_output['by_2d_center']
        self.output_bz_2d_center = self.idl_save_output['bz_2d_center']
        self.output_bx_2d_integrated = self.idl_save_output['bx_2d_integrated']
        self.output_by_2d_integrated = self.idl_save_output['by_2d_integrated']
        self.output_bz_2d_integrated = self.idl_save_output['bz_2d_integrated']

        self.idl_save_coaligned_output = readsav(os.path.join(self.repo_path,'Output/Coaligned_Parameters/{}.sav'.format(params)))
        self.coaligned_output_dens_2d_center = self.idl_save_coaligned_output['psi_central_dens_coaligned']
        self.coaligned_output_dens_integrated_2d = self.idl_save_coaligned_output['psi_integrated_dens_coaligned']
        self.coaligned_output_forward_pb_image = self.idl_save_coaligned_output['psi_forward_pb_coaligned']
        self.coaligned_output_bx_2d_center = self.idl_save_coaligned_output['bx_central_coaligned']
        self.coaligned_output_by_2d_center = self.idl_save_coaligned_output['by_central_coaligned']
        self.coaligned_output_bz_2d_center = self.idl_save_coaligned_output['bz_central_coaligned']
        self.coaligned_output_bx_2d_integrated = self.idl_save_coaligned_output['bx_integrated_coaligned']
        self.coaligned_output_by_2d_integrated = self.idl_save_coaligned_output['by_integrated_coaligned']
        self.coaligned_output_bz_2d_integrated = self.idl_save_coaligned_output['bz_integrated_coaligned']

    def test_electron_density_center(self):
        for i in range(arrDens_central.shape[0]):
            for j in range(arrDens_central.shape[0]):
                # assert arrays are equivalent to within six decimal places
                self.assertAlmostEqual(self.output_dens_2d_center[i][j],arrDens_central[i][j],6)

        central_dens_coaligned = pd.read_csv(os.path.join(self.repo_path,'Output/FORWARD_MLSO_Rotated_Data/PSI_MLSO_Central_Electron_Density_Coalignment.csv'), sep=',',header=None).values
        # central_dens_coaligned_values = central_dens_coaligned.values
        for i in range(central_dens_coaligned.shape[0]):
            for j in range(central_dens_coaligned.shape[0]):
                # assert arrays are equivalent to within six decimal places
                self.assertAlmostEqual(self.coaligned_output_dens_2d_center[i][j],central_dens_coaligned[i][j],6)

    def test_electron_density_integrated(self):
        for i in range(image_sum.shape[0]):
            for j in range(image_sum.shape[0]):
                # assert arrays are equivalent to within six decimal places
                self.assertAlmostEqual(self.output_dens_integrated_2d[i][j],image_sum[i][j],6)

        integrated_dens_coaligned = pd.read_csv(os.path.join(self.repo_path,'Output/FORWARD_MLSO_Rotated_Data/PSI_MLSO_Integrated_Electron_Density_Coalignment.csv'), sep=',',header=None).values
        # integrated_dens_coaligned_values = integrated_dens_coaligned.values
        for i in range(integrated_dens_coaligned.shape[0]):
            for j in range(integrated_dens_coaligned.shape[0]):
                # assert arrays are equivalent to within six decimal places
                self.assertAlmostEqual(self.coaligned_output_dens_integrated_2d[i][j],integrated_dens_coaligned[i][j],5)

    def test_forward_dens(self):
        for i in range(arrDens_forward.shape[0]):
            for j in range(arrDens_forward.shape[1]):
                self.assertAlmostEqual(self.output_forward_pb_image[i][j],arrDens_forward[i][j],6)

        forward_pb_coaligned = pd.read_csv(os.path.join(self.repo_path,'Output/FORWARD_MLSO_Rotated_Data/PSI_PB_MLSO_Coalignment.csv'), sep=',',header=None).values
        # integrated_dens_coaligned_values = integrated_dens_coaligned.values
        for i in range(forward_pb_coaligned.shape[0]):
            for j in range(forward_pb_coaligned.shape[0]):
                # assert arrays are equivalent to within six decimal places
                self.assertAlmostEqual(self.coaligned_output_forward_pb_image[i][j],forward_pb_coaligned[i][j],5)

    def test_bx_central(self):
        for i in range(bx_central_image_data.shape[0]):
            for j in range(bx_central_image_data.shape[0]):
                # assert arrays are equivalent to within six decimal places
                self.assertAlmostEqual(self.output_bx_2d_center[i][j],bx_central_image_data[i][j],6)

        bx_central_coaligned = pd.read_csv(os.path.join(self.repo_path,'Output/FORWARD_MLSO_Rotated_Data/PSI_Bx_Central_MLSO_Coalignment.csv'), sep=',',header=None).values
        # integrated_dens_coaligned_values = integrated_dens_coaligned.values
        for i in range(bx_central_coaligned.shape[0]):
            for j in range(bx_central_coaligned.shape[0]):
                # assert arrays are equivalent to within six decimal places
                self.assertAlmostEqual(self.coaligned_output_bx_2d_center[i][j],bx_central_coaligned[i][j],5)

    def test_by_central(self):
        for i in range(by_central_image_data.shape[0]):
            for j in range(by_central_image_data.shape[0]):
                # assert arrays are equivalent to within six decimal places
                self.assertAlmostEqual(self.output_by_2d_center[i][j],by_central_image_data[i][j],6)

        by_central_coaligned = pd.read_csv(os.path.join(self.repo_path,'Output/FORWARD_MLSO_Rotated_Data/PSI_By_Central_MLSO_Coalignment.csv'), sep=',',header=None).values
        # integrated_dens_coaligned_values = integrated_dens_coaligned.values
        for i in range(by_central_coaligned.shape[0]):
            for j in range(by_central_coaligned.shape[0]):
                # assert arrays are equivalent to within six decimal places
                self.assertAlmostEqual(self.coaligned_output_by_2d_center[i][j],by_central_coaligned[i][j],5)

    def test_bz_central(self):
        for i in range(bz_central_image_data.shape[0]):
            for j in range(bz_central_image_data.shape[0]):
                # assert arrays are equivalent to within six decimal places
                self.assertAlmostEqual(self.output_bz_2d_center[i][j],bz_central_image_data[i][j],6)

        bz_central_coaligned = pd.read_csv(os.path.join(self.repo_path,'Output/FORWARD_MLSO_Rotated_Data/PSI_Bz_Central_MLSO_Coalignment.csv'), sep=',',header=None).values
        # integrated_dens_coaligned_values = integrated_dens_coaligned.values
        for i in range(bz_central_coaligned.shape[0]):
            for j in range(bz_central_coaligned.shape[0]):
                # assert arrays are equivalent to within six decimal places
                self.assertAlmostEqual(self.coaligned_output_bz_2d_center[i][j],bz_central_coaligned[i][j],5)

    def test_bx_integrated(self):
        for i in range(image_sum_bx.shape[0]):
            for j in range(image_sum_bx.shape[0]):
                # assert arrays are equivalent to within six decimal places
                self.assertAlmostEqual(self.output_bx_2d_integrated[i][j],image_sum_bx[i][j],6)

        bx_integrated_coaligned = pd.read_csv(os.path.join(self.repo_path,'Output/FORWARD_MLSO_Rotated_Data/PSI_Bx_LOS_MLSO_Coalignment.csv'), sep=',',header=None).values
        # integrated_dens_coaligned_values = integrated_dens_coaligned.values
        for i in range(bx_integrated_coaligned.shape[0]):
            for j in range(bx_integrated_coaligned.shape[0]):
                # assert arrays are equivalent to within six decimal places
                self.assertAlmostEqual(self.coaligned_output_bx_2d_integrated[i][j],bx_integrated_coaligned[i][j],5)

    def test_by_integrated(self):
        for i in range(image_sum_by.shape[0]):
            for j in range(image_sum_by.shape[0]):
                # assert arrays are equivalent to within six decimal places
                self.assertAlmostEqual(self.output_by_2d_integrated[i][j],image_sum_by[i][j],6)

        by_integrated_coaligned = pd.read_csv(os.path.join(self.repo_path,'Output/FORWARD_MLSO_Rotated_Data/PSI_By_LOS_MLSO_Coalignment.csv'), sep=',',header=None).values
        # integrated_dens_coaligned_values = integrated_dens_coaligned.values
        for i in range(by_integrated_coaligned.shape[0]):
            for j in range(by_integrated_coaligned.shape[0]):
                # assert arrays are equivalent to within six decimal places
                self.assertAlmostEqual(self.coaligned_output_by_2d_integrated[i][j],by_integrated_coaligned[i][j],5)

    def test_bz_integrated(self):
        for i in range(image_sum_bz.shape[0]):
            for j in range(image_sum_bz.shape[0]):
                # assert arrays are equivalent to within six decimal places
                self.assertAlmostEqual(self.output_bz_2d_integrated[i][j],image_sum_bz[i][j],6)

        bz_integrated_coaligned = pd.read_csv(os.path.join(self.repo_path,'Output/FORWARD_MLSO_Rotated_Data/PSI_Bz_LOS_MLSO_Coalignment.csv'), sep=',',header=None).values
        # integrated_dens_coaligned_values = integrated_dens_coaligned.values
        for i in range(bz_integrated_coaligned.shape[0]):
            for j in range(bz_integrated_coaligned.shape[0]):
                # assert arrays are equivalent to within six decimal places
                self.assertAlmostEqual(self.coaligned_output_bz_2d_integrated[i][j],bz_integrated_coaligned[i][j],5)

if __name__ == '__main__':
    unittest.main()
