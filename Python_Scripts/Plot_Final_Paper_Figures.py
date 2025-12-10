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
import os
import git
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir


os.path.join(repo_path, 'Output/Plots/QRaFT_Figures/2017-09-06_ne_COR1_fig_1.png')
# Load the images
img1 = mpimg.imread(os.path.join(repo_path, 'Output/Plots/QRaFT_Figures/2017-09-06_ne_COR1_fig_1.png'))
img2 = mpimg.imread(os.path.join(repo_path, 'Output/Plots/QRaFT_Figures/2017-09-06_ne_LOS_COR1_fig_1.png'))
img3 = mpimg.imread(os.path.join(repo_path, 'Output/Plots/QRaFT_Figures/2017-09-06_pB_COR1_fig_1.png'))
img4 = mpimg.imread(os.path.join(repo_path, 'Output/Plots/QRaFT_Figures/2017-09-06_COR1_fig_1.png'))

# Create a figure and four subplots
fig, axs = plt.subplots(4, 1, figsize=(12, 12))

# Display each image in a subplot
axs[0].imshow(img1)
axs[1].imshow(img2)
axs[2].imshow(img3)
axs[3].imshow(img4)

# Remove the x and y ticks
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

# Adjust the spacing between subplots
# plt.subplots_adjust(hspace=0, wspace=0)
plt.tight_layout()
plt.savefig(os.path.join(repo_path, 'Output/Plots/QRaFT_Figures/Test_Combined_QraFT_Fig.png'))



# plot a side by side figure
# Load the images
img1 = mpimg.imread(os.path.join(repo_path, 'Output/Plots/QRaFT_Figures/2017-08-29_ne_COR1_fig_5.png'))
img2 = mpimg.imread(os.path.join(repo_path, 'Output/Plots/QRaFT_Figures/2017-08-29_COR1_fig_5.png'))

# Create a figure and two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Display each image in a subplot
axs[0].imshow(img1)
axs[1].imshow(img2)

for ax in axs:
    ax.axis('off')   # turn off  ticks, labels, and borders

plt.subplots_adjust(wspace=0, hspace=0) # remove padding between images

plt.savefig(
    os.path.join(repo_path, 'Output/Plots/Test_Combined_QraFT_Fig.eps'),
    format='eps',
    bbox_inches='tight',
    pad_inches=0
)
plt.close()


def Create1x2Figure(image1, image2, output_file):
    # Load the images
    img1 = mpimg.imread(image1)
    img2 = mpimg.imread(image2)

    # Create a figure and two subplots
    fig, axs = plt.subplots(1, 2, figsize=(8, 8))

    # Display each image in a subplot
    axs[0].imshow(img1)
    axs[1].imshow(img2)

    # # set titles for each axes
    # axs[0].set_title(title1)
    # axs[1].set_title(title2)


    # Remove the x and y ticks
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(output_file, format='eps')

image_path_1 = os.path.join(repo_path, 'Output/Plots/Features_Angle_Error_2017_08_20_COR1_ne_PSI.eps')
image_path_2 = os.path.join(repo_path, 'Output/Plots/Features_Angle_Error_2017_08_20_COR1_COR1.eps')
output_file = os.path.join(repo_path, 'Output/Plots/Test_Combined_Angle_Error_Fig.eps')
Create1x2Figure(image_path_1, image_path_2, output_file)


import numpy as np
import matplotlib.pyplot as plt

def rot_POS_misalignment(alpha_deg):
    """
    Compute the angular distortion in the plane-of-sky (POS) projection
    due to solar rotation over a specified angle.

    Parameters:
    - alpha_deg: rotation angle in degrees (e.g., 4.4 for 8 hours of rotation)

    Returns:
    - theta_deg: array of latitudes (0 to 89 degrees)
    - dtheta_deg: array of angular distortions in degrees
    """
    alpha = np.radians(alpha_deg)
    theta_arr = np.radians(np.arange(90))  # 0° to 89° in radians

    dtheta = np.arctan(np.tan(theta_arr) / np.cos(alpha)) - theta_arr
    dtheta_deg = np.degrees(dtheta)
    theta_deg = np.degrees(theta_arr)

    return theta_deg, dtheta_deg

# Calculate rotation misalignment
theta_deg, dtheta_deg = rot_POS_misalignment(4.4)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(theta_deg, dtheta_deg)
plt.title(r'Change in POS angle $\Delta \theta$ due to solar rotation ($\alpha = 4.4^\circ$)')
plt.xlabel('Colatitude, degrees')
plt.ylabel(r'Change in $\Delta \theta$, degrees')
plt.grid(True, color='gray', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
