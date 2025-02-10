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




# Plot a side by side figure with two png files 
# Load the images
img1 = mpimg.imread(os.path.join(repo_path, 'Output/Plots/Updated_COR1_PSI_vs_FORWARD_Feature_Tracing_Performance.png'))
img2 = mpimg.imread(os.path.join(repo_path, 'Output/Plots/Updated_COR1_PSI_vs_FORWARD_Feature_Tracing_Performance_Log.png'))

# Create a figure and two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Display each image in a subplot
axs[0].imshow(img1)
axs[1].imshow(img2)

# Remove the x and y ticks
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

# Adjust the spacing between subplots
# plt.subplots_adjust(hspace=0, wspace=0)
plt.tight_layout()
plt.savefig(os.path.join(repo_path, 'Output/Plots/Test_Combined_Performance_Fig.png'))



# plot a side by side figure
# Load the images
img1 = mpimg.imread(os.path.join(repo_path, 'Output/Plots/QRaFT_Figures/2017-08-20_ne_COR1_fig_5.png'))
img2 = mpimg.imread(os.path.join(repo_path, 'Output/Plots/QRaFT_Figures/2017-08-20_COR1_fig_5.png'))

# Create a figure and two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Display each image in a subplot
axs[0].imshow(img1)
axs[1].imshow(img2)

# Remove the x and y ticks
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(repo_path, 'Output/Plots/Test_Combined_QraFT_Fig.eps'), format='eps')


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

image_path_1 = os.path.join(repo_path, 'Output/Plots/Features_Angle_Error_2017_08_20_COR1_ne_PSI.png')
image_path_2 = os.path.join(repo_path, 'Output/Plots/Features_Angle_Error_2017_08_20_COR1_COR1.png')
output_file = os.path.join(repo_path, 'Output/Plots/Test_Combined_Angle_Error_Fig.eps')
Create1x2Figure(image_path_1, image_path_2, output_file)
