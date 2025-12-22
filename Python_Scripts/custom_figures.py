import os
import git
from pypdf import PdfReader, PdfWriter, PageObject
from pypdf.generic import RectangleObject

repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir

def tighten_page(page):
    # Use the bounding box of the content as the crop box
    bbox = page.mediabox
    page.cropbox = RectangleObject(bbox)
    page.trimbox = RectangleObject(bbox)
    page.bleedbox = RectangleObject(bbox)
    page.artbox = RectangleObject(bbox)
    return page


def create_side_by_side_figure(input1, input2, output):
    left_pdf = input1
    right_pdf = input2

    L = tighten_page(PdfReader(left_pdf).pages[0])
    R = tighten_page(PdfReader(right_pdf).pages[0])

    # # Crop boxes (so you don't carry excess whitespace)
    # L.cropbox = L.mediabox
    # R.cropbox = R.mediabox

    Lw, Lh = float(L.cropbox.width), float(L.cropbox.height)
    Rw, Rh = float(R.cropbox.width), float(R.cropbox.height)

    gap = 18  # points (~0.25 inch). adjust to mimic LaTeX spacing

    # Choose output height as max of the two; keep each at native scale
    out_h = max(Lh, Rh)
    out_w = Lw + gap + Rw

    new_page = PageObject.create_blank_page(width=out_w, height=out_h)

    # Center vertically if heights differ
    new_page.merge_translated_page(L, 0, (out_h - Lh) / 2)
    new_page.merge_translated_page(R, Lw + gap, (out_h - Rh) / 2)

    writer = PdfWriter()
    writer.add_page(new_page)

    with open(output, "wb") as f:
        writer.write(f)

left_pdf = os.path.join(repo_path,"Output/Plots/Features_Angle_Error_2017_08_20_COR1_ne_PSI.pdf")
right_pdf = os.path.join(repo_path,"Output/Plots/Features_Angle_Error_2017_08_20_COR1_COR1.pdf")
output = os.path.join(repo_path,"Output/Plots/Angle_Error_Example.pdf")
create_side_by_side_figure(left_pdf, right_pdf, output)

left_pdf = os.path.join(repo_path,"Output/Plots/COR1_Combined_JSD_no_random_heatmap.pdf")
right_pdf = os.path.join(repo_path,"Output/Plots/COR1_Combined_JSD_heatmap.pdf")
output = os.path.join(repo_path,"Output/Plots/JSD_Heatmap_Example.pdf")
create_side_by_side_figure(left_pdf, right_pdf, output)
