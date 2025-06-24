import os
from matplotlib import pyplot as plt
import torch
import h5py
def vis_data(pd, pdfs, pd_name, pdfs_name, slice, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    title = pd_name.split('/')[-1] + ' and ' + pdfs_name.split('/')[-1] + ':' + str(slice)

    fig, axs = plt.subplots(1, 2, figsize=(20, 20))
    axs[0].imshow(pd, cmap='gray')
    axs[1].imshow(pdfs, cmap='gray')
    plt.suptitle(title)
    figname = pd_name.split('/')[-1] + '_' + pdfs_name.split('/')[-1] + '_' + str(slice) + '.png'

    figpath = os.path.join(output_dir, figname)

    plt.savefig(figpath)
    plt.close('all')

def vis_img(img, fname, ftype ,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.imshow(img, cmap='gray')
    figname = fname + '_' + ftype + '.png'
    figpath = os.path.join(output_dir, figname)
    plt.savefig(figpath)



import os
import torch
import numpy as np
from PIL import Image

def save_reconstructions(reconstructions, out_dir):
    # """
    # Save reconstruction images.

    # This function writes to h5 files that are appropriate for submission to the
    # leaderboard.

    # Args:
    #     reconstructions (dict[str, np.array]): A dictionary mapping input
    #         filenames to corresponding reconstructions (of shape num_slices x
    #         height x width).
    #     out_dir (pathlib.Path): Path to the output directory where the
    #         reconstructions should be saved.
    # """
    # os.makedirs(str(out_dir), exist_ok=True)
    # print(out_dir)
    # for fname in reconstructions.keys():
    #     f_output = torch.stack([v for _, v in reconstructions[fname].items()])

    #     basename = fname.split('/')[-1]
    #     with h5py.File(str(out_dir) + '/' + str(basename) + '.hdf5', "w") as f:
    #         print(fname)
    #         f.create_dataset("reconstruction", data=f_output.cpu())

    os.makedirs(str(out_dir), exist_ok=True)
    print("Saving to:", out_dir)

    for fname in reconstructions.keys():
        slices = reconstructions[fname]  # dict[int, tensor] or list of tensors
        stacked = torch.stack([v for _, v in sorted(slices.items())])  # [N, H, W]

        basename = os.path.splitext(os.path.basename(fname))[0]
        volume_dir = os.path.join(out_dir, basename)
        os.makedirs(volume_dir, exist_ok=True)

        for i, slice_tensor in enumerate(stacked):
            img = slice_tensor.cpu().numpy()
            img = normalize_to_uint8(img)
            img_pil = Image.fromarray(img)
            img_pil.save(os.path.join(volume_dir, f"{i:03d}.png"))

def normalize_to_uint8(img):
    """
    Normalize a single 2D image to uint8 for saving as PNG.
    """
    img = np.abs(img)  # remove imaginary part if needed
    img = (img - img.min()) / (img.ptp() + 1e-8)  # normalize to [0,1]
    return (img * 255).astype(np.uint8)