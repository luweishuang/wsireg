"""Example functions.
Hopefully these are stable, but if one of them doesn't work anymore you can search the git history for
the commit where it was defined with 'git log -p demo.py'."""
import cv2
import numpy as np
import matplotlib.pyplot as plt

import bilinear
import patchreg
import viz

from scipy.interpolate import interp2d
from skimage.util import view_as_windows


def bilinear_interpolation_of_patch_registration():
    """Primary demo. Computes and applies a global and patch level registration transform,
    applies them to a small region of the whole slide image, and displays the results.

    WARNING: This method only calculates and applies the transformation on a (large) subregion of the image
    as a proof of concept. Even so it uses a lot of memory. The memory usage scales linearly with the image size,
    if that's any comfort."""
    print("Beginning bilinear_interpolation_of_patch_registration...")
    root = "../data/"
    reg1 = cv2.cvtColor(cv2.imread(root + "NG1_1.jpg"), code=cv2.COLOR_BGRA2RGBA)
    reg2 = cv2.cvtColor(cv2.imread(root + "OK1_1.jpg"), code=cv2.COLOR_BGRA2RGBA)

    # Stage One: Low-precision feature alignment
    h, _ =  patchreg.alignFeatures(reg1, reg2)
    height, width = reg1.shape[:2]
    reg2_aligned = cv2.warpPerspective(reg2, h, (width, height))

    # Stage Two: Calculate patch-level registrations
    w_shape = (1000, 1000, 4)
    w_step = (500, 500, 4)

    stack1 = np.concatenate((reg1, reg2_aligned), axis=-1)
    patches = view_as_windows(stack1, window_shape=w_shape, step=w_step)
    morphs = patchreg.calcPlateMorphs(patches)

    # Stage Three: Compute patch-level DVFs
    id_patches = patchreg.calc_id_patches(img_shape=reg2_aligned.shape, patch_size=1000)

    # We copy the morph so it will be applied to both xv and yv sincefirst layer is ignored by applyMorphs.
    map_morphs = np.append(morphs, morphs[:, :, 1, None], axis=2)
    # Apply transformation to identity deformation-result fields.
    reg_patches = patchreg.applyMorphs(id_patches, map_morphs)
    # patches, reg_patches, id_patches = patchreg.test()
    print("reg_patches.shape=", reg_patches.shape)

    # Get subregions (5x5) of patch sets so we don't need to deal with the whole image.
    # You can skip to stage four if you're using the whole image, but beware of memory usage!
    f_img_patches = patches[0:5, 0:5, :1]   # 8:13, 8:13
    m_img_patches = patches[0:5, 0:5, 1:]
    reg_patches = reg_patches[0:5, 0:5, 1:, 500:1500, 500:1500, :]
    # Shift because identity map is location dependent.
    map_patches = reg_patches  #  - 4000

    # Stage Four: Merge patch-level DVFs into a single global transform.
    quilts = bilinear.quilter(map_patches)
    wquilts = bilinear.bilinear_wquilts(map_patches)
    qmaps = [q * w for q, w in zip(quilts, wquilts)]
    tt = qmaps[0] + qmaps[1] + qmaps[2] + qmaps[3]
    print("tt.shape=", tt.shape)
    summed = (qmaps[0] + qmaps[1] + qmaps[2] + qmaps[3]).reshape(2, 3000, 3000).astype(np.float32)

    # Reconstruct images- for demo only, normally we'd just use the full original image and the full patch sets.
    f_img_quilts = bilinear.quilter(f_img_patches)
    f_img_recons = [q * w for q, w in zip(f_img_quilts, wquilts)]
    f_img_recon = (f_img_recons[0] + f_img_recons[1] + f_img_recons[2] + f_img_recons[3]).reshape(3000, 3000, 4).astype(np.uint8)

    m_img_quilts = bilinear.quilter(m_img_patches)
    m_img_recons = [q * w for q, w in zip(m_img_quilts, wquilts)]
    m_img_recon = (m_img_recons[0] + m_img_recons[1] + m_img_recons[2] + m_img_recons[3]).reshape(3000, 3000, 4).astype(np.uint8)

    print("m_img_recon.shape=", m_img_recon.shape)
    reg = cv2.remap(m_img_recon, summed[0], summed[1], interpolation=cv2.INTER_LINEAR)
    print("summed[0].shape=", summed[0].shape)
    print("summed[1].shape=", summed[1].shape)
    print("reg.shape=", reg.shape)
    # Stage 5: Display.
    # Display bilinear mapping details (for x and y dimensions)
    for j in (0,1):
        for i, q in enumerate(quilts):
            plt.subplot(4, 4, i + 1)
            plt.title("Quilt %s" % i)
            plt.imshow(q[j].reshape(3000, 3000))

        for i, wq in enumerate(wquilts):
            plt.subplot(4, 4, i + 5)
            plt.title("Weights %s" % i)
            plt.imshow(wq.reshape(3000, 3000))

        for i, qm in enumerate(qmaps):
            plt.subplot(4, 4, i + 9)
            plt.title("Weighted Quilt %s" % i)
            plt.imshow(qm[j].reshape(3000, 3000))

        plt.subplot(4, 4, 13)
        plt.title("Summed Weighted Quilts")
        plt.imshow(summed[j].reshape(3000, 3000))
        plt.show()

    # Display registered images.
    plt.gcf().set_size_inches(15, 15)
    plt.subplot(2, 2, 1)
    plt.title("Fixed Image")
    plt.imshow(f_img_recon)
    plt.grid()
    plt.subplot(2, 2, 2)
    plt.title("Moving Image")
    plt.imshow(m_img_recon)
    plt.grid()
    plt.subplot(2, 2, 3)
    plt.title("Without Patch Transforms")
    plt.imshow(viz.overlay([f_img_recon, m_img_recon]))
    # plt.grid()
    plt.subplot(2, 2, 4)
    plt.title("With Patch Transforms")
    plt.imshow(viz.overlay([f_img_recon, reg]))
    plt.grid()

    plt.savefig(root + "bilinear_overlay.png")
    plt.show()



def deform_image():
    """Generate a random distortion DVF and apply it to an image."""
    # x_peaks, y_peaks = 5, 5
    # dx, dy = 5, 5
    # spline_length = 10
    # x, y = 50, 40

    x_peaks, y_peaks = 390, 500
    dx, dy = 390, 500
    spline_length = 10
    x, y = 3946, 5240

    # We'll just use a small patch from the image.
    img = cv2.imread("../data/OK1_1.jpg")[:y, :x]
    grid_x_id, grid_y_id = np.mgrid[0:x_peaks-1:complex(x_peaks), 0:y_peaks-1:complex(y_peaks)] * spline_length
    print("grid_x_id.min(), grid_x_id.max()==", grid_x_id.min(), grid_x_id.max())
    print("grid_y_id.min(), grid_y_id.max()==", grid_y_id.min(), grid_y_id.max())
    # perturb
    grid_x_peaks = grid_x_id + np.random.randint(-dx, dx+1, size=(x_peaks,y_peaks))
    grid_y_peaks = grid_y_id + np.random.randint(-dx, dx+1, size=(x_peaks,y_peaks))
    print("grid_x_peaks.min(), grid_x_peaks.max()==", grid_x_peaks.min(), grid_x_peaks.max())
    print("grid_y_peaks.min(), grid_y_peaks.max()==", grid_y_peaks.min(), grid_y_peaks.max())
    #interpolate
    grid_x = interp2d(grid_x_id, grid_y_id, grid_x_peaks)(np.linspace(0, x-1, x), np.linspace(0, y-1, y)).astype(np.float32)
    grid_y = interp2d(grid_x_id, grid_y_id, grid_y_peaks)(np.linspace(0, x-1, x), np.linspace(0, y-1, y)).astype(np.float32)
    #apply maps
    print("img.shape=", img.shape)
    print("grid_x.shape=", grid_x.shape)
    out = cv2.remap(img, grid_x, grid_y, interpolation=cv2.INTER_LINEAR)
    print("grid_x.min(), grid_x.max()==", grid_x.min(), grid_x.max())
    print("grid_y.min(), grid_y.max()==", grid_y.min(), grid_y.max())
    print("out.min(), out.max()==", out.min(), out.max())
    #show results
    fig, axes = plt.subplots(3,2)
    for im, ax in zip([grid_x_id, grid_y_id, grid_x, grid_y, img, out], axes.flatten()):
        ax.imshow(im)
    fig.show()
    plt.show()
    cv2.imwrite("yy.jpg", out)


# deform_image()
bilinear_interpolation_of_patch_registration()

