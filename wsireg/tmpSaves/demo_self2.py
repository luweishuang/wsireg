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
    # reg1 = cv2.cvtColor(cv2.imread(root + "NG1_1.jpg"), code=cv2.COLOR_BGRA2RGBA)
    # reg2 = cv2.cvtColor(cv2.imread(root + "OK1_1.jpg"), code=cv2.COLOR_BGRA2RGBA)
    reg1 = cv2.cvtColor(cv2.imread(root + "NG1_1_3.jpg"), code=cv2.COLOR_BGRA2RGBA)
    reg2 = cv2.cvtColor(cv2.imread(root + "OK1_1_3.jpg"), code=cv2.COLOR_BGRA2RGBA)

    # Stage One: Low-precision feature alignment
    h, _ =  patchreg.alignFeatures(reg1, reg2)
    height, width = reg1.shape[:2]
    reg2_aligned = cv2.warpPerspective(reg2, h, (width, height))

    # Stage Two: Calculate patch-level registrations
    w_shape = (1000, 1000, 4)    # window_size
    w_step = (500, 500, 4)       # 步长

    stack1 = np.concatenate((reg1, reg2_aligned), axis=-1)  # (5240, 3946, 8)
    print("stack1.min(), stack1.max()==", stack1.min(), stack1.max())
    patches = view_as_windows(stack1, window_shape=w_shape, step=w_step)
    print("patches.shape=", patches.shape)   #(9, 6, 2, 1000, 1000, 4) 第一个shape是行切割数；第二个shape列切割数；最后3个是切割的窗口大小
    morphs = patchreg.calcPlateMorphs(patches)   # (9,6,2,3,3)
    # morphs[:, :, 0, None] == np.identity(3) 没有有用信息
    tt = morphs[:, :, 1, None]
    print("morphs.min(), morphs.max()==", tt.min(), tt.max())

    # Stage Three: Compute patch-level DVFs=dense displacement vector field
    id_patches = patchreg.calc_id_patches_src(img_shape=reg2_aligned.shape, patch_size=1000)  # (9,6,3,2000,2000,1)
    print("id_patches.shape=", id_patches.shape)
    # id_patches[:, :, 0, None] == id_patches[:, :, 1, None]
    print("id_patches.min(), id_patches.max()==", id_patches.min(), id_patches.max())

    # We copy the morph so it will be applied to both xv and yv since first layer is ignored by applyMorphs.
    map_morphs = np.append(morphs, morphs[:, :, 1, None], axis=2)  # (9,6,3,3,3)
    # map_morphs[:, :, 1, None] == map_morphs[:, :, 2, None]
    # Apply transformation to identity deformation-result fields.
    reg_patches_src = patchreg.applyMorphs(id_patches, map_morphs)   # (9,6,3,2000,2000,1)
    print("reg_patches_src.shape=", reg_patches_src.shape)
    # ll = reg_patches_src[:, :, 1, None]
    # lt = reg_patches_src[:, :, 2, None]
    # ii = ll - lt
    # print("ii.min(), ii.max()==", ii.min(), ii.max())

    # map_patches_src = reg_patches_src - id_patches
    reg_patches = reg_patches_src[:, :, 1:, 500:1500, 500:1500, :]
    # Restrict to actual patch regions (remove buffers).
    map_patches = reg_patches    # (9,6,2,1000,1000,1)


    # print("reg_patches.min(), reg_patches.max()==", reg_patches.min(), reg_patches.max())
    print("map_patches.min(), map_patches.max()==", map_patches.min(), map_patches.max())
    print("map_patches.shape=", map_patches.shape)

    # Stage Four: Merge patch-level DVFs into a single global transform.
    quilts = bilinear.quilter(map_patches)
    wquilts = bilinear.bilinear_wquilts(map_patches)
    wquilts1 = [wquilt.reshape(wquilt.shape[1:3]) for wquilt in wquilts]
    summed_wq = wquilts1[0] + wquilts1[1] + wquilts1[2] + wquilts1[3]
    print("summed_wq.shape=", summed_wq.shape)
    print("summed_wq.min(), summed_wq.max()==", summed_wq.min(), summed_wq.max())

    qmaps = [q * w for q, w in zip(quilts, wquilts)]
    print("quilts[0].shape=",quilts[0].shape)
    print("wquilts[0].shape=",wquilts[0].shape)
    print("qmaps[0].shape=",qmaps[0].shape)
    tt = qmaps[0] + qmaps[1] + qmaps[2] + qmaps[3]
    print("tt.shape=", tt.shape)
    summed = (qmaps[0] + qmaps[1] + qmaps[2] + qmaps[3]).reshape(2, 5000, 3500).astype(np.float32)
    print("summed[0].min(), summed[0].max()==", summed[0].min(), summed[0].max())
    print("summed[1].min(), summed[1].max()==", summed[1].min(), summed[1].max())
    # If you're using the whole image and the whole patches object (but again, beware of memory usage!):
    # f_img_recon = reg1
    # m_img_recon = reg2
    # print("m_img_recon.shape==", m_img_recon.shape)

    f_img_patches = patches[:, :, :1]
    f_img_quilts = bilinear.quilter(f_img_patches)
    f_img_recons = [q * w for q, w in zip(f_img_quilts, wquilts)]
    f_img_recon = (f_img_recons[0] + f_img_recons[1] + f_img_recons[2] + f_img_recons[3]).reshape(5000, 3500, 4).astype(np.uint8)
    m_img_patches = patches[:, :, 1:]
    print("m_img_patches.shape==", m_img_patches.shape)
    m_img_quilts = bilinear.quilter(m_img_patches)
    print("m_img_quilts[0].shape==", m_img_quilts[0].shape)
    m_img_recons = [q * w for q, w in zip(m_img_quilts, wquilts)]
    print("m_img_recons[0].shape==", m_img_recons[0].shape)
    m_img_recon = (m_img_recons[0] + m_img_recons[1] + m_img_recons[2] + m_img_recons[3]).reshape(5000, 3500, 4).astype(np.uint8)
    print("m_img_recon.shape==", m_img_recon.shape)

    print("summed.shape==", summed.shape)
    reg = cv2.remap(m_img_recon, summed[0], summed[1], interpolation=cv2.INTER_LINEAR)    # summed 是坐标映射关系
    print("summed[0].min(), summed[0].max()==", summed[0].min(), summed[0].max())
    print("summed[1].min(), summed[1].max()==", summed[1].min(), summed[1].max())
    # print("m_img_recon.min(), m_img_recon.max()==", m_img_recon.min(), m_img_recon.max())
    # print("f_img_recon.min(), f_img_recon.max()==", f_img_recon.min(), f_img_recon.max())
    # print("reg.min(), reg.max()==", reg.min(), reg.max())
    cv2.imwrite("f_img_recon.png", f_img_recon)
    cv2.imwrite("m_img_recon.png", m_img_recon)
    cv2.imwrite("reg.png", reg)


def deform_image():
    """Generate a random distortion DVF and apply it to an image."""
    x_peaks, y_peaks = 5, 5
    dx, dy = 5, 5
    spline_length = 10
    x, y = 50, 40

    # We'll just use a small patch from the image.
    img = cv2.imread("../data/OK1_1.jpg")[:y, :x]
    grid_x_id, grid_y_id = np.mgrid[0:x_peaks-1:complex(x_peaks), 0:y_peaks-1:complex(y_peaks)] * spline_length
    # perturb
    grid_x_peaks = grid_x_id + np.random.randint(-dx, dx+1, size=(x_peaks,y_peaks))
    grid_y_peaks = grid_y_id + np.random.randint(-dx, dx+1, size=(x_peaks,y_peaks))
    #interpolate
    grid_x = interp2d(grid_x_id, grid_y_id, grid_x_peaks)(np.linspace(0, x-1, x), np.linspace(0, y-1, y)).astype(np.float32)
    grid_y = interp2d(grid_x_id, grid_y_id, grid_y_peaks)(np.linspace(0, x-1, x), np.linspace(0, y-1, y)).astype(np.float32)
    #apply maps
    out = cv2.remap(img, grid_x, grid_y, interpolation=cv2.INTER_LINEAR)
    #show results
    fig, axes = plt.subplots(3,2)
    for im, ax in zip([grid_x_id, grid_y_id, grid_x, grid_y, img, out], axes.flatten()):
        ax.imshow(im)
    fig.show()
    plt.show()


def partially_overlapping_strips():
    """Shows what partially overlapping strips looks like."""
    # Assuming equally sized strips,
    # composing 2 functions this way yields 4 different functions.
    y = np.zeros((20,100,3), dtype=np.uint8)
    shift = np.zeros((10,100,3), dtype=np.uint8)

    bx = np.full((20,100,3), [255,0,0], dtype=np.uint8)
    bstrip = np.vstack((bx,y))
    btile = np.tile(bstrip, (3,1,1))   # 把数组沿各个方向复制
    blue = np.vstack((btile,shift))

    rx = np.full((20,100,3), [0,0,255], dtype=np.uint8)
    rstrip = np.vstack((rx,y))
    rtile = np.tile(rstrip, (3,1,1))
    red = np.vstack((shift,rtile))

    both = red + blue

    plt.imshow(blue)
    plt.show()
    plt.imshow(red)
    plt.show()
    plt.imshow(both)
    plt.show()



if __name__ == "__main__":
    # partially_overlapping_strips()
    # deform_image()
    bilinear_interpolation_of_patch_registration()