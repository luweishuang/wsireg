import cv2
import numpy as np
import bilinear
import patchreg
import matplotlib.pyplot as plt
from skimage.util import view_as_windows


def bilinear_interpolation_of_patch_registration():
    print("Beginning bilinear_interpolation_of_patch_registration...")
    root = "../data/"
    master_srcdata = cv2.imread(root + "OK1_1_32.jpg")
    target_srcdata = cv2.imread(root + "NG1_1_32.jpg")
    padding = 500
    master_data = cv2.copyMakeBorder(master_srcdata, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    target_data = cv2.copyMakeBorder(target_srcdata, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    master_img = cv2.cvtColor(master_data, code=cv2.COLOR_BGRA2RGBA)
    target_img = cv2.cvtColor(target_data, code=cv2.COLOR_BGRA2RGBA)

    # Stage One: Low-precision feature alignment
    h, _ =  patchreg.alignFeatures(target_img, master_img)
    height, width = target_img.shape[:2]
    master_aligned = cv2.warpPerspective(master_img, h, (width, height))

    # Stage Two: Calculate patch-level registrations
    w_shape = (1000, 1000, 4)    # window_size
    w_step = (500, 500, 4)       # 步长

    stack1 = np.concatenate((target_img, master_aligned), axis=-1)  # (2000, 40000, 8)
    patches = view_as_windows(stack1, window_shape=w_shape, step=w_step)
    print("patches.shape=", patches.shape)   #(3, 7, 2, 1000, 1000, 4) 第一个shape是行切割数；第二个shape列切割数；最后3个是切割的窗口大小
    morphs = patchreg.calcPlateMorphs(patches)   # (3,7,2,3,3)
    tt = morphs[:, :, 1, None]
    print("morphs.min(), morphs.max()==", tt.min(), tt.max())

    # Stage Three: Compute patch-level DVFs=dense displacement vector field
    id_patches = patchreg.calc_id_patches_src(img_shape=master_aligned.shape, patch_size=1000)  # (3,7,3,2000,2000,1)
    print("id_patches.shape=", id_patches.shape)
    print("id_patches.min(), id_patches.max()==", id_patches.min(), id_patches.max())

    map_morphs = np.append(morphs, morphs[:, :, 1, None], axis=2)  # (3,7,3,3,3)
    reg_patches_src = patchreg.applyMorphs(id_patches, map_morphs)   # (3,7,3,2000,2000,1)
    print("reg_patches_src.shape=", reg_patches_src.shape)

    map_patches_src = reg_patches_src - id_patches
    # Restrict to actual patch regions (remove buffers).
    map_patches = reg_patches_src[:, :, 1:, 500:1500, 500:1500, :]
    # map_patches = map_patches_src[:, :, 1:, 500:1500, 500:1500, :]   # (3,7,2,1000,1000,1)
    print("map_patches.min(), map_patches.max()==", map_patches.min(), map_patches.max())
    print("map_patches.shape=", map_patches.shape)

    # Stage Four: Merge patch-level DVFs into a single global transform.
    quilts = bilinear.quilter(map_patches)
    print("quilts[0].min(), quilts[0].max()==", quilts[0].min(), quilts[0].max())
    print("quilts[1].min(), quilts[1].max()==", quilts[1].min(), quilts[1].max())
    print("quilts[2].min(), quilts[2].max()==", quilts[2].min(), quilts[2].max())
    print("quilts[3].min(), quilts[3].max()==", quilts[3].min(), quilts[3].max())
    wquilts = bilinear.bilinear_wquilts(map_patches)
    wquilts1 = [wquilt.reshape(wquilt.shape[1:3]) for wquilt in wquilts]
    summed_wq = wquilts1[0] + wquilts1[1] + wquilts1[2] + wquilts1[3]
    print("summed_wq.shape=", summed_wq.shape)
    print("summed_wq.min(), summed_wq.max()==", summed_wq.min(), summed_wq.max())

    qmaps = [q * w for q, w in zip(quilts, wquilts)]
    tt = qmaps[0] + qmaps[1] + qmaps[2] + qmaps[3]
    print("tt.shape=", tt.shape)
    summed = (qmaps[0] + qmaps[1] + qmaps[2] + qmaps[3]).reshape(2, 3000, 5000).astype(np.float32)
    print("summed[0].min(), summed[0].max()==", summed[0].min(), summed[0].max())
    print("summed[1].min(), summed[1].max()==", summed[1].min(), summed[1].max())

    reg = cv2.remap(master_img, summed[0], summed[1], interpolation=cv2.INTER_LINEAR)    # summed 是坐标映射关系
    cv2.imwrite("target.jpg", target_img[500:2500, 500:4500, :])
    cv2.imwrite("master.jpg", master_img[500:2500, 500:4500, :])
    cv2.imwrite("reg.jpg", reg[500:2500, 500:4500, :])


def partially_overlapping_strips():
    """Shows what partially overlapping strips looks like."""
    # Assuming equally sized strips,
    # composing 2 functions this way yields 4 different functions.
    y = np.zeros((20,100,3), dtype=np.uint8)
    shift = np.zeros((10,100,3), dtype=np.uint8)

    bx = np.full((20,100,3), [255,0,0], dtype=np.uint8)
    bstrip = np.vstack((bx,y))
    btile = np.tile(bstrip, (3,1,1))
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
    bilinear_interpolation_of_patch_registration()