import cv2
import numpy as np
import bilinear
import patchreg
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
    morphs = patchreg.calcPlateMorphs(patches)   # (3,7,2,3,3)
    # tt = morphs[:, :, 1, None]
    # print("morphs.min(), morphs.max()==", tt.min(), tt.max())

    # Stage Three: Compute patch-level DVFs=dense displacement vector field
    id_patches = patchreg.calc_id_patches_src(img_shape=master_aligned.shape, patch_size=1000)  # (3,7,3,2000,2000,1)

    map_morphs = np.append(morphs, morphs[:, :, 1, None], axis=2)  # (3,7,3,3,3)
    reg_patches_src = patchreg.applyMorphs(id_patches, map_morphs)   # (3,7,3,2000,2000,1)

    map_patches = reg_patches_src[:, :, 1:, 500:1500, 500:1500, :]

    # Stage Four: Merge patch-level DVFs into a single global transform.
    quilts = bilinear.quilter(map_patches)
    wquilts = bilinear.bilinear_wquilts(map_patches)
    qmaps = [q * w for q, w in zip(quilts, wquilts)]
    tt = qmaps[0] + qmaps[1] + qmaps[2] + qmaps[3]
    # print("tt.shape=", tt.shape)
    summed = (qmaps[0] + qmaps[1] + qmaps[2] + qmaps[3]).reshape(2, 3000, 5000).astype(np.float32)

    master_reg = cv2.remap(master_img, summed[0], summed[1], interpolation=cv2.INTER_LINEAR)    # summed 是坐标映射关系
    cv2.imwrite("target.jpg", target_img[500:2500, 500:4500, :])
    cv2.imwrite("master.jpg", master_img[500:2500, 500:4500, :])
    cv2.imwrite("master_reg.jpg", master_reg[500:2500, 500:4500, :])


if __name__ == "__main__":
    # partially_overlapping_strips()
    bilinear_interpolation_of_patch_registration()