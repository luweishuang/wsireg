import cv2
import numpy as np
import bilinear
import patchreg
from skimage.util import view_as_windows


def bilinear_interpolation_of_patch_registration(master_srcdata, target_srcdata):
    print("Beginning bilinear_interpolation_of_patch_registration...")
    w_shape = (1000, 1000, 4)    # window_size
    w_step = (500, 500, 4)       # 步长
    padding = w_step[0]
    # master_data = cv2.copyMakeBorder(master_srcdata, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    # target_data = cv2.copyMakeBorder(target_srcdata, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    master_data = cv2.copyMakeBorder(master_srcdata, padding, padding, padding, padding, cv2.BORDER_REFLECT)
    target_data = cv2.copyMakeBorder(target_srcdata, padding, padding, padding, padding, cv2.BORDER_REFLECT)
    master_img = cv2.cvtColor(master_data, code=cv2.COLOR_BGRA2RGBA)
    target_img = cv2.cvtColor(target_data, code=cv2.COLOR_BGRA2RGBA)

    # Stage One: Low-precision feature alignment
    h, _ =  patchreg.alignFeatures(target_img, master_img)
    height, width = target_img.shape[:2]
    master_aligned = cv2.warpPerspective(master_img, h, (width, height))

    # Stage Two: Calculate patch-level registrations
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
    print("tt.shape=", tt.shape)
    summed = (qmaps[0] + qmaps[1] + qmaps[2] + qmaps[3]).reshape(2, 3000, 5000).astype(np.float32)

    master_remap = cv2.remap(master_img, summed[0], summed[1], interpolation=cv2.INTER_LINEAR)    # summed 是坐标映射关系
    master_reg = master_remap[padding:height-padding, padding:width-padding, :]
    # cv2.imwrite("target.jpg", target_img[padding:height-padding, padding:width-padding, :])
    # cv2.imwrite("master.jpg", master_img[padding:height-padding, padding:width-padding, :])
    # cv2.imwrite("master_reg.jpg", master_reg)
    return master_reg


def draw_img():
    master_srcdata = cv2.imread("../data/OK1_1_32.jpg")
    padding = 500
    master_data = cv2.copyMakeBorder(master_srcdata, padding, padding, padding, padding, cv2.BORDER_CONSTANT,value=(255, 255, 255))

    cv2.line(master_data, (0, 1000), (5000, 1000), (0, 255, 0), 2)
    cv2.line(master_data, (0, 2000), (5000, 2000), (0, 255, 0), 2)
    cv2.line(master_data, (1000, 0), (1000, 3000), (0, 255, 0), 2)
    cv2.line(master_data, (2000, 0), (2000, 3000), (0, 255, 0), 2)
    cv2.line(master_data, (3000, 0), (3000, 3000), (0, 255, 0), 2)
    cv2.line(master_data, (4000, 0), (4000, 3000), (0, 255, 0), 2)
    cv2.imwrite("master_data.jpg", master_data)


def pad_imgs(master_srcdata, target_srcdata):
    master3 = master_srcdata[4050:4850,:,:]
    # cv2.imwrite("master3.jpg", master3)
    target3 = target_srcdata[4470:5270,:,:]
    # cv2.imwrite("target3.jpg", target3)

    master_h, master_w, _ = master3.shape
    target_h, target_w, _ = target3.shape
    assert master_h == target_h and master_w == target_w

    src_w = master_w
    src_h = master_h
    mid_h = 2000     # at least 2000
    mid_w = 4000
    assert mid_w > src_w and mid_h > src_h
    left_pad = int((mid_w-src_w)/2)
    right_pad = int(mid_w - src_w - left_pad)
    top_pad = int((mid_h - src_h) / 2)
    down_pad = int(mid_h - src_h - top_pad)
    master3_pad = cv2.copyMakeBorder(master3, top_pad, down_pad, left_pad, right_pad, cv2.BORDER_REFLECT)
    target3_pad = cv2.copyMakeBorder(target3, top_pad, down_pad, left_pad, right_pad, cv2.BORDER_REFLECT)
    return master3_pad, target3_pad, top_pad, down_pad, left_pad, right_pad



if __name__ == "__main__":
    # draw_img()
    # exit()

    root = "../data/"
    master_srcdata = cv2.imread(root + "OK1_1.jpg")
    target_srcdata = cv2.imread(root + "NG1_1.jpg")
    master3_pad, target3_pad, top_pad, down_pad, left_pad, right_pad = pad_imgs(master_srcdata, target_srcdata)
    # cv2.imwrite("master3_pad.jpg", master3_pad)
    # cv2.imwrite("target3_pad.jpg", target3_pad)
    # exit()

    master_reg_pad = bilinear_interpolation_of_patch_registration(master3_pad, target3_pad)
    master_reg = master_reg_pad[top_pad: master_height - down_pad, :, : ]
