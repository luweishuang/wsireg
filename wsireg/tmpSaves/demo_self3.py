import cv2
import numpy as np
import bilinear
import patchreg
from skimage.util import view_as_windows


def bilinear_interpolation_of_patch_registration():
    print("Beginning bilinear_interpolation_of_patch_registration...")
    root = "../data/"
    master_img = cv2.cvtColor(cv2.imread(root + "OK1_1_32.jpg"), code=cv2.COLOR_BGRA2RGBA)
    target_img = cv2.cvtColor(cv2.imread(root + "NG1_1_32.jpg"), code=cv2.COLOR_BGRA2RGBA)

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
    summed = (qmaps[0] + qmaps[1] + qmaps[2] + qmaps[3]).reshape(2, 2000, 4000).astype(np.float32)
    print("summed[0].min(), summed[0].max()==", summed[0].min(), summed[0].max())
    print("summed[1].min(), summed[1].max()==", summed[1].min(), summed[1].max())
    # If you're using the whole image and the whole patches object (but again, beware of memory usage!):

    # f_img_patches = patches[:, :, :1]
    # f_img_quilts = bilinear.quilter(f_img_patches)
    # f_img_recons = [q * w for q, w in zip(f_img_quilts, wquilts)]
    # f_img_recon = (f_img_recons[0] + f_img_recons[1] + f_img_recons[2] + f_img_recons[3]).reshape(2000, 4000, 4).astype(np.uint8)
    # m_img_patches = patches[:, :, 1:]
    # print("m_img_patches.shape==", m_img_patches.shape)
    # m_img_quilts = bilinear.quilter(m_img_patches)
    # print("m_img_quilts[0].shape==", m_img_quilts[0].shape)
    # m_img_recons = [q * w for q, w in zip(m_img_quilts, wquilts)]
    # print("m_img_recons[0].shape==", m_img_recons[0].shape)
    # m_img_recon = (m_img_recons[0] + m_img_recons[1] + m_img_recons[2] + m_img_recons[3]).reshape(2000, 4000, 4).astype(np.uint8)
    # print("m_img_recon.shape==", m_img_recon.shape)

    reg = cv2.remap(master_img, summed[0], summed[1], interpolation=cv2.INTER_LINEAR)    # summed 是坐标映射关系
    cv2.imwrite("target.jpg", target_img)
    cv2.imwrite("master.jpg", master_img)
    cv2.imwrite("reg.jpg", reg)



if __name__ == "__main__":
    bilinear_interpolation_of_patch_registration()