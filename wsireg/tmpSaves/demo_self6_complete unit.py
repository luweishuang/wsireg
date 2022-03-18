import cv2
import numpy as np
import bilinear
import patchreg
from skimage.util import view_as_windows


def bilinear_interpolation_of_patch_registration(master_srcdata, target_srcdata):
    print("Beginning bilinear_interpolation_of_patch_registration...")
    w_shape = (1000, 1000, 4)    # window_size
    w_step = (500, 500, 4)       # 步长
    padding = w_step[0]          # must do step padding
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

    # Stage Three: Compute patch-level DVFs=dense displacement vector field
    id_patches = patchreg.calc_id_patches(img_shape=master_aligned.shape, patch_size=1000)  # (3,7,3,2000,2000,1)

    map_morphs = np.append(morphs, morphs[:, :, 1, None], axis=2)  # (3,7,3,3,3)
    reg_patches_src = patchreg.applyMorphs(id_patches, map_morphs)   # (3,7,3,2000,2000,1)

    map_patches = reg_patches_src[:, :, 1:, 500:1500, 500:1500, :]

    # Stage Four: Merge patch-level DVFs into a single global transform.
    quilts = bilinear.quilter(map_patches)
    wquilts = bilinear.bilinear_wquilts(map_patches)
    qmaps = [q * w for q, w in zip(quilts, wquilts)]   # 对应位置的元素相乘
    qmaps_sum = qmaps[0] + qmaps[1] + qmaps[2] + qmaps[3]
    summed = (qmaps_sum).reshape(qmaps_sum.shape[:-1]).astype(np.float32)

    master_remap = cv2.remap(master_img, summed[0], summed[1], interpolation=cv2.INTER_LINEAR)    # summed 是坐标映射关系
    master_reg = master_remap[padding:height-padding, padding:width-padding, :]
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


def pad_imgs(master3, target3):
    master_h, master_w, _ = master3.shape
    target_h, target_w, _ = target3.shape
    assert master_h == target_h and master_w == target_w

    src_w = master_w
    src_h = master_h
    mid_h = int(max(2000, np.ceil(src_h/1000)*1000))
    mid_w = int(max(2000, np.ceil(src_w/1000)*1000))
    assert mid_w >= src_w and mid_h >= src_h
    left_pad = int((mid_w-src_w)/2)
    right_pad = int(mid_w - src_w - left_pad)
    top_pad = int((mid_h - src_h) / 2)
    down_pad = int(mid_h - src_h - top_pad)
    master3_pad = cv2.copyMakeBorder(master3, top_pad, down_pad, left_pad, right_pad, cv2.BORDER_REFLECT)
    target3_pad = cv2.copyMakeBorder(target3, top_pad, down_pad, left_pad, right_pad, cv2.BORDER_REFLECT)
    return master3_pad, target3_pad, top_pad, down_pad, left_pad, right_pad


MAX_FEATURES = 5000
GOOD_MATCH_PERCENT = 0.45
def alignImages_Perspective(img1, img2):
    im1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = list(matcher.match(descriptors1, descriptors2, None))

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    height, width, channels = img2.shape
    # Perspective
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    im1Reg_Perspective = cv2.warpPerspective(img1, h, (width, height))   # 透视变换
    return im1Reg_Perspective


def process_single_imgpart(img_master, target_img):
    master_height, master_width, _ = img_master.shape
    cur_height, cur_width, _ = target_img.shape
    assert cur_width == master_width
    top_pad, down_pad = 0, 0
    target_imgpad = target_img.copy()
    if master_height > cur_height:
        top_pad = int((master_height - cur_height)/2)
        down_pad = master_height - cur_height - top_pad
        target_imgpad = cv2.copyMakeBorder(target_img, top_pad, down_pad, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    elif master_height < cur_height:
        print("cur_height > master_height", cur_height, master_height)
    img_show = target_imgpad.copy()
    im2Gray = cv2.cvtColor(target_imgpad, cv2.COLOR_BGR2GRAY)

    im1Reg_Perspective = alignImages_Perspective(img_master, target_imgpad)

    imRegGray = cv2.cvtColor(im1Reg_Perspective, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(imRegGray, im2Gray)
    # cv2.imwrite("diff.jpg", diff)
    ret, thresh = cv2.threshold(diff, 120, 255, cv2.THRESH_BINARY)   # 120
    # cv2.imwrite("thresh.jpg", thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1 and area < max(cur_height, cur_width):
            cv2.drawContours(img_show, cnt, -1, (0, 0, 255), 2)

    img_out = img_show[top_pad: master_height - down_pad, :, : ]
    return img_out


if __name__ == "__main__":
    # draw_img()
    # exit()

    root = "../data/"
    master_srcdata = cv2.imread(root + "OK1_1.jpg")
    target_srcdata = cv2.imread(root + "NG1_1.jpg")
    master3 = master_srcdata[300:4850,:,:]
    # cv2.imwrite("master3.jpg", master3)
    target3 = target_srcdata[720:5270,:,:]
    # cv2.imwrite("target3.jpg", target3)


    # padding to 1000s, at least 2000
    master3_pad, target3_pad, top_pad, down_pad, left_pad, right_pad = pad_imgs(master3, target3)
    # cv2.imwrite("master3_pad.jpg", master3_pad)
    # cv2.imwrite("target3_pad.jpg", target3_pad)

    masterpad_h, masterpad_w, _ = master3_pad.shape
    master_reg_pad = bilinear_interpolation_of_patch_registration(master3_pad, target3_pad)
    master3_reg = master_reg_pad[top_pad: masterpad_h-down_pad, left_pad:masterpad_w-right_pad, : ]
    cv2.imwrite("master3_reg.jpg", master3_reg)
    cv2.imwrite("master3.jpg", master3)
    cv2.imwrite("target3.jpg", target3)

    # Stage Five: high-precision feature alignment
    master_reg_out = process_single_imgpart(master3_reg, target3)
    cv2.imwrite("master_reg_out.jpg", master_reg_out)

    master_out = process_single_imgpart(master3, target3)
    cv2.imwrite("master_out.jpg", master_out)