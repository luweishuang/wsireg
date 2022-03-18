import cv2
import numpy as np
import bilinear
import patchreg
from utils import get_x, get_y
from skimage.util import view_as_windows


def bilinear_interpolation_of_patch_registration(master_srcdata, target_srcdata, wsize):
    print("Beginning bilinear_interpolation_of_patch_registration...")
    w_shape = (wsize, wsize, 3)
    stepsize = int(wsize/2)# window_size
    w_step = (stepsize, stepsize, 3)       # 步长
    padding = stepsize          # must do step padding
    master_img = cv2.copyMakeBorder(master_srcdata, padding, padding, padding, padding, cv2.BORDER_REFLECT)
    target_img = cv2.copyMakeBorder(target_srcdata, padding, padding, padding, padding, cv2.BORDER_REFLECT)

    # Stage One: Low-precision feature alignment
    h, _ =  patchreg.alignFeatures(target_img, master_img)
    height, width = target_img.shape[:2]
    master_aligned = cv2.warpPerspective(master_img, h, (width, height))

    # Stage Two: Calculate patch-level registrations
    stack1 = np.concatenate((target_img, master_aligned), axis=-1)
    patches = view_as_windows(stack1, window_shape=w_shape, step=w_step)
    morphs = patchreg.calcPlateMorphs(patches)   # (3,7,2,3,3)

    # Stage Three: Compute patch-level DVFs=dense displacement vector field
    id_patches = patchreg.calc_id_patches(img_shape=master_aligned.shape, patch_size=wsize)  # (3,7,3,2000,2000,1)

    map_morphs = np.append(morphs, morphs[:, :, 1, None], axis=2)  # (3,7,3,3,3)
    reg_patches_src = patchreg.applyMorphs(id_patches, map_morphs)   # (3,7,3,2000,2000,1)

    map_patches = reg_patches_src[:, :, 1:, stepsize:wsize+stepsize, stepsize:wsize+stepsize, :]

    # Stage Four: Merge patch-level DVFs into a single global transform.
    quilts = bilinear.quilter(map_patches)
    wquilts = bilinear.bilinear_wquilts(map_patches)
    qmaps = [q * w for q, w in zip(quilts, wquilts)]
    qmaps_sum = qmaps[0] + qmaps[1] + qmaps[2] + qmaps[3]
    summed = (qmaps_sum).reshape(qmaps_sum.shape[:-1]).astype(np.float32)

    master_remap = cv2.remap(master_img, summed[0], summed[1], interpolation=cv2.INTER_LINEAR)    # summed 是坐标映射关系
    master_reg = master_remap[padding:height-padding, padding:width-padding, :]
    # cv2.imwrite("target.jpg", target_img[padding:height-padding, padding:width-padding, :])
    # cv2.imwrite("master.jpg", master_img[padding:height-padding, padding:width-padding, :])
    # cv2.imwrite("master_reg.jpg", master_reg)
    return master_reg


def draw_grid_img(srcdata, wsize):
    src_h, src_w, _ = srcdata.shape
    img_show = srcdata.copy()
    for ii in range(1, int(src_h/wsize)):
        cur_y = int(wsize * ii)
        cv2.line(img_show, (0, cur_y), (src_w, cur_y), (0, 255, 0), 2)
    for jj in range(1, int(src_w / wsize)):
        cur_x = int(wsize * jj)
        cv2.line(img_show, (cur_x, 0), (cur_x, src_h), (0, 255, 0), 2)
    return img_show


def get_text_img(img_src, ratio = 0.35):
    '''
    Shrink the image appropriately and use a rectangle to frame the text area
    '''
    src_h, src_w, _ = img_src.shape
    img_bgr = cv2.resize(img_src, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)

    img = img_bgr.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t, binary = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)

    y_min, y_max = get_y(binary)
    x_min, x_max = get_x(binary)
    # resize to source image size
    x_min = max(0, int(x_min / ratio)-10)
    y_min = max(0, int(y_min / ratio)-10)
    x_max = min(src_w, int(x_max / ratio)+10)
    y_max = min(src_h, int(y_max / ratio)+10)
    text_area_rect = (x_min, y_min, x_max, y_max)
    text_img = img_src[y_min : y_max, x_min:x_max, :]
    return text_img, text_area_rect


def single_img_pad(img_in, wsize):
    src_h, src_w, _ = img_in.shape
    mid_h = int(max(wsize * 2, np.ceil(src_h / wsize) * wsize))
    mid_w = int(max(wsize * 2, np.ceil(src_w / wsize) * wsize))
    assert mid_w >= src_w and mid_h >= src_h
    left_pad = int((mid_w - src_w) / 2)
    right_pad = int(mid_w - src_w - left_pad)
    top_pad = int((mid_h - src_h) / 2)
    down_pad = int(mid_h - src_h - top_pad)
    img_in_pad = cv2.copyMakeBorder(img_in, top_pad, down_pad, left_pad, right_pad, cv2.BORDER_REFLECT)
    return img_in_pad, top_pad, down_pad, left_pad, right_pad


def pad_imgs(master3_textArea, target3_textArea, wsize):
    master3_pad, m3_topPad, m3_downPad, m3_leftPad, m3_rightPad = single_img_pad(master3_textArea, wsize)
    target3_pad, t3_topPad, t3_downPad, t3_leftPad, t3_rightPad = single_img_pad(target3_textArea, wsize)
    return master3_pad, target3_pad, (m3_topPad, m3_downPad, m3_leftPad, m3_rightPad), (t3_downPad, t3_leftPad, t3_rightPad)


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
    root = "../data/"
    master_srcdata = cv2.imread(root + "OK1_1.jpg")
    target_srcdata = cv2.imread(root + "NG1_1.jpg")
    master3 = master_srcdata[4050:4850,:,:]
    target3 = target_srcdata[4470:5270,:,:]
    # cv2.imwrite("master3.jpg", master3)
    # cv2.imwrite("target3.jpg", target3)

    master3_textArea, m3_rect = get_text_img(master3)
    target3_textArea, t3_rect  = get_text_img(target3)
    # cv2.imwrite("master3_textArea.jpg", master3_textArea)
    # cv2.imwrite("target3_textArea.jpg", target3_textArea)

    # padding
    wsize = 500
    master3_pad, target3_pad, m3Pad_ar, t3Pad_ar = pad_imgs(master3_textArea, target3_textArea, wsize)
    # cv2.imwrite("master3_pad.jpg", master3_pad)
    # cv2.imwrite("target3_pad.jpg", target3_pad)

    master3_grid = draw_grid_img(master3_pad, wsize)
    cv2.imwrite("master3_grid.jpg", master3_grid)

    masterpad_h, masterpad_w, _ = master3_pad.shape
    targetpad_h, targetpad_w, _ = target3_pad.shape
    assert masterpad_h == targetpad_h and masterpad_w == targetpad_w
    master_reg_pad = bilinear_interpolation_of_patch_registration(master3_pad, target3_pad, wsize)
    top_pad, down_pad, left_pad, right_pad = m3Pad_ar
    master3_reg = master_reg_pad[top_pad: masterpad_h-down_pad, left_pad:masterpad_w-right_pad, : ]
    cv2.imwrite("master3_reg.jpg", master3_reg)
    # cv2.imwrite("master3.jpg", master3)
    # cv2.imwrite("target3.jpg", target3)
    exit()
    # Stage Five: high-precision feature alignment
    master_reg_out = process_single_imgpart(master3_reg, target3)
    cv2.imwrite("master_reg_out.jpg", master_reg_out)

    master_out = process_single_imgpart(master3, target3)
    cv2.imwrite("master_out.jpg", master_out)
