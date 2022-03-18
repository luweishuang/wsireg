import cv2
import os
import numpy as np
import bilinear
import patchreg
from skimage.util import view_as_windows


def get_y(binary):
    '''
    Horizontal projection, to find text area's horizontal begin position y_min and horizontal end position y_max
    '''
    rows, cols = binary.shape
    hor_list = [0] * rows
    for i in range(rows):
        for j in range(cols):
            if binary.item(i, j) == 0:
                hor_list[i] = hor_list[i] + 1
    '''
    filter hor_list to remove noise points
    '''
    hor_arr = np.array(hor_list)
    hor_arr[np.where(hor_arr < 5)] = 0
    use_list = list(np.where(hor_arr > 0)[0])
    y_min = use_list[0]
    y_max = use_list[-1]
    return y_min, y_max


def get_x(binary):
    '''
    vertical projection, to find text area's vertical begin position x_min and vertical end position x_max
    '''
    rows, cols = binary.shape
    ver_list = [0] * cols
    for i in range(rows):
        for j in range(cols):
            if binary.item(i, j) == 0:
                ver_list[j] = ver_list[j] + 1
    '''
    filter ver_list to remove noise points点
    '''
    ver_arr = np.array(ver_list)
    ver_arr[np.where(ver_arr < 5)] = 0
    use_list = list(np.where(ver_arr > 0)[0])
    x_min = use_list[0]
    x_max = use_list[-1]
    return x_min, x_max


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
    # tt = morphs[:, :, 1, None]
    # print("morphs.min(), morphs.max()==", tt.min(), tt.max())

    # Stage Three: Compute patch-level DVFs=dense displacement vector field
    id_patches = patchreg.calc_id_patches(img_shape=master_aligned.shape, patch_size=1000)  # (3,7,3,2000,2000,1)

    map_morphs = np.append(morphs, morphs[:, :, 1, None], axis=2)  # (3,7,3,3,3)
    reg_patches_src = patchreg.applyMorphs(id_patches, map_morphs)   # (3,7,3,2000,2000,1)

    map_patches = reg_patches_src[:, :, 1:, 500:1500, 500:1500, :]

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


def printer23_run():
    root = "../data/"
    master_srcdata = cv2.imread(root + "OK1_1.jpg")
    target_srcdata = cv2.imread(root + "NG1_1.jpg")
    master3 = master_srcdata[4050:4850, :, :]
    # cv2.imwrite("master3.jpg", master3)
    target3 = target_srcdata[4470:5270, :, :]
    # cv2.imwrite("target3.jpg", target3)

    # padding to 1000s, at least 2000
    master3_pad, target3_pad, top_pad, down_pad, left_pad, right_pad = pad_imgs(master3, target3)
    # cv2.imwrite("master3_pad.jpg", master3_pad)
    # cv2.imwrite("target3_pad.jpg", target3_pad)

    # master3_grid = draw_grid_img(master3_pad, 1000)
    # cv2.imwrite("master3_grid.jpg", master3_grid)

    masterpad_h, masterpad_w, _ = master3_pad.shape
    master_reg_pad = bilinear_interpolation_of_patch_registration(master3_pad, target3_pad)
    master3_reg = master_reg_pad[top_pad: masterpad_h - down_pad, left_pad:masterpad_w - right_pad, :]
    cv2.imwrite("master3_reg.jpg", master3_reg)
    cv2.imwrite("master3.jpg", master3)
    cv2.imwrite("target3.jpg", target3)

    # Stage Five: high-precision feature alignment
    master_reg_out = process_single_imgpart(master3_reg, target3)
    cv2.imwrite("master_reg_out.jpg", master_reg_out)

    master_out = process_single_imgpart(master3, target3)
    cv2.imwrite("master_out.jpg", master_out)



def pad_img_constant(master_textArea, target_textArea):
    m_h, m_w, _ = master_textArea.shape
    t_h, t_w, _ = target_textArea.shape
    assert m_h > t_h and m_w > t_w
    ratio = min(t_h/m_h, t_w/m_w)
    master_new = cv2.resize(master_textArea, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    src_h, src_w, _ = master_new.shape
    mid_h = t_h
    mid_w = t_w
    assert mid_w >= src_w and mid_h >= src_h
    left_pad = int((mid_w - src_w) / 2)
    right_pad = int(mid_w - src_w - left_pad)
    top_pad = int((mid_h - src_h) / 2)
    down_pad = int(mid_h - src_h - top_pad)
    master_pad = cv2.copyMakeBorder(master_new, top_pad, down_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    # master_pad = cv2.copyMakeBorder(master_new, top_pad, down_pad, left_pad, right_pad, cv2.BORDER_REFLECT)
    return master_pad, (top_pad, down_pad, left_pad, right_pad)


def printCheck_run():
    root = "../data/printCheck"
    master_srcdata = cv2.imread(os.path.join(root, "master_color.jpg"))
    target_srcdata = cv2.imread(os.path.join(root, "target.jpg"))

    master_textArea, m_rect = get_text_img(master_srcdata)
    target_textArea, t_rect = get_text_img(target_srcdata)
    # cv2.imwrite("master_textArea.jpg", master_textArea)
    # cv2.imwrite("target_textArea.jpg", target_textArea)

    master_new, mPad_array = pad_img_constant(master_textArea, target_textArea)
    # cv2.imwrite("master_pad.jpg", master_pad)
    # cv2.imwrite("target.jpg", target_textArea)

    masterpad_h, masterpad_w, _ = master_new.shape
    targetpad_h, targetpad_w, _ = target_textArea.shape
    assert masterpad_h == targetpad_h and masterpad_w == targetpad_w

    master_pad, target_pad, top_pad, down_pad, left_pad, right_pad = pad_imgs(master_new, target_textArea)
    # cv2.imwrite("master3_pad.jpg", master3_pad)
    # cv2.imwrite("target3_pad.jpg", target3_pad)

    # master3_grid = draw_grid_img(master3_pad, 1000)
    # cv2.imwrite("master3_grid.jpg", master3_grid)

    masterpad_h, masterpad_w, _ = master_pad.shape
    master_reg_pad = bilinear_interpolation_of_patch_registration(master_pad, target_pad)
    master_reg = master_reg_pad[top_pad: masterpad_h - down_pad, left_pad:masterpad_w - right_pad, :]
    master_reg1 = cv2.cvtColor(master_reg, code=cv2.COLOR_BGRA2RGBA)
    cv2.imwrite("master_reg.jpg", master_reg1)
    cv2.imwrite("master.jpg", master_new)
    cv2.imwrite("target.jpg", target_textArea)

    # Stage Five: high-precision feature alignment
    master_reg_out = process_single_imgpart(master_reg, target_textArea)
    cv2.imwrite("master_reg_out.jpg", master_reg_out)

    master_out = process_single_imgpart(master_new, target_textArea)
    cv2.imwrite("master_out.jpg", master_out)


if __name__ == "__main__":
    # printer23_run()
    printCheck_run()