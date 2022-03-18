import cv2
import numpy as np


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

#TODO new loss function: Use Normalized Gradient Field measure instead of plain norm?
def m_norm(im1, im2):
    gray1 = ensure_gray(im1)
    gray2 = ensure_gray(im2)
    diff = gray1 - gray2
    m_norm = np.mean(abs(diff))

    return m_norm

def ensure_gray(img):
    """WARNING: img must be in RGB/A or GRAY format, not BGR/A format."""
    if len(img.shape) == 4:
        return cv2.cvtColor(img, code=cv2.COLOR_RGBA2GRAY)
    elif len(img.shape) == 3:
        return cv2.cvtColor(img, code=cv2.COLOR_RGB2GRAY)
    elif len(img.shape) == 2:
        return img
    else:
        print("WARNING: ensure_gray does not recognize the image format.")
        return None

def make_test_img(shape=(250, 250)):
    """Generates a test image with colors arranged so that any transformation is recognizable."""
    stripe = int(shape[0]/10)
    bar = int(shape[1]/10)
    img = np.full((*shape, 3), 0, np.uint8)

    # img[:,100:150,0] = 255
    img[:, int((shape[1]-bar)/2):int((shape[1]+bar)/2), 0] = 255
    # img[:,:20,1] = 255
    img[:,:bar,1] = 255
    img[:,-bar:,2] = 255
    img[:stripe,:,2] = 255
    img[-stripe:,:,1] = 255

    return img
