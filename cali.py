# video calibration and modifying for precise analysis
import numpy as np

def video_rotation(filepath):
    # rotate video to get right align


def cal_img_param():
    # calculate image parameteres



def align_frame():
    # align the frame as two cameras have slightly different height


def obj_size(pixel_area, constant):
    """
    calculte the real size of object depending on it's distance to lens
    parameters to consider:
    pixel_area = calculated pixel area of an object
    distance = distance between object and lens
    constant = known constanct value (img_size_A * object_A_distance == img_size_B * object_B_distance)
    """
    
    constant = np.square(pixel_area)*di
    real_size = pixel_area
    return real_size