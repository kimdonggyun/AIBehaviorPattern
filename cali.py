# video calibration and modifying for precise analysis
import numpy as np
import pandas as pd


def combine_video(txtfilepath_L, txtfilepath_R):
    # combine two data file from both side 
    # based on frame and obj_id
    L = pd.read_csv(txtfilepath_L, sep="\t")
    R = pd.read_csv(txtfilepath_R, sep="\t")

    df = L[["x", "y", "w", "h", "obj_id" ,"frame"]].merge(R[["z", "obj_id", "frame"]], on = ["obj_id", "frame"], how = "left")

    df_name = txtfilepath_L.replace("L.txt", "both.txt")
    df.to_csv(df_name, sep="\t")
    print("saved at %s" %(df_name ,))


"""
def video_rotation(filepath):
    # rotate video to get right align


def cal_img_param():
    # calculate image parameteres



def align_frame():
    # align the frame as two cameras have slightly different height


def obj_size(pixel_area, constant):

    #calculte the real size of object depending on it's distance to lens
    #parameters to consider:
    #pixel_area = calculated pixel area of an object
    #distance = distance between object and lens
    #constant = known constanct value (img_size_A * object_A_distance == img_size_B * object_B_distance)

    
    constant = np.square(pixel_area)*di
    real_size = pixel_area
    return real_size
"""

if __name__ == "__main__":
    combine_video("/Users/dkim/Desktop/basler_camera/recording/sp16_5_L.txt", "/Users/dkim/Desktop/basler_camera/recording/sp16_5_R.txt")