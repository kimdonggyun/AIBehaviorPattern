# supporting script for cvt and YoloV5

# import packages
from genericpath import getsize
import cv2
import glob
import os
import re
from pathlib import Path
import pandas as pd

def img_2_video (filepath, video_name, fps):
    """
    convert image to video clip
    """
    img_array = []
    
    for filename in sorted(glob.glob(filepath)):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    
    save_path = os.path.join( str(Path(filepath).parents[0] ) , video_name)
    print(save_path)
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    
    return out

def remove_empty_frame(dirpath):
    """
    remove images and labels which doesn't have any label (i.e., 0 byte of lable file)
    """

    # get zero size label files
    labels = glob.glob(os.path.join(dirpath, "*.txt"))
    zero_labels = []
    for l in labels:
        if os.path.getsize(l) == 0:
            zero_labels.append(l)
        else:
            pass
    print("%s of %s files will be deleted" %(len(labels), len(zero_labels)) )
    
    # delete lables having zero size and also corresponding image files
    for z in zero_labels:
        os.remove(z) # remove lables
        os.remove(z.replace(".txt", ".PNG")) # remove images

def yolo_2_dataframe(filepath):
    """
    convert txt file from yolo prediction and convert it to dataframe
    """
    # read in dataframe
    df = pd.read_csv(filepath, sep="\),|\)]," , header=None, engine="python")
    df.columns = ["frame_name", "X1", "Y1", "X2", "Y2", "label"]

    # modify data to get a form of easier use
    df["frame_name"] = df["frame_name"].apply(lambda x: x.replace("[PosixPath(", ""))

    df["X1"] = df["X1"].apply(lambda x: float(re.findall(r"\d+" , x)[0]) )
    df["Y1"] = df["Y1"].apply(lambda x: float(re.findall(r"\d+" , x)[0]) )
    df["X2"] = df["X2"].apply(lambda x: float(re.findall(r"\d+" , x)[0]) )
    df["Y2"] = df["Y2"].apply(lambda x: float(re.findall(r"\d+" , x)[0]) )

    df["probability"] = df["label"].apply(lambda x: float(re.findall(r"\d+.\d+" , x)[0]) ) # first create probability
    df["label"] = df["label"].apply(lambda x: re.findall(r"\w+" , x)[0]) # and then label
    
    # save dataframe
    df.to_csv(filepath.replace(".txt", "_df.txt"), sep="\t")
    print("file is saved at %s" %(filepath.replace(".txt", "df.txt") ,))



if __name__ == "__main__":
    run = int(input("which function?(0: img_2_video, 1: remove_empty_frame):, 2: yolo_2_dataframe :"))
    if run == 0:
        filepath = input("type full filepath and *.container_name: ")
        savename = input("type video name (including video container): ")
        fps = int(input("type fps: "))
        img_2_video(filepath, savename, fps)
    elif run == 1:
        dirpath = input("type full dirpath: ")
        remove_empty_frame(dirpath)
    elif run == 2:
        filepath = input("type full filepath: ")
        yolo_2_dataframe(filepath)

