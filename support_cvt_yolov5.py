# supporting script for cvt and YoloV5

# import packages
import cv2
import glob
import os
from pathlib import Path

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


if __name__ == "__main__":
    filepath = input("type full filepath and *.img:")
    savename = input("type video name:")
    fps = int(input("type fps:"))
    img_2_video(filepath, savename, fps)