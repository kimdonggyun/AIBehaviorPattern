# plot for all data
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




def plt_3D(txtfilepath):
    # generate 3D coordination plot
    # further developement is needed
    # 1. reading in dataframe and modifying
    df = pd.read_csv(txtfilepath, sep="\t")
    df = df.drop(labels=0, axis=0) # delete first row

    
    # 2. plotting
    fig = plt.figure()
    ax = plt.axes(projection= "3d")
    ax.set_xlim3d(0, 1000)
    ax.set_ylim3d(1000, 0) # also invert y axis
    ax.set_zlim3d(0, 1000)
    ax.scatter(df["x"], df["z"], df["y"])

    plt.show()


if __name__ == "__main__":
    plt_3D("/Users/dkim/Desktop/basler_camera/recording/sp16_5_both.txt")
