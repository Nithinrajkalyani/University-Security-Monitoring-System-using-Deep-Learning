import splitfolders as split_folders
import os
import numpy as np
from sklearn.model_selection import train_test_split


inputdataset='C:\Mini\Augmented'

output='C:\Mini\SplitDatabase'

split_folders.ratio(inputdataset, output=output,seed=1332,ratio=(0.8,0.1,0.1))