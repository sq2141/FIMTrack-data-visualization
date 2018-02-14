import numpy as np
import Tkinter as tk
import tkFileDialog
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os

#Define variables
fps = 10

#Plotting label size and rotation
xlabelsize = 6
rot = 45

#Define functions & secondary variables
def numRemover(feat):
    if isinstance(feat, basestring)==1: #Temp fix to bypass nan in edited .csv files
        trimmed = feat.split('(')
        return trimmed[0]

#Ask user to select excel files. Store file paths in fileList
root = tk.Tk()
root.withdraw()
filePathList = tkFileDialog.askopenfilenames(parent=root, title='Choose .csv files')

#Declare lists to store info data about each file. range is the number of variables
numLarv, col, fileNameExt, fileName, x, y, v, v_avg, dst, org, bend, go, go_avg, left, left_avg, right, right_avg, coil, coil_avg = ([0]*len(filePathList) for i in range(19))

#Get directory of files
dir = os.path.dirname(filePathList[0])

#Create list of file names from the full file paths
for i in range(0,len(filePathList)):
    fileNameExt[i] = os.path.basename(filePathList[i])#Extract file name w/ extension
    fileName[i] = os.path.splitext(fileNameExt[i])[0] #Remove extension
    # print fileName[i]

#Loop each file
for i in range(0,len(filePathList)):
    raw_data = pd.read_csv(dir + '/' + fileNameExt[i])
    col[i] = len(raw_data.columns)
    numLarv[i] = col[i]-1
    data_df = raw_data.iloc[:, :col[i]]
    data_df['Feature'] = data_df.iloc[:,0].apply(numRemover) #Create feature col
    data_df['Avg_group'] = data_df.mean(axis=1) #Create Group mean col

    #Extract features
    x[i] = data_df.loc[data_df['Feature'] == 'mom_x']
    y[i] = data_df.loc[data_df['Feature'] == 'mom_y']
    v[i] = data_df.loc[data_df['Feature'] == 'velocity']
    v_avg[i] = v[i].mean(axis=0)
    v[i]['Frame'] = range(1, len(v[i]) + 1)
    v[i]['Time'] = v[i]['Frame'] / fps
    dst[i] = data_df.loc[data_df['Feature'] == 'acc_dst']
    org[i] = data_df.loc[data_df['Feature'] == 'dst_to_origin']
    bend[i] = data_df.loc[data_df['Feature'] == 'bending']
    go[i] = data_df.loc[data_df['Feature'] == 'go_phase']
    go_avg[i] = go[i].mean(axis=0)
    left[i] = data_df.loc[data_df['Feature'] == 'left_bended']
    left_avg[i] = left[i].mean(axis=0)
    right[i] = data_df.loc[data_df['Feature'] == 'right_bended']
    right_avg[i] = right[i].mean(axis=0)
    coil[i] = data_df.loc[data_df['Feature'] == 'is_coiled']
    coil_avg[i] = coil[i].mean(axis=0)

#Start a pdf to save figures
pp = PdfPages('results.pdf')

#Trajectory Figures
for i in range(0,len(filePathList)):
    plt.figure(figsize=(8, 8))
    plt.plot(x[i].iloc[:, 1:col[i]], y[i].iloc[:, 1:col[i]])
    plt.grid(True)
    plt.ylim(0, 2048)
    plt.xlim(0, 2048)
    plt.title(fileName[i] + ' Trajectory')
    plt.xlabel('Pixels')
    plt.ylabel('Pixels')
    pp.savefig()

#Velocity over time
plt.figure()
for i in range(0,len(filePathList)):
    plt.plot(v[i]['Time'] , v[i]['Avg_group'])
plt.title('Velocity over time')
plt.ylabel('Pixels per frame')
plt.xlabel('Seconds')
plt.legend(fileName)
pp.savefig()

#Avg veloctiy
plt.figure()
for i in range(0,len(filePathList)):
    v_avg[i].drop('Avg_group',inplace=True)
v_avgs_joined = pd.concat(v_avg, axis=1)
v_avgs_joined[:max(numLarv)].boxplot(return_type='dict')
plt.title('Velocity')
plt.ylabel('Pixels per frame')
plt.xticks(np.arange(1,len(filePathList)+1),fileName)
locs, labels = plt.xticks()
plt.setp(labels, rotation=rot)
plt.tick_params(axis='x', labelsize=xlabelsize)
pp.savefig()

#Accumulated distance
plt.figure()
for i in range(0,len(filePathList)):
    plt.plot(v[i]['Time'], dst[i]['Avg_group'])
plt.title('Accumulated distance')
plt.ylabel('Pixels')
plt.xlabel('Seconds')
plt.legend(fileName, loc='upper left')
pp.savefig()

#Distance from Origin figure
plt.figure()
for i in range(0,len(filePathList)):
    plt.plot(v[i]['Time'], org[i]['Avg_group'])
plt.title('Distance from origin')
plt.ylabel('Pixels')
plt.xlabel('Seconds')
plt.legend(fileName, loc='upper left')
pp.savefig()

#Bending angle
plt.figure()
for i in range(0,len(filePathList)):
    bend[i]['Avg_group'].dropna(inplace=True)
    print bend[i]
    sns.distplot(bend[i]['Avg_group'])
plt.title('Body bend angle')
plt.ylabel('Frequency')
plt.xlabel('Degrees')
plt.legend(fileName)
pp.savefig()

#Go frequency
plt.figure()
for i in range(0,len(filePathList)):
    go_avg[i].drop('Avg_group', inplace=True)
go_avgs_joined = pd.concat(go_avg, axis=1)
go_avgs_joined[:max(numLarv)].boxplot(return_type='dict')
plt.title('Go frequency')
plt.ylabel('Frequency')
plt.xticks(np.arange(1,len(filePathList)+1),fileName)
locs, labels = plt.xticks()
plt.setp(labels, rotation=rot)
plt.tick_params(axis='x', labelsize=xlabelsize)
pp.savefig()

#Left bended
plt.figure()
for i in range(0,len(filePathList)):
    left_avg[i].drop('Avg_group', inplace=True)
left_avgs_joined = pd.concat(left_avg, axis=1)
left_avgs_joined[:max(numLarv)].boxplot(return_type='dict')
plt.title('Left bend frequency')
plt.ylabel('Frequency')
plt.xticks(np.arange(1,len(filePathList)+1),fileName)
locs, labels = plt.xticks()
plt.setp(labels, rotation=rot)
plt.tick_params(axis='x', labelsize=xlabelsize)
pp.savefig()

#Right bended
plt.figure()
for i in range(0,len(filePathList)):
    right_avg[i].drop('Avg_group', inplace=True)
right_avgs_joined = pd.concat(right_avg, axis=1)
right_avgs_joined[:max(numLarv)].boxplot(return_type='dict')
plt.title('Right bend frequency')
plt.ylabel('Frequency')
plt.xticks(np.arange(1,len(filePathList)+1),fileName)
locs, labels = plt.xticks()
plt.setp(labels, rotation=rot)
plt.tick_params(axis='x', labelsize=xlabelsize)
pp.savefig()

#Coiling frequency
plt.figure()
for i in range(0,len(filePathList)):
    coil_avg[i].drop('Avg_group', inplace=True)
coil_avgs_joined = pd.concat(coil_avg, axis=1)
coil_avgs_joined[:max(numLarv)].boxplot(return_type='dict')
plt.title(' Coil frequency')
plt.ylabel('Frequency')
plt.xticks(np.arange(1,len(filePathList)+1),fileName)
locs, labels = plt.xticks()
plt.setp(labels, rotation=rot)
plt.tick_params(axis='x', labelsize=xlabelsize)
pp.savefig()

#Finish pdf file and open
pp.close()

#On Windows, use:
os.startfile('results.pdf')

#On Mac, use:
#os.system('open ' + 'results.pdf')
