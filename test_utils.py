import sys
import os
import shutil
import plotly
import plotly.express as px
from   plotly.colors import n_colors
from   plotly.subplots import make_subplots
import math
import h5py
import socket
import shutil
import subprocess
import pandas as pd
import numpy as np
from   numpy.random import uniform
import matplotlib.pyplot as plt
import threadpoolctl
import scipy
import joblib
import cmake
import sklearn
import seaborn as sns
import tensorflow
import path
import open3d as o3d
from open3d import utility as utility
V3dV =  o3d.utility.Vector3dVector
import torch
import MyCloud_utils
from MyCloud_utils import *
import time
import IPython
from IPython.display import Javascript


def plot_from_file(filename,title):
  results = {}
  with open (filename) as output:
    lines = output.readlines()
    for l in lines:
      if l[:4] in ['eval','mean','accu']:
        key, val = l.split(': ')
        if key in list(results.keys()):
          results[key] = results[key] + [float(val)]
        else:
          results[key] = [float(val)]
  import matplotlib.pyplot as plt
  keys = list(results.keys())
  for k in keys:
    plt.plot(range(len(results[k])),results[k],label=k)
  plt.title(title)
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.show()          
    
    

n_colors = lambda n: [np.round(c,3) for c in sns.color_palette('hls',n)]

def txt_file_to_npy(txt_file):
  with open (txt_file) as f:
    data = f.readlines()
  data  = np.array([[float(val) for val in row.split()] for row in data])
  return data

def cloud_from_text(txt_file):
  data = txt_file_to_npy(txt_file)
  pts   = data[:,:3]
  cls   = data[:,3:6]
  this_pcd = o3d.geometry.PointCloud()
  this_pcd.points = V3dV(pts)
  this_pcd.colors = V3dV(cls)
  return myCloud(pcd=this_pcd)

def cloud_from_data(data):
  pts   = data.iloc[:,:3]
  cls   = data.iloc[:,3:6]
  this_pcd = o3d.geometry.PointCloud()
  this_pcd.points = V3dV(np.asarray(pts))
  this_pcd.colors = V3dV(np.asarray(cls))
  return myCloud(pcd=this_pcd)
 
def this_color(predictions,p,n):
  color_dict = dict(zip(list(set(predictions)),n_colors(n)))
  return color_dict[p]
  
def pcd_colors(predictions):
  n = len(set(predictions))
  color_dict = dict(zip(list(set(predictions)),n_colors(n)))
  ret = [color_dict[p] for p in predictions]
  return ret

def show_seg(pred_txt_file):
  data = txt_file_to_npy(pred_txt_file)
  pts  = data[:,:3]
  pred = data[:,-1]
  color_list = pcd_colors(preds)
  mn,mx,count = 0, len(pts), min(10000,len(pred))
  idxs = list(uniform(mn,mx,count))
  idxs = [int(i) for i in idxs]
  this_pcd = o3d.geometry.PointCloud()
  this_pcd.points = V3dV(pts[idxs,:])
  this_pcd.colors = V3dV([color_list[i] for i in idxs])
  myCloud(pcd=this_pcd).show_pcd()

the_colors = ['red', 'sienna', 'chartreuse', 'green', 
              'blue', 'plum', 'gold', 'black', 'aqua', 'orange','lightgray','yellowgreen','peachpuff']

def color_code_file(pred_txt_file):
  data = txt_file_to_npy(pred_txt_file)
  pts  = data[:,:3]
  mn,mx,count = 0, len(pts), min(100000,len(pts))
  idxs = list(uniform(mn,mx,count))

  data = np.take(data,idxs,0)
  points, predictions = data[:,:3], data[:,-1]

  class_dict = {'0':  'ceiling',
                '1':  'floor',
                '2':  'wall',
                '3':  'beam',
                '4':  'column',
                '5':  'window',
                '6':  'door',
                '7':  'table',
                '8':  'chair',
                '9':  'sofa',
                '10': 'bookcase',
                '11': 'board',
                '12': 'clutter'}
  data = pd.DataFrame(data,
                      columns=['X','Y','Z','r','g','b','p','class']).drop(columns=['r','g','b','p'])
  try:
    num_to_class = lambda num: class_dict[str(int(num))]
    data['class'] = data['class'].apply(num_to_class)
  except:
    pass

  fig = px.scatter_3d(data, x='X', y='Y', z='Z',
                      color='class',color_discrete_sequence = the_colors)
  fig.layout.legend.itemsizing = 'constant'
  fig.frames=frames
  fig.update_traces(marker=dict(size=1.2),
                    selector=dict(mode='markers'))
  fig.layout.updatemenus =[dict(
                                                    type="buttons",
                                                    buttons=[dict(label="Play",
                                                                  method="animate",
                                                                  #args=[None,{"frame": {"duration": 40}}]
                                                                  args=[None,
                                                                        {"frame": {"duration": 5, 'redraw':True},
                                                                        "transition": {"duration":0},
                                                                        "fromcurrent":True,
                                                                        "mode":"immediate"}])])]
  fig.show()



def rotate_z(x, y, z, theta):
    w = x+1j*y
    return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z


x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
frames=[]
for t in np.arange(0, 6.26, 0.01):
    xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
    frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))



def color_code_compare(pred_txt_file):
  data = txt_file_to_npy(pred_txt_file)
  pts  = data[:,:3]
  mn,mx,count = 0, len(pts), min(100000,len(pts))
  idxs = list(uniform(mn,mx,count))

  data = np.take(data,idxs,0)
  points, colors, predictions = data[:,:3], data[:,3:6], data[:,-1]

  class_dict = {'0':  'ceiling',
                '1':  'floor',
                '2':  'wall',
                '3':  'beam',
                '4':  'column',
                '5':  'window',
                '6':  'door',
                '7':  'table',
                '8':  'chair',
                '9':  'sofa',
                '10': 'bookcase',
                '11': 'board',
                '12': 'clutter'}
  data = pd.DataFrame(data, columns=['X','Y','Z','r','g','b','p','class'])
  data['color_formatted'] = ['rgb(' + ', '.join([str(data['r'][i]), str(data['g'][i]), str(data['b'][i])])  + ')' for i in range(len(data))]
  data = data.drop(columns=['p'])
  try:
    num_to_class = lambda num: class_dict[str(int(num))]
    data['class'] = data['class'].apply(num_to_class)
  except:
    pass

  fig1 = px.scatter_3d(data, x='X', y='Y', z='Z', color='class', color_discrete_sequence = the_colors)
  fig1.layout.legend.itemsizing='constant'
  fig1.layout.legend.itemwidth=31
  fig1.update_traces(marker   = dict(size=0.9),
                     selector = dict(mode='markers'))

  fig = make_subplots(rows=1, cols=2,specs=[[{'type':'scatter3d'}]*2],horizontal_spacing = 0.01)
  for i in range(len(fig1['data'])):
   fig.add_trace(fig1['data'][i],row=1,col=2)

  datasets = []
  for class_ in set(data['class']):
    idxs = [i for i in range(len(data)) if data['class'][i] == class_]
    datasets.append(data.iloc[idxs,:].reset_index(drop=True).drop(columns=['color_formatted']))
  for d in datasets:
    c = cloud_from_data(d)
    this_fig = get_plotly_fig(c)
    this_fig.update_traces(marker   = dict(size=1.2),
                           selector = dict(mode='markers'))
    for i in range(len(this_fig['data'])):
      fig.add_trace(this_fig['data'][i],row=1,col=1)
  
  classes = list(set(data['class']))
  for idx, trace in enumerate(fig["data"]):
    if idx >= len(classes):
      idx = idx - len(classes)
      trace["name"] = classes[idx]

  fig.layout.legend.itemsizing='constant'
  fig.layout.legend.itemwidth=31
  fig.show()



def segment_original_colors(pred_txt_file):
  data = txt_file_to_npy(pred_txt_file)
  pts  = data[:,:3]
  mn,mx,count = 0, len(pts), min(100000,len(pts))
  idxs = list(uniform(mn,mx,count))

  data = np.take(data,idxs,0)
  points, colors, predictions = data[:,:3], data[:,3:6], data[:,-1]

  class_dict = {'0':  'ceiling',
                '1':  'floor',
                '2':  'wall',
                '3':  'beam',
                '4':  'column',
                '5':  'window',
                '6':  'door',
                '7':  'table',
                '8':  'chair',
                '9':  'sofa',
                '10': 'bookcase',
                '11': 'board',
                '12': 'clutter'}
  data = pd.DataFrame(data, columns=['X','Y','Z','r','g','b','p','class'])
  data['color_formatted'] = ['rgb(' + ', '.join([str(data['r'][i]), str(data['g'][i]), str(data['b'][i])])  + ')' for i in range(len(data))]
  data = data.drop(columns=['p'])
  try:
    num_to_class = lambda num: class_dict[str(int(num))]
    data['class'] = data['class'].apply(num_to_class)
  except:
    pass

  fig = go.Figure()

  datasets = []
  for class_ in set(data['class']):
    idxs = [i for i in range(len(data)) if data['class'][i] == class_]
    datasets.append(data.iloc[idxs,:].reset_index(drop=True).drop(columns=['color_formatted']))
  for d in datasets:
    c = cloud_from_data(d)
    this_fig = get_plotly_fig(c)
    this_fig.update_traces(marker   = dict(size=1.2),
                           selector = dict(mode='markers'))
    for i in range(len(this_fig['data'])):
      fig.add_trace(this_fig['data'][i])
  
  classes = list(set(data['class']))
  for idx, trace in enumerate(fig["data"]):
    #if idx >= len(classes):
     # idx = idx - len(classes)
    trace["name"] = classes[idx]

  fig.layout.legend.itemsizing='constant'
  fig.layout.legend.itemwidth=31
  fig.show()



def get_plotly_fig(cloud, mins=None, maxs=None, title=''):
    if mins == None and maxs == None:
        mins,maxs = cloud.mins_maxs(min), cloud.mins_maxs(max)
    #camera = dict(up=xyz_d(0,1,0), center=xyz_d(0,0,0), eye=xyz_d(1.75,1.5,1.75))
    layout = go.Layout(scene=dict(aspectmode='data',
                                  xaxis=dict(title='x',range=[mins['x'],maxs['x']]),
                                  yaxis=dict(title='y',range=[mins['y'],maxs['y']]),
                                  zaxis=dict(title='z',range=[mins['z'],maxs['z']])),
                                  # camera=camera),
                        title=title)
    fig = go.Figure(data=[scatter3d(pcd=cloud.pcd)],layout=layout)
    return fig