import sys
import os
current_dir = os.getcwd()
#PKG_DIR      = '/content/drive/MyDrive/Point cloud files/PKG_DIR'
#os.chdir(PKG_DIR)
#sys.path.append(PKG_DIR)
#sys.path = list(set(sys.path))

import plotly.express as px
import itertools
from   itertools import combinations
import math as m
import matplotlib
import matplotlib.pyplot as plt
from   matplotlib.ticker import FormatStrFormatter
import numpy as np
from   numpy.random import uniform
from   numpy import concatenate
import open3d as o3d
from   open3d import utility as utility
### Desktop:
#from   open3d.cpu.pybind.utility import Vector3dVector as V3dV
### Colab:
#import   utility.Vector3dVector as 
V3dV = o3d.utility.Vector3dVector
import os
from   path import Path
import pandas as pd
import plotly.graph_objects as go
from   plotly.subplots import make_subplots
import pyransac3d as pyrsc
import random
import sys
import torch
from   torch.utils.data import Dataset, DataLoader
import tqdm
from   tqdm import tqdm

os.chdir(current_dir)
random.seed = 42
the_colors = ['red', 'sienna', 'chartreuse', 'green', 
              'blue', 'plum', 'gold', 'black', 'aqua', 'orange','lightgray','yellowgreen','peachpuff']


points_f      = lambda cloud: np.asarray(cloud.pcd.points)
colors_f      = lambda cloud: np.asarray(cloud.pcd.colors)
clean_cloud_f = lambda cloud: myCloud(pcd=cloud.pcd.remove_statistical_outlier(nb_neighbors=10,std_ratio=1.5)[0])
sub_pcd_f     = lambda pcd,indices,invert=False: pcd.select_by_index(indices,invert=invert)


def add_listlike(L1,L2,type_):
    L1 = [] if str(type(L1)) == "<class 'NoneType'>" else L1
    L2 = [] if str(type(L2)) == "<class 'NoneType'>" else L2
    L1 = L1.tolist() if str(type(L1)) == "<class 'numpy.ndarray'>" else list(L1)
    L2 = L2.tolist() if str(type(L2)) == "<class 'numpy.ndarray'>" else list(L2)
    if type_ == "array":
        return(np.asarray(L1+L2))
    else:
        return L1+L2

def side_by_side(c1,c2,t1,t2):
    mins1,mins2,maxs1,maxs2 = c1.mins_maxs(min), c2.mins_maxs(min), c1.mins_maxs(max), c2.mins_maxs(max)
    mins    = {var:min(mins1[var],mins2[var]) for var in ['x','y','z']}
    maxs    = {var:max(maxs1[var],maxs2[var]) for var in ['x','y','z']}
    x_range = maxs['x']-mins['x']
    y_prop  = (maxs['y']-mins['y'])/x_range
    z_prop  = (maxs['z']-mins['z'])/x_range
    fig     = make_subplots(rows=1,cols=2,specs=[[{'type':'scatter3d'}]*2],subplot_titles=(t1,t2))
    col_    =1
    for c in [c1,c2]:
        c_points = points_f(c)
        if c_points[0].size != 0:
            c_data = [scatter3d(c.pcd)]
            for i in range(len(c_data)):
                fig.add_trace(c_data[i],row=1,col=col_)
        col_+=1
    camera = dict(up=xyz_d(0,1,0), center=xyz_d(0,0,0), eye=xyz_d(1.75,1.5,1.75))
    fig.update_scenes(aspectratio=dict(x=1,y=y_prop,z=1),
                      xaxis=dict(title='x',range=[mins['x'],maxs['x']]),
                      yaxis=dict(title='y',range=[mins['y'],maxs['y']]),
                      zaxis=dict(title='z',range=[mins['z'],maxs['z']]),
                      camera=camera)
    fig.show()    

def xyz_d(x_input,y_input,z_input):
    return {'x':x_input,'y':y_input,'z':z_input}

def concat_data(cloud):
    marker_ = {'size':2}
    mode_   = 'markers'
    xs, ys, zs = [],[],[]
    for d in cloud.data:
        xs.append(d.__getitem__(0).x)
        ys.append(d.__getitem__(0).y)
        zs.append(d.__getitem__(0).z)
    xs = concatenate(xs)
    ys = concatenate(ys)
    zs = concatenate(zs)
    return [go.Scatter3d(marker=marker_,mode=mode_,x=xs,y=ys,z=zs)]

def data_to_points(data):
    x,y,z = xyz(data=data)[0].tolist(),xyz(data=data)[1].tolist(),xyz(data=data)[2].tolist()
    return np.asarray(list(zip(x,y,z)))

def xyz(pcdObject=None,myCloudObject=None,data=[],points_array=np.array([])):
    if pcdObject != None:
        return (np.asarray(pcdObject.points)).T
    if myCloudObject != None:
        return (points_f(myCloudObject)).T
    elif len(data) != 0:
        data = concat_data(data)
        return data.__getitem__(0).x, data.__getitem__(0).y, data.__getitem__(0).z
    elif points_array.size != 0:
        return points_array.T
    else:
        return np.array([[],[],[]])
        
def scatter3d(pcd=None,points_array=None):
    x,y,z = [],[],[]
    if pcd != None:
        if str(type(pcd)) == "<class 'open3d.geometry.PointCloud'>":
            x,y,z = xyz(pcdObject=pcd)
        elif str(type(pcd)) == "<class 'open3d.cpu.pybind.geometry.PointCloud'>":
            x,y,z = xyz(pcdObject=pcd)
        elif str(type(pcd)) == "<class 'open3d.cuda.pybind.geometry.PointCloud'>":
            x,y,z = xyz(pcdObject=pcd)
        else:
            x,y,z = xyz(myCloudObject=pcd)
    elif len(points_array) != 0:
#    elif points_array != None:
        x,y,z = points_array.T
    if pcd != None:
        return go.Scatter3d(x=x,y=y,z=z,mode='markers',marker={'size':2,'color':np.asarray(pcd.colors)})
    else:
        return go.Scatter3d(x=x,y=y,z=z,mode='markers',marker={'size':2})

    
class myCloud:
    def __init__(self, filepath='', pcd=o3d.geometry.PointCloud()):

        self.filepath = filepath
        self.pcd      = pcd
        
        if len(self.filepath) > 0:
            self.pcd = o3d.io.read_point_cloud(self.filepath,'auto')

    def rotate(self, axes, theta):
        Rx = np.matrix([[ 1, 0           , 0           ],
                        [ 0, m.cos(theta),-m.sin(theta)],
                        [ 0, m.sin(theta), m.cos(theta)]])
        Ry = np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                        [ 0           , 1, 0           ],
                        [-m.sin(theta), 0, m.cos(theta)]])
        Rz = np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                        [ m.sin(theta), m.cos(theta) , 0 ],
                        [ 0           , 0            , 1 ]])
        rot_dict = {'x':Rx,'y':Ry,'z':Rz}
        
        points = np.asarray(self.pcd.points)
        for ax in axes:
            coordinate_matrix = np.matmul(points,rot_dict[ax])
        
        rotated_pcd        = o3d.geometry.PointCloud()
        rotated_pcd.points = V3dV(coordinate_matrix)
        rotated_pcd.colors = V3dV(self.pcd.colors)
        return myCloud(pcd=rotated_pcd)
    
    def mins_maxs(self,f=min):
        xs,ys,zs = xyz(points_array=np.asarray(self.pcd.points))
        try:
            ret_x, ret_y, ret_z = f(xs), f(ys), f(zs)
        except ValueError as e:
            [ret_x,ret_y,ret_z] = [100,100,100] if f==min else [-100,-100,-100]

        mins_or_maxs = dict(x=ret_x, y=ret_y, z=ret_z)
        return mins_or_maxs

    def downsample(self,f,value):
        if f == 'voxel':
            return myCloud(pcd=self.pcd.voxel_down_sample(voxel_size=value))
        elif f == 'uniform':
            return myCloud(pcd=self.pcd.uniform_down_sample(every_k_points=value))
        else:
            return myCloud(pcd=self.pcd)
    def show_pcd(self, mins=None, maxs=None, title=''):
        if mins == None and maxs == None:
            mins,maxs = self.mins_maxs(min), self.mins_maxs(max)
        camera = dict(up=xyz_d(0,1,0), center=xyz_d(0,0,0), eye=xyz_d(1.75,1.5,1.75))
        layout = go.Layout(scene=dict(aspectmode='data',
                                      xaxis=dict(title='x',range=[mins['x'],maxs['x']]),
                                      yaxis=dict(title='y',range=[mins['y'],maxs['y']]),
                                      zaxis=dict(title='z',range=[mins['z'],maxs['z']]),
                                      camera=camera),
                           title=title)
        fig = go.Figure(data=[scatter3d(pcd=self.pcd)],layout=layout)
        fig.show()
        
def remove_planes_weird(in_cloud, prev_out_cloud, n_loops, thresh, show_in=True,show_out=True):
    plane_points,plane_colors = [],[]
    if prev_out_cloud != None:
        plane_points, plane_colors = points_f(prev_out_cloud),colors_f(prev_out_cloud)
    in_points, in_length = points_f(in_cloud),len(points_f(in_cloud))
    leftover=in_cloud
    for i in range(int(n_loops)):
        colormap      = list(plt.get_cmap("tab20")(i))
        _, plane_idxs = pyrsc.Plane().fit(points_f(leftover),thresh)
        plane_pcd     = sub_pcd_f(leftover.pcd, plane_idxs).paint_uniform_color(colormap[:3])
        add_to_plane  = myCloud(pcd = plane_pcd)
        leftover      = myCloud(pcd = sub_pcd_f(leftover.pcd,plane_idxs,invert=True))
        plane_points  = points_f(add_to_plane)
        plane_colors  = points_f(add_to_plane)
        add_to_plane.show_pcd()
    go_pcd        = pcd=o3d.geometry.PointCloud()
    go_pcd.points = V3dV(plane_points)
    go_pcd.colors = V3dV(plane_colors)
    stay = myCloud(pcd=leftover.pcd)
    print('***')
    go_display = myCloud(pcd=go_pcd)
    if   show_in == True and show_out == True:
        side_by_side(stay,go_display,'Stay','Go')
    elif show_in == True and show_out == False:
        stay.show_pcd()
    elif show_in == False and show_out == True:
        go_display.show_pcd(mins=stay.mins_maxs(min), maxs=stay.mins_maxs(max))
    return stay,go_display


def remove_planes(in_cloud, prev_out_cloud, n_loops, thresh, show_in=True,show_out=True,clean=False):
    plane_points,plane_colors = [],[]
    if prev_out_cloud != None:
        plane_points, plane_colors = points_f(prev_out_cloud),colors_f(prev_out_cloud)
    in_points, in_length = points_f(in_cloud),len(points_f(in_cloud))
    leftover=in_cloud
    for i in tqdm(range(int(n_loops))):
        colormap      = list(plt.get_cmap("tab20")(i))
        _, plane_idxs = pyrsc.Plane().fit(points_f(leftover),thresh)
        plane_pcd     = sub_pcd_f(leftover.pcd, plane_idxs).paint_uniform_color(colormap[:3])
        add_to_plane  = myCloud(pcd = plane_pcd)
        leftover      = myCloud(pcd = sub_pcd_f(leftover.pcd,plane_idxs,invert=True))
        plane_points  = add_listlike(plane_points, points_f(add_to_plane),'array')
        plane_colors  = add_listlike(plane_colors, colors_f(add_to_plane),'array')
    
    go_pcd        = o3d.geometry.PointCloud()
    go_pcd.points = V3dV(plane_points)
    go_pcd.colors = V3dV(plane_colors)
    go_ = go_display = myCloud(pcd=go_pcd)
    
    stay = myCloud(pcd=leftover.pcd)
    stay_display = clean_cloud_f(stay) if clean==True else stay
    
    if  show_in == True and show_out == True:
        side_by_side(stay_display,go_display,'Stay','Go')
    elif show_in == True and show_out == False:
        stay_display.show_pcd()
    elif show_in == False and show_out == True:
        go_display.show_pcd(mins=stay_display.mins_maxs(min), maxs=stay_display.mins_maxs(max))
    return stay,go_

def best_hyper_params(cloud, eps_min_max, pts_min_max, iterations,low_percentile,high_percentile):
    [min_e,max_e]           = eps_min_max
    [min_minpts,max_minpts] = pts_min_max
    hyp_params              = {'eps':[],'min_points':[],'iqr_normalized':[]}
    pcd                     = cloud.pcd

    for i in tqdm(range(iterations)):
        eps          = round(random.uniform(min_e,max_e),3)
        min_points   = random.randint(min_minpts,max_minpts)
        labels       = np.array(pcd.cluster_dbscan(eps=eps,min_points=min_points))
        _25th, _75th = np.percentile(labels, [low_percentile, high_percentile])
        range_       = labels.max()-labels.min()
        iqr          = 0 if range_ == 0 else (_75th-_25th)*10/range_

        hyp_params['eps'].append(eps)
        hyp_params['min_points'].append(min_points)
        hyp_params['iqr_normalized'].append(iqr) 

    best_i          = hyp_params['iqr_normalized'].index(max(hyp_params['iqr_normalized']))    
    best_eps        = hyp_params['eps'][best_i]
    best_min_points = hyp_params['min_points'][best_i]
    labels          = np.array(pcd.cluster_dbscan(eps=best_eps,min_points=best_min_points))
    
    plt.hist(labels)
    plt.title('Distribution of Clusters with Optimal Hyperparameters')
    plt.show()
    print('\n\nbest eps: {}, best min_points: {}'.format(best_eps,best_min_points))
    
    return labels, best_eps, best_min_points

def get_indices(list_,value):
    idx = []
    for i in range(len(list_)):
        if list_[i] == value:
            idx.append(i)
    return idx

def color_by_label(in_cloud, prev_out_cloud, labels, display=True):
    max_label       = labels.max()
    set_labels = list(set(labels))
    colors_          = plt.get_cmap("tab20")(set_labels/max_label)
    label_indices    = [get_indices(labels,l) for l in set_labels]
    label_index_dict = dict(zip(set_labels,label_indices))
    leftover=in_cloud
    clusters,cluster_points,cluster_colors=[],[], []
    
    for i in range(len(set_labels)):
        color = colors_[i][:3]
        add_indices = label_index_dict[set_labels[i]]
        this_cluster_pcd = sub_pcd_f(in_cloud.pcd,add_indices).paint_uniform_color(color)
        this_cluster = myCloud(pcd=this_cluster_pcd)
        leftover=myCloud(pcd=sub_pcd_f(in_cloud.pcd,add_indices,invert=True))
        cluster_points = add_listlike(cluster_points,points_f(this_cluster),'array')
        cluster_colors = add_listlike(cluster_colors,colors_f(this_cluster),'array')
        clusters.append(this_cluster)

    clustered_pcd = o3d.geometry.PointCloud()
    clustered_pcd.points=V3dV(cluster_points)
    clustered_pcd.colors=V3dV(cluster_colors)
    clustered_cloud = myCloud(pcd=clustered_pcd)
    
    if display==True:
        clustered_cloud.show_pcd(title='Clustered Objects')
    return clusters, clustered_cloud


def best_hyper_params2(cloud, eps_min_max, pts_min_max, iterations,low_percentile,high_percentile):
    [min_e,max_e]           = eps_min_max
    [min_minpts,max_minpts] = pts_min_max
    hyp_params              = {'eps':[],'min_points':[],'std':[],'num_clusters':[]}
    pcd                     = cloud.pcd

    for i in tqdm(range(iterations)):
        eps          = round(random.uniform(min_e,max_e),3)
        min_points   = random.randint(min_minpts,max_minpts)
        labels       = np.array(pcd.cluster_dbscan(eps=eps,min_points=min_points))
        _25th, _75th = np.percentile(labels, [low_percentile, high_percentile])
        range_       = labels.max()-labels.min()
        #iqr          = 0 if range_ == 0 else (_75th-_25th)*10/range_
        std = np.std([list(labels).count(l) for l in labels])

        hyp_params['eps'].append(eps)
        hyp_params['min_points'].append(min_points)
        hyp_params['std'].append(std) 
        hyp_params['num_clusters'].append(len(labels))

    i_candidates = [i for i in range(len(hyp_params)) if hyp_params['num_clusters'][i]>4 and hyp_params['num_clusters'][i]<10]
    hyp_params['eps'] =  [hyp_params['eps'][i] for i in i_candidates]
    hyp_params['min_points'] =  [hyp_params['min_points'][i] for i in i_candidates]
    hyp_params['std'] =  [hyp_params['std'][i] for i in i_candidates]
    hyp_params['num_clusters'] =  [hyp_params['num_clusters'][i] for i in i_candidates]


    best_i          = hyp_params['std'].index(min(hyp_params['std']))    
    best_eps        = hyp_params['eps'][best_i]
    best_min_points = hyp_params['min_points'][best_i]
    labels          = np.array(pcd.cluster_dbscan(eps=best_eps,min_points=best_min_points))
    
    plt.hist(labels)
    plt.title('Distribution of Clusters with Optimal Hyperparameters')
    plt.show()
    print('\n\nbest eps: {}, best min_points: {}'.format(best_eps,best_min_points))
    
    return labels, best_eps, best_min_points


import glob
import trimesh
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(1234)

def parse_dataset(num_points=2048):

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )


def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label


def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))
    
def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

### Pretty sure this data needs to be in the form 'x y z label'
def visualize_data(point_cloud, labels):
    df = pd.DataFrame(
        data={
            "x": point_cloud[:, 0],
            "y": point_cloud[:, 1],
            "z": point_cloud[:, 2],
            "label": labels,
        }
    )
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection="3d")
    for index, label in enumerate(LABELS):
        c_df = df[df["label"] == label]
        try:
            ax.scatter(
                c_df["x"], c_df["y"], c_df["z"], label=label, alpha=0.5, c=COLORS[index]
            )
        except IndexError:
            pass
    ax.legend()
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

def rotate_z(x, y, z, theta):
    w = x+1j*y
    return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
frames=[]
for t in np.arange(0, 6.26, 0.01):
    xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
    frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))


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



def color_code_setup(pred_txt_file):
  data = txt_file_to_npy(pred_txt_file)
  pts  = data[:,:3]
  mn,mx,count = 0, len(pts), min(100000,len(pts))
  idxs = list(uniform(mn,mx,count))

  data = np.take(data,idxs,0)
  points, colors, predictions = data[:,:3], data[:,3:6], data[:,-1]

  class_vals  = ['0',      '1',    '2',   '3',   '4',     '5',     '6',   '7',    '8',    '9',   '10',      '11',   '12']
  class_names = ['ceiling','floor','wall','beam','column','window','door','table','chair','sofa','bookcase','board','clutter']
  class_dict = dict(zip(class_vals, class_names))

  data = pd.DataFrame(data, columns=['X','Y','Z','r','g','b','p','class'])
  try:
    num_to_class = lambda num: class_dict[str(int(num))]
    data['class'] = data['class'].apply(num_to_class)
  except:
    pass

  return data

def color_code_file(pred_txt_file):
  data = color_code_setup(pred_txt_file)
  fig = px.scatter_3d(data, x='X', y='Y', z='Z', color='class',color_discrete_sequence = the_colors)
  fig.layout.legend.itemsizing = 'constant'
  fig.frames=frames
  fig.update_traces(marker   = dict(size=1.2),
                    selector = dict(mode='markers'))
  
  fig.layout.updatemenus =[dict(type   ="buttons",
                                buttons=[dict(label  = "Play",
                                              method = "animate",
                                              args   = [None,
                                                        {"frame":       {"duration": 5,'redraw':True},
                                                         "transition":  {"duration":0},
                                                         "fromcurrent": True,
                                                         "mode":        "immediate"}])])]
  fig.show()

def color_code_compare(pred_txt_file):
  data = color_code_setup(pred_txt_file)
  data['color_formatted'] = ['rgb(' + ', '.join([str(data['r'][i]), str(data['g'][i]), str(data['b'][i])])  + ')' for i in range(len(data))]
  data = data.drop(columns=['p'])

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
  data = color_code_setup(pred_txt_file)
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
    trace["name"] = classes[idx]

  fig.layout.legend.itemsizing='constant'
  fig.layout.legend.itemwidth=31
  fig.show()
