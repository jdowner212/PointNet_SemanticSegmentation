# PointNet_SemanticSegmentation
The semantic segmentation portion of the original PointNet repository, converted to compatibility with Tensorflow 2.x and with a 'myCloud' class + utility files for preparing data and visualizing results.

I completed this project through the Interprofessional Program (IPRO) at Illinois Institute of Technology (IIT).

1. Collecting point cloud scans of IIT campus buildings using SiteScape, a LIDAR-equipped iOS application.
2. Performing semantic segmentation on resulting data.

In this context, semantic segmentation refers to the classification of each point in the pointcloud input into one of a predetermined set of categories. The model I used is PointNet -- '_a unified architecture for applications ranging from object classification, part segmentation, to scene semantic parsing_'[^1]. The training data we used[^2] provided the following labels: Clutter, Ceiling, Floor, Wall, Beam, Column, Window, Door, Chair, Table, Bookcase, Sofa, Board. By training PointNet on this particular data, we were able to segment our own pointcloud input into these categories.

![Semantic segmentation on an IIT pointcloud](https://github.com/jdowner212/PointNet_SemanticSegmentation/blob/main/Large%20GIF%20(464x274).gif)

Our original idea was that we could use this tool to create an inventory of various essential components of campus buildings -- plumbing, HVAC equipment, computers, etc. Theoretically, given the appropriate dataset and labeling scheme, PointNet could be used to identify and label other types of objects.


### Notes before trying it out:

Before running, you'll need to download the S3DIS dataset, which requires submitting a Google form. (link: https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1). After downloading, place the unzipped "Stanford3dDataset_v1.2_Aligned_Version" folder within the "data" folder.

View sem_seg/instructions.txt to get started.


<!-- 
This repository references two others:

1. The original PointNet repository,
https://github.com/charlesq34/pointnet

and

2. A modified version with Tensorflow 2.x compatibility.
https://github.com/RobinBaumann/pointnet/
 -->

[^1] http://stanford.edu/~rqi/pointnet/
[^2] http://buildingparser.stanford.edu/dataset.html
