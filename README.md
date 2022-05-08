# PointNet_SemanticSegmentation
The semantic segmentation portion of the original PointNet repository, converted to compatibility with Tensorflow 2.x and with a 'myCloud' class + utility files for preparing data and visualizing results.

This repository references two others --

1. The original PointNet repository,
https://github.com/charlesq34/pointnet

and

2. A modified version with Tensorflow 2.x compatibility.
https://github.com/RobinBaumann/pointnet/

Before running, you'll need to download the S3DIS dataset, which requires submitting a Google form. (link: https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1). After downloading, place the unzipped "Stanford3dDataset_v1.2_Aligned_Version" folder within the "data" folder.

View PointNet_SemanticSegmentation/sem_seg/training_instructions.txt for instructions on modified features.
