import os
import sys
#import argparse 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
import indoor3d_util_TEST as indoor3d_util


# parser = argparse.ArgumentParser()
# parser.add_argument('--room_name', type=str, required=True)
# parser.add_argument('--log_src', type=str, required=True)
# FLAGS = parser.parse_args()
# room_name = FLAGS.room_name
# log_src   = FLAGS.log_src

#print('room_name:',room_name)
#print('log_src:',log_src)

anno_paths = [[line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/anno_paths_iit.txt'))][-1]]
anno_paths = [os.path.join(indoor3d_util.DATA_PATH, p) for p in anno_paths]
print('anno_paths:',anno_paths)
output_folder = os.path.join(ROOT_DIR, 'data/iit_indoor3d') 
if not os.path.exists(output_folder):
    print('making directory:',output_folder)
    os.mkdir(output_folder)

# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
for anno_path in anno_paths:
    print('anno_path:',anno_path)
    try:
        elements = anno_path.split('/')
        out_filename = elements[-3]+'_'+elements[-2]+'.npy' # Area_1_hallway_1.npy
        indoor3d_util.collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')
    except:
        print(anno_path, 'ERROR!!')
