#!/usr/bin/python3

import argparse
import os
import shutil
import subprocess
exists = os.path.exists
_J_    = os.path.join

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--file_src', type=str, required=True)
parser.add_argument('--log_src',  type=str, required=True)
parser.add_argument('--fixed', type=str,default=False)
FLAGS = parser.parse_args()

FILE_SRC = FLAGS.file_src
LOG_SRC = FLAGS.log_src
FIXED = FLAGS.fixed




def prep_one_file(FILE_SRC, LOG_SRC):
  room      = FILE_SRC.split('/')[-1][:-4]
  room_room = room + '_' + room
  print('filename:',FILE_SRC)

  #tf2       =  ROOT_DIR
  data      =  _J_(ROOT_DIR,'data/IIT_Data')
  semseg    =  BASE_DIR
  meta      =  _J_(semseg,'meta')
  log       =  _J_(semseg, 'log_' + room_room)

  #if FIXED==False:
  for directory in [_J_(data,room),
                    _J_(data,room,room),
                    _J_(data,room,room,'Annotations'),
                    log,
                    _J_(log,'dump')]:
    if not os.path.exists(directory):
      print('making directory: ', directory)
      os.mkdir(directory)
    if directory == _J_(data,room,room) and room + '.txt' not in os.listdir(directory):
      print('copying from',FILE_SRC,'to', _J_(directory, room + '.txt'))
      shutil.copy2(FILE_SRC, _J_(directory, room + '.txt'))
    if directory == _J_(data,room,room,'Annotations') and 'clutter_1.txt\n' not in os.listdir(directory):
      print('copying from', FILE_SRC, 'to', _J_(directory,'clutter_1.txt'))
      shutil.copy2(FILE_SRC, _J_(directory,'clutter_1.txt'))
  for extension in ['.data-00000-of-00001', '.index', '.meta']:
    model_file = 'model.ckpt' + extension
    if _J_(log,model_file+'\n') not in os.listdir(log):
      print('copying from',  _J_(LOG_SRC,model_file),  'to',   _J_(log,model_file))
      shutil.copy2(_J_(LOG_SRC,model_file), _J_(log,model_file))

  for (src,to_add) in [(_J_(meta,'anno_paths_iit.txt'), _J_(room,room,'Annotations')),
                      (_J_(meta,'all_data_label_iit.txt'), room_room + '.npy\n')]:
    with open(src,'r') as f:
      lines = f.readlines()
      f.close()
    instruction = 'a' if os.path.exists(src) else 'w'
    with open(src,instruction) as f:
      if to_add not in lines:
        print('Writing \'' + to_add + '\' to ' + src)
        f.write(to_add + '\n')
      f.close()
  collect = _J_(semseg,'collect_indoor3d_data_TEST_copy.py')
  if room_room + '.npy' not in os.listdir(_J_(ROOT_DIR,'data/iit_indoor3d')):
    to_run1 = '!python ' + _J_(semseg,'collect_indoor3d_data_TEST_copy.py')
    to_run2 = '!python ' + _J_(semseg,'segment_my_data.py --file_src ' + FILE_SRC + ' --log_src ' + LOG_SRC + ' --fixed=True')
    print('\n' + room_room + '.npy not found in data/iit_indoor3d.')
    print('Run these commands:\n')
    print(to_run1)
    print(to_run2 + '\n')
    exit()
  if room_room + '.npy' in os.listdir(_J_(ROOT_DIR,'data/iit_indoor3d')):
    filename = _J_(meta, room_room + '_data_label.txt')#/office_furn_1_office_furn_1_data_label.txt'
    with open(filename,'w') as this_room_label_iit:
      this_room_label_iit.write(_J_('data/iit_indoor3d',room_room +'.npy\n'))
    this_room_label_iit.close()

def segment_my_data():
    sem_seg = BASE_DIR
    prep_one_file(FILE_SRC, LOG_SRC)
    room = FILE_SRC.split('/')[-1][:-4]
    room_room = room + '_' + room
    calls =  ['python',
              _J_(sem_seg,'batch_inference_TEST.py'),
              '--model_path',
              _J_(LOG_SRC,'model.ckpt'),
              '--dump_dir',
              _J_(sem_seg, 'log_' + room_room, 'dump'),
              '--output_filelist',
              _J_(sem_seg, 'log_' + room_room, 'output_filelist.txt'),
              '--room_data_filelist',
              _J_(sem_seg, 'meta', room_room + '_data_label.txt'),
              '--visu']
   # call = ' '.join(calls)
    #return call
    subprocess.run(calls)

if __name__ == "__main__":
    if not os.path.exists(_J_(ROOT_DIR,'data/IIT_Data')):
        os.mkdir(_J_(ROOT_DIR,'data/IIT_Data'))
        print('location of new directory:\n' + _J_(ROOT_DIR,'data/IIT_Data'))
    if not os.path.exists(_J_(BASE_DIR,'meta/anno_paths_iit.txt')):
       with open(_J_(BASE_DIR,'meta/anno_paths_iit.txt'),'w') as anno_paths_txt:
           pass
       anno_paths_txt.close()
       print('location of new file:\n' + _J_(BASE_DIR,'meta/anno_paths_iit.txt'))
    if not os.path.exists(_J_(BASE_DIR,'meta/all_data_label_iit.txt')):
       with open(_J_(BASE_DIR,'meta/all_data_label_iit.txt'),'w') as anno_paths_txt:
           pass
       anno_paths_txt.close()
       print('location of new file:\n' + _J_(BASE_DIR,'meta/all_data_label_iit.txt'))
    segment_my_data()
