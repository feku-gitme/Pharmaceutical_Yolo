from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import os
import sys
import shutil
import argparse

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--datapath', help='Path to data folder containing image and annotation files',
                    required=True)
parser.add_argument('--train_pct', help='Ratio of images to go to train folder; \
                    the rest go to validation folder (example: ".5")',
                    default=.5)

args = parser.parse_args()

data_path = args.datapath
train_percent = float(args.train_pct)

# Check for valid entries
if not os.path.isdir(data_path):
   print('Directory specified by --datapath not found. Verify the path is correct (and uses double back slashes if on Windows) and try again.')
   sys.exit(0)
if train_percent < .01 or train_percent > 0.99:
   print('Invalid entry for train_pct. Please enter a number between .01 and .99.')
   sys.exit(0)
val_percent = 1 - train_percent
test_percent = 0

# Define path to input dataset 
input_image_path = os.path.join(data_path,'images')
input_label_path = os.path.join(data_path,'labels')

# Define paths to image and annotation folders
cwd = os.getcwd()
train_img_path = os.path.join(cwd,'data/train/images')
train_txt_path = os.path.join(cwd,'data/train/labels')
val_img_path = os.path.join(cwd,'data/validation/images')
val_txt_path = os.path.join(cwd,'data/validation/labels')

# Create folders if they don't already exist
for dir_path in [train_img_path, train_txt_path, val_img_path, val_txt_path]:
   if not os.path.exists(dir_path):
      os.makedirs(dir_path)
      print(f'Created folder at {dir_path}.')

# Get list of all images and annotation files
img_file_list = [path for path in Path(input_image_path).rglob('*')]
txt_file_list = [path for path in Path(input_label_path).rglob('*')]

print(f'Number of image files: {len(img_file_list)}')
print(f'Number of annotation files: {len(txt_file_list)}')

# Split the data into training and validation sets
train_img_files, val_img_files, train_txt_files, val_txt_files = train_test_split(img_file_list, txt_file_list, test_size=0.5, random_state=42)

# Copy files to the respective folders
for img_path, txt_path in zip(train_img_files, train_txt_files):
    shutil.copy(img_path, os.path.join(train_img_path, img_path.name))
    shutil.copy(txt_path, os.path.join(train_txt_path, txt_path.name))

for img_path, txt_path in zip(val_img_files, val_txt_files):
    shutil.copy(img_path, os.path.join(val_img_path, img_path.name))
    shutil.copy(txt_path, os.path.join(val_txt_path, txt_path.name))

print('Images moved to train: %d' % len(train_img_files))
print('Images moved to validation: %d' % len(val_img_files))
