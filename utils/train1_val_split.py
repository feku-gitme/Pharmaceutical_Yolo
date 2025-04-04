from pathlib import Path
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
                    the rest go to validation folder (example: ".7")',
                    default=.7)

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

# Define path to input dataset
input_image_path = os.path.join(data_path, 'images')
input_label_path = os.path.join(data_path, 'labels')

# Define paths to image and annotation folders
cwd = os.getcwd()
train_img_path = os.path.join(cwd, 'data/train/images')
train_txt_path = os.path.join(cwd, 'data/train/labels')
val_img_path = os.path.join(cwd, 'data/validation/images')
val_txt_path = os.path.join(cwd, 'data/validation/labels')

# Create folders if they don't already exist
for dir_path in [train_img_path, train_txt_path, val_img_path, val_txt_path]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f'Created folder at {dir_path}.')

# Get list of all images and annotation files
img_file_list = [path for path in Path(input_image_path).rglob('*') if path.suffix in ['.jpg', '.jpeg', '.png']]
txt_file_list = [path for path in Path(input_label_path).rglob('*') if path.suffix == '.txt']

print(f'Number of image files: {len(img_file_list)}')
print(f'Number of annotation files: {len(txt_file_list)}')

# Group images by class
class_images = {}
for img_path in img_file_list:
    class_name = img_path.stem.split('_')[0]  # Assuming class name is part of the filename before an underscore
    if class_name not in class_images:
        class_images[class_name] = []
    class_images[class_name].append(img_path)

# Split the images for each class into train and validation sets
train_images = []
val_images = []

for class_name, images in class_images.items():
    num_images = len(images)
    
    # Calculate how many images to take for training
    num_train = int(num_images * train_percent)
    
    # Randomly shuffle images for this class and split them
    random.shuffle(images)
    train_images.extend(images[:num_train])
    val_images.extend(images[num_train:])

    print(f'Class {class_name}: {num_images} images, {num_train} for training, {num_images - num_train} for validation.')

# Copy the selected images and annotations into the respective folders
for img_path in train_images:
    img_fn = img_path.name
    base_fn = img_path.stem
    txt_fn = base_fn + '.txt'
    txt_path = os.path.join(input_label_path, txt_fn)

    # Copy to train folder
    shutil.copy(img_path, os.path.join(train_img_path, img_fn))
    if os.path.exists(txt_path):
        shutil.copy(txt_path, os.path.join(train_txt_path, txt_fn))

for img_path in val_images:
    img_fn = img_path.name
    base_fn = img_path.stem
    txt_fn = base_fn + '.txt'
    txt_path = os.path.join(input_label_path, txt_fn)

    # Copy to validation folder
    shutil.copy(img_path, os.path.join(val_img_path, img_fn))
    if os.path.exists(txt_path):
        shutil.copy(txt_path, os.path.join(val_txt_path, txt_fn))

print('Image and annotation files have been successfully moved to train and validation directories.')
