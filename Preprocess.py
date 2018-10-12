from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from PIL.Image import fromarray
import pandas as pd
import pydicom
import os
import zipfile
curr_path = path = os.getcwd()


with zipfile.ZipFile("stage_1_detailed_class_info.csv.zip","r") as zip_ref:
    zip_ref.extractall(curr_path)
    
with zipfile.ZipFile("stage_1_train_labels.csv.zip","r") as zip_ref:
    zip_ref.extractall(curr_path)
    
with zipfile.ZipFile("stage_1_train_images.zip","r") as zip_ref:
    zip_ref.extractall(curr_path+'/data/train')

with zipfile.ZipFile("stage_1_test_images.zip","r") as zip_ref:
    zip_ref.extractall(curr_path+'/data/test')

train_labels = pd.read_csv('stage_1_train_labels.csv')

positive = train_labels[train_labels['Target'] == 1]
negative = train_labels[train_labels['Target'] == 0]
os.mkdir(curr_path + '/data/train/positive')
os.mkdir(curr_path + '/data/train/negative')

for pid in positive['patientId']:
    img_path = 'data/train/{}.dcm'.format(pid)
    if os.path.exists(img_path):
        ds = pydicom.dcmread(img_path)
        im = fromarray(ds.pixel_array)
        im.save('data/train/positive/{}.jpg'.format(pid))
        os.remove(img_path)

for pid in negative['patientId']:
    img_path = 'data/train/{}.dcm'.format(pid)
    if os.path.exists(img_path):
        ds = pydicom.dcmread(img_path)
        im = fromarray(ds.pixel_array)
        im.save('data/train/negative/{}.jpg'.format(pid))
        os.remove(img_path)

for file in os.listdir(curr_path + '/data/test'):
    if file.endswith(".dcm"):
        ds = pydicom.dcmread('data/test/{}'.format(file))
        im = fromarray(ds.pixel_array)
        im.save('data/test/{}.jpg'.format(file[:-4]))
        os.remove('data/test/{}'.format(file))

batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'data/train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# # this is a similar generator, for validation data
# validation_generator = test_datagen.flow_from_directory(
#         'data/validation',
#         target_size=(150, 150),
#         batch_size=batch_size,
#         class_mode='binary')