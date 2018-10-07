import pandas as pd
import pydicom
import numpy as np
from clint.textui import progress
from sklearn.model_selection import train_test_split
from scipy.misc import toimage
from matplotlib import pyplot as plt
import os

def parse_csv(CSV):
    df = pd.read_csv(CSV, delim_whitespace=False, header=0)
    df = df.drop_duplicates(subset=['patientId']).set_index('patientId')
    return df

def load_data_into_memory(DIR, ANNO, ATTRIBUTE):
    if DIR[:-1] != '/': DIR += '/'
    df = parse_csv(ANNO)
    X, y = [], []
    for image_path in progress.bar(os.listdir(DIR)):
        key = os.path.splitext(image_path)[0]
        try:
            mu = df[ATTRIBUTE][key]
            y.append(mu)
            img = pydicom.read_file(DIR + image_path).pixel_array
            X.append(img)
        except:
            print('patient id \'{}\' cannot be found'.format(key))

    x, y = np.array(X), np.array(y)
    print('Loaded {} images into memory'.format(len(y)))
    return x, y

def visualize_images(images, n):
	while n > 0:
		toimage(images[n]).show()
		n -= 1

if __name__ == '__main__':

    TEST_PERCENT = 0.1
    ATTRIBUTE = 'Target'
    ANNO = '../dataset/stage_1_train_labels.csv'
    TRAIN_DIR = '../dataset/sample_data'

    print('Loading Train Data')
    X, Y = load_data_into_memory(DIR=TRAIN_DIR, ANNO=ANNO, ATTRIBUTE=ATTRIBUTE)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_PERCENT, random_state = 51)
    print(X_train.shape, 'training data')
    print(X_test.shape, 'test data')

    # visualize_images(X_train, 2)