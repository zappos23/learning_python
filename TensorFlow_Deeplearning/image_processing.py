import matplotlib.pyplot as plt
import glob
from skimage.color import rgb2lab
from skimage.transform import resize
from collections import namedtuple
import numpy as np

np.random.seed(101)
#%matplotlib inline
Dataset = namedtuple('Dataset',['X','y'])

def to_tf_format(imgs):
    return np.stack([img[:,:,np.newaxis] for img in imgs], axis=0).astype(np.float32)


def read_dataset_ppm(rootpath, n_labels, resize_to):
    images = []
    labels = []

    for c in range(n_labels):
        full_path = rootpath + '/' + format(c, '05d') + '/'
        for img_name in glob.glob(full_path + "*.ppm"):
            img = plt.imread(img_name).astype(np.float32)
            img = rgb2lab(img / 255.0)[:, :, 0]

            if resize_to:
                img = resize(img, resize_to, mode='reflect')

            label = np.zeros((n_labels,), dtype=np.float32)
            label[c] = 1.0
            images.append(img.astype(np.float32))
            labels.append(label)

    return Dataset(X=to_tf_format(images).astype(np.float32), y=np.matrix(labels).astype(np.float32))

N_CLASSES = 43
RESIZED_IMAGE = (32,32)
dataset = read_dataset_ppm('/Users/admin/python_workarea/dataset_ml/GTSRB_dataset/GTSRB_TRAINING/Final_Training/Images', N_CLASSES, RESIZED_IMAGE)

print(dataset.X.shape)
print(dataset.y.shape)
print(dataset.y[0, :])
print(dataset.y[10,:])

from sklearn.model_selection import train_test_split
idx_train, idx_test = train_test_split(range(dataset.X.shape[0]), test_size=0.25, random_state=101)

X_train = dataset.X[idx_train,:,:,:]
y_train = dataset.y[idx_train,:]
X_test = dataset.X[idx_test,:,:,:]
y_test = dataset.y[idx_test,:]
print("X_train data set")
print(X_train.shape)
print("y_train data set")
print(y_train.shape)
print("X_train test set")
print(X_test.shape)
print("y_train test set")
print(y_test.shape)