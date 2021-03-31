# CS300-Artificial Intelligence : Đồ án cuối kì

---



## I. Sơ lược về đồ án

### 1. Yêu cầu bài toán

### 2. Dữ liệu huấn luyện

## II. Thành phần nhóm đồ án

### 1. Bảng thông tin thành viên nhóm

### 2. Bảng phân công nhiệm vụ thực hiện

## III. Các phương pháp giải quyết bài toán

### 1. Tự thêm vào

### 2. Mạng nơ-ron tích chập - Convolutional Neural Network

## IV. Thực nghiệm

### 1. Tự thêm vào

### 2. Mạng nơ-ron tích chập - Convolutional Neural Network

#### a. Chuẩn bị các thư viện cần thiết


```python
!git clone https://github.com/liem18112000/CS300_FInalTerm.git
```

    fatal: destination path 'CS300_FInalTerm' already exists and is not an empty directory.
    


```python
%load_ext tensorboard
```

    The tensorboard extension is already loaded. To reload it, use:
      %reload_ext tensorboard
    


```python
import tensorflow as tf
from datetime import datetime
import os
from tensorflow.keras.optimizers import RMSprop,Adam
from CS300_FInalTerm.factory import *
from CS300_FInalTerm.loader import *
from CS300_FInalTerm.utility import *

factory = ModelFactory.instance()
loader = Loader()

def train_model(model):
    model.compile(
        optimizer ='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    history = model.fit(
        x_train_1, y_train_1, 
        validation_data = (x_val_1, y_val_1),
        batch_size=BATCH_SIZE, 
        epochs=EPOCHS,
        # callbacks = callbacks + [tensorboard_callback]
        callbacks = callbacks
    )

    return history
```


```python
!nvidia-smi
```

    NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.
    
    

#### b. Dữ liệu huấn luyện : Fashion MNIST


```python
(x_train, y_train), (x_test, y_test) = loader.load_dataset()
```

    Shape of original training examples: (60000, 28, 28)
    Shape of original test examples: (10000, 28, 28)
    Shape of original training result: (60000,)
    Shape of original test result: (10000,)
    


```python
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize=(16,16))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.show()
```


![png](output_20_0.png)


#### c. Các mẩu huấn luyện


```python
(x_train_1, y_train_1), (x_val_1, y_val_1), (x_test_1, y_test_1) = loader.load_dataset_expanddim()
```

    Shape of original training examples: (60000, 28, 28)
    Shape of original validation examples: (8000, 28, 28)
    Shape of original test examples: (2000, 28, 28)
    Shape of original training result: (60000, 10)
    Shape of original validation result: (8000, 10)
    Shape of original test result: (2000, 10)
    


```python
BATCH_SIZE = 4096
EPOCHS = 200
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights= True),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights= True),
]
histories = {}
```


```python
import sys
print ('Running in colab:', 'google.colab' in sys.modules)
```

    Running in colab: True
    

---


```python
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

with strategy.scope():
    model = factory.createMiniVGGModel(32)
    history = train_model(model)
```

    WARNING:tensorflow:TPU system grpc://10.31.158.74:8470 has already been initialized. Reinitializing the TPU can cause previously created variables on TPU to be lost.
    

    WARNING:tensorflow:TPU system grpc://10.31.158.74:8470 has already been initialized. Reinitializing the TPU can cause previously created variables on TPU to be lost.
    

    INFO:tensorflow:Initializing the TPU system: grpc://10.31.158.74:8470
    

    INFO:tensorflow:Initializing the TPU system: grpc://10.31.158.74:8470
    

    INFO:tensorflow:Clearing out eager caches
    

    INFO:tensorflow:Clearing out eager caches
    

    INFO:tensorflow:Finished initializing TPU system.
    

    INFO:tensorflow:Finished initializing TPU system.
    

    INFO:tensorflow:Found TPU system:
    

    INFO:tensorflow:Found TPU system:
    

    INFO:tensorflow:*** Num TPU Cores: 8
    

    INFO:tensorflow:*** Num TPU Cores: 8
    

    INFO:tensorflow:*** Num TPU Workers: 1
    

    INFO:tensorflow:*** Num TPU Workers: 1
    

    INFO:tensorflow:*** Num TPU Cores Per Worker: 8
    

    INFO:tensorflow:*** Num TPU Cores Per Worker: 8
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:CPU:0, CPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:CPU:0, CPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:CPU:0, CPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:CPU:0, CPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:0, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:0, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:1, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:1, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:2, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:2, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:3, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:3, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:4, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:4, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:5, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:5, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:6, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:6, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:7, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:7, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU_SYSTEM:0, TPU_SYSTEM, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU_SYSTEM:0, TPU_SYSTEM, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 0, 0)
    

    Model: "sequential_4"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    up_sampling3d_4 (UpSampling3 (None, 28, 28, 1)         0         
    _________________________________________________________________
    conv2d_48 (Conv2D)           (None, 28, 28, 32)        320       
    _________________________________________________________________
    batch_normalization_56 (Batc (None, 28, 28, 32)        112       
    _________________________________________________________________
    conv2d_49 (Conv2D)           (None, 28, 28, 32)        9248      
    _________________________________________________________________
    batch_normalization_57 (Batc (None, 28, 28, 32)        112       
    _________________________________________________________________
    max_pooling2d_16 (MaxPooling (None, 14, 14, 32)        0         
    _________________________________________________________________
    dropout_16 (Dropout)         (None, 14, 14, 32)        0         
    _________________________________________________________________
    conv2d_50 (Conv2D)           (None, 14, 14, 64)        18496     
    _________________________________________________________________
    batch_normalization_58 (Batc (None, 14, 14, 64)        56        
    _________________________________________________________________
    conv2d_51 (Conv2D)           (None, 14, 14, 64)        36928     
    _________________________________________________________________
    batch_normalization_59 (Batc (None, 14, 14, 64)        56        
    _________________________________________________________________
    max_pooling2d_17 (MaxPooling (None, 7, 7, 64)          0         
    _________________________________________________________________
    dropout_17 (Dropout)         (None, 7, 7, 64)          0         
    _________________________________________________________________
    conv2d_52 (Conv2D)           (None, 7, 7, 128)         73856     
    _________________________________________________________________
    batch_normalization_60 (Batc (None, 7, 7, 128)         28        
    _________________________________________________________________
    conv2d_53 (Conv2D)           (None, 7, 7, 128)         147584    
    _________________________________________________________________
    batch_normalization_61 (Batc (None, 7, 7, 128)         28        
    _________________________________________________________________
    conv2d_54 (Conv2D)           (None, 7, 7, 128)         147584    
    _________________________________________________________________
    batch_normalization_62 (Batc (None, 7, 7, 128)         28        
    _________________________________________________________________
    conv2d_55 (Conv2D)           (None, 7, 7, 128)         147584    
    _________________________________________________________________
    batch_normalization_63 (Batc (None, 7, 7, 128)         28        
    _________________________________________________________________
    max_pooling2d_18 (MaxPooling (None, 3, 3, 128)         0         
    _________________________________________________________________
    dropout_18 (Dropout)         (None, 3, 3, 128)         0         
    _________________________________________________________________
    conv2d_56 (Conv2D)           (None, 3, 3, 256)         295168    
    _________________________________________________________________
    batch_normalization_64 (Batc (None, 3, 3, 256)         12        
    _________________________________________________________________
    conv2d_57 (Conv2D)           (None, 3, 3, 256)         590080    
    _________________________________________________________________
    batch_normalization_65 (Batc (None, 3, 3, 256)         12        
    _________________________________________________________________
    conv2d_58 (Conv2D)           (None, 3, 3, 256)         590080    
    _________________________________________________________________
    batch_normalization_66 (Batc (None, 3, 3, 256)         12        
    _________________________________________________________________
    conv2d_59 (Conv2D)           (None, 3, 3, 256)         590080    
    _________________________________________________________________
    batch_normalization_67 (Batc (None, 3, 3, 256)         12        
    _________________________________________________________________
    max_pooling2d_19 (MaxPooling (None, 1, 1, 256)         0         
    _________________________________________________________________
    dropout_19 (Dropout)         (None, 1, 1, 256)         0         
    _________________________________________________________________
    flatten_4 (Flatten)          (None, 256)               0         
    _________________________________________________________________
    dense_12 (Dense)             (None, 256)               65792     
    _________________________________________________________________
    batch_normalization_68 (Batc (None, 256)               1024      
    _________________________________________________________________
    dense_13 (Dense)             (None, 256)               65792     
    _________________________________________________________________
    batch_normalization_69 (Batc (None, 256)               1024      
    _________________________________________________________________
    dense_14 (Dense)             (None, 10)                2570      
    =================================================================
    Total params: 2,783,706
    Trainable params: 2,782,434
    Non-trainable params: 1,272
    _________________________________________________________________
    Epoch 1/200
    15/15 [==============================] - 26s 783ms/step - loss: 1.9809 - accuracy: 0.3697 - val_loss: 2.2771 - val_accuracy: 0.2098
    Epoch 2/200
    15/15 [==============================] - 1s 81ms/step - loss: 0.6759 - accuracy: 0.7405 - val_loss: 2.3157 - val_accuracy: 0.1598
    Epoch 3/200
    15/15 [==============================] - 1s 83ms/step - loss: 0.5523 - accuracy: 0.7941 - val_loss: 2.3581 - val_accuracy: 0.1585
    Epoch 4/200
    15/15 [==============================] - 1s 82ms/step - loss: 0.4718 - accuracy: 0.8240 - val_loss: 2.4897 - val_accuracy: 0.1873
    Epoch 5/200
    15/15 [==============================] - 1s 81ms/step - loss: 0.4146 - accuracy: 0.8438 - val_loss: 2.5488 - val_accuracy: 0.1531
    Epoch 6/200
    15/15 [==============================] - 1s 81ms/step - loss: 0.3703 - accuracy: 0.8598 - val_loss: 2.4771 - val_accuracy: 0.1493
    Epoch 7/200
    15/15 [==============================] - 1s 83ms/step - loss: 0.3374 - accuracy: 0.8729 - val_loss: 2.7591 - val_accuracy: 0.1011
    Epoch 8/200
    15/15 [==============================] - 1s 85ms/step - loss: 0.3179 - accuracy: 0.8815 - val_loss: 2.4321 - val_accuracy: 0.1011
    Epoch 9/200
    15/15 [==============================] - 1s 84ms/step - loss: 0.2908 - accuracy: 0.8915 - val_loss: 2.2605 - val_accuracy: 0.1300
    Epoch 10/200
    15/15 [==============================] - 1s 84ms/step - loss: 0.2777 - accuracy: 0.8957 - val_loss: 2.1012 - val_accuracy: 0.2080
    Epoch 11/200
    15/15 [==============================] - 2s 105ms/step - loss: 0.2640 - accuracy: 0.9012 - val_loss: 2.2550 - val_accuracy: 0.2086
    Epoch 12/200
    15/15 [==============================] - 1s 84ms/step - loss: 0.2494 - accuracy: 0.9066 - val_loss: 2.5071 - val_accuracy: 0.1556
    Epoch 13/200
    15/15 [==============================] - 1s 84ms/step - loss: 0.2358 - accuracy: 0.9120 - val_loss: 2.0407 - val_accuracy: 0.3328
    Epoch 14/200
    15/15 [==============================] - 1s 82ms/step - loss: 0.2306 - accuracy: 0.9142 - val_loss: 1.8617 - val_accuracy: 0.3283
    Epoch 15/200
    15/15 [==============================] - 1s 83ms/step - loss: 0.2169 - accuracy: 0.9196 - val_loss: 2.1135 - val_accuracy: 0.3524
    Epoch 16/200
    15/15 [==============================] - 1s 83ms/step - loss: 0.2092 - accuracy: 0.9221 - val_loss: 1.6877 - val_accuracy: 0.4415
    Epoch 17/200
    15/15 [==============================] - 1s 81ms/step - loss: 0.2048 - accuracy: 0.9232 - val_loss: 1.5598 - val_accuracy: 0.5215
    Epoch 18/200
    15/15 [==============================] - 1s 82ms/step - loss: 0.1989 - accuracy: 0.9239 - val_loss: 1.3700 - val_accuracy: 0.5694
    Epoch 19/200
    15/15 [==============================] - 1s 80ms/step - loss: 0.1859 - accuracy: 0.9301 - val_loss: 1.2765 - val_accuracy: 0.5816
    Epoch 20/200
    15/15 [==============================] - 1s 81ms/step - loss: 0.1897 - accuracy: 0.9302 - val_loss: 1.2734 - val_accuracy: 0.5614
    Epoch 21/200
    15/15 [==============================] - 1s 82ms/step - loss: 0.1729 - accuracy: 0.9364 - val_loss: 1.1573 - val_accuracy: 0.6168
    Epoch 22/200
    15/15 [==============================] - 1s 82ms/step - loss: 0.1721 - accuracy: 0.9363 - val_loss: 0.8227 - val_accuracy: 0.6919
    Epoch 23/200
    15/15 [==============================] - 1s 86ms/step - loss: 0.1665 - accuracy: 0.9370 - val_loss: 0.7443 - val_accuracy: 0.7499
    Epoch 24/200
    15/15 [==============================] - 1s 82ms/step - loss: 0.1641 - accuracy: 0.9380 - val_loss: 0.9059 - val_accuracy: 0.6936
    Epoch 25/200
    15/15 [==============================] - 1s 84ms/step - loss: 0.1524 - accuracy: 0.9436 - val_loss: 0.6990 - val_accuracy: 0.7663
    Epoch 26/200
    15/15 [==============================] - 1s 81ms/step - loss: 0.1519 - accuracy: 0.9428 - val_loss: 0.6065 - val_accuracy: 0.8006
    Epoch 27/200
    15/15 [==============================] - 1s 81ms/step - loss: 0.1476 - accuracy: 0.9446 - val_loss: 0.4822 - val_accuracy: 0.8293
    Epoch 28/200
    15/15 [==============================] - 1s 84ms/step - loss: 0.1418 - accuracy: 0.9469 - val_loss: 0.4741 - val_accuracy: 0.8379
    Epoch 29/200
    15/15 [==============================] - 1s 81ms/step - loss: 0.1321 - accuracy: 0.9495 - val_loss: 0.5204 - val_accuracy: 0.8284
    Epoch 30/200
    15/15 [==============================] - 1s 80ms/step - loss: 0.1323 - accuracy: 0.9503 - val_loss: 0.4115 - val_accuracy: 0.8635
    Epoch 31/200
    15/15 [==============================] - 1s 81ms/step - loss: 0.1293 - accuracy: 0.9524 - val_loss: 0.3853 - val_accuracy: 0.8786
    Epoch 32/200
    15/15 [==============================] - 1s 80ms/step - loss: 0.1282 - accuracy: 0.9510 - val_loss: 0.3952 - val_accuracy: 0.8713
    Epoch 33/200
    15/15 [==============================] - 1s 80ms/step - loss: 0.1288 - accuracy: 0.9507 - val_loss: 0.3014 - val_accuracy: 0.9019
    Epoch 34/200
    15/15 [==============================] - 1s 82ms/step - loss: 0.1203 - accuracy: 0.9530 - val_loss: 0.2883 - val_accuracy: 0.9065
    Epoch 35/200
    15/15 [==============================] - 1s 81ms/step - loss: 0.1134 - accuracy: 0.9567 - val_loss: 0.2836 - val_accuracy: 0.9105
    Epoch 36/200
    15/15 [==============================] - 1s 84ms/step - loss: 0.1119 - accuracy: 0.9577 - val_loss: 0.2601 - val_accuracy: 0.9174
    Epoch 37/200
    15/15 [==============================] - 1s 82ms/step - loss: 0.1057 - accuracy: 0.9594 - val_loss: 0.2645 - val_accuracy: 0.9203
    Epoch 38/200
    15/15 [==============================] - 1s 81ms/step - loss: 0.1052 - accuracy: 0.9603 - val_loss: 0.2816 - val_accuracy: 0.9106
    Epoch 39/200
    15/15 [==============================] - 1s 84ms/step - loss: 0.0996 - accuracy: 0.9621 - val_loss: 0.2956 - val_accuracy: 0.9108
    Epoch 40/200
    15/15 [==============================] - 1s 83ms/step - loss: 0.0955 - accuracy: 0.9637 - val_loss: 0.2545 - val_accuracy: 0.9251
    Epoch 41/200
    15/15 [==============================] - 1s 82ms/step - loss: 0.0939 - accuracy: 0.9643 - val_loss: 0.2971 - val_accuracy: 0.9151
    Epoch 42/200
    15/15 [==============================] - 1s 84ms/step - loss: 0.0922 - accuracy: 0.9650 - val_loss: 0.3201 - val_accuracy: 0.9124
    Epoch 43/200
    15/15 [==============================] - 1s 80ms/step - loss: 0.0885 - accuracy: 0.9663 - val_loss: 0.2788 - val_accuracy: 0.9236
    Epoch 44/200
    15/15 [==============================] - 1s 81ms/step - loss: 0.0868 - accuracy: 0.9667 - val_loss: 0.2662 - val_accuracy: 0.9250
    Epoch 45/200
    15/15 [==============================] - 1s 81ms/step - loss: 0.0846 - accuracy: 0.9676 - val_loss: 0.2556 - val_accuracy: 0.9315
    Epoch 46/200
    15/15 [==============================] - 1s 82ms/step - loss: 0.0801 - accuracy: 0.9702 - val_loss: 0.2697 - val_accuracy: 0.9234
    Epoch 47/200
    15/15 [==============================] - 1s 82ms/step - loss: 0.0792 - accuracy: 0.9702 - val_loss: 0.2656 - val_accuracy: 0.9276
    Epoch 48/200
    15/15 [==============================] - 2s 116ms/step - loss: 0.0738 - accuracy: 0.9720 - val_loss: 0.2543 - val_accuracy: 0.9294
    Epoch 49/200
    15/15 [==============================] - 1s 81ms/step - loss: 0.0704 - accuracy: 0.9744 - val_loss: 0.2651 - val_accuracy: 0.9259
    Epoch 50/200
    15/15 [==============================] - 1s 83ms/step - loss: 0.0718 - accuracy: 0.9731 - val_loss: 0.2638 - val_accuracy: 0.9290
    Epoch 51/200
    15/15 [==============================] - 1s 85ms/step - loss: 0.0681 - accuracy: 0.9743 - val_loss: 0.2751 - val_accuracy: 0.9299
    Epoch 52/200
    15/15 [==============================] - 1s 80ms/step - loss: 0.0653 - accuracy: 0.9753 - val_loss: 0.2788 - val_accuracy: 0.9293
    Epoch 53/200
    15/15 [==============================] - 1s 80ms/step - loss: 0.0632 - accuracy: 0.9748 - val_loss: 0.2862 - val_accuracy: 0.9281
    Epoch 54/200
    15/15 [==============================] - 1s 79ms/step - loss: 0.0605 - accuracy: 0.9779 - val_loss: 0.2844 - val_accuracy: 0.9346
    Epoch 55/200
    15/15 [==============================] - 1s 81ms/step - loss: 0.0604 - accuracy: 0.9772 - val_loss: 0.2921 - val_accuracy: 0.9278
    Epoch 56/200
    15/15 [==============================] - 1s 81ms/step - loss: 0.0565 - accuracy: 0.9784 - val_loss: 0.2894 - val_accuracy: 0.9296
    Epoch 57/200
    15/15 [==============================] - 1s 81ms/step - loss: 0.0589 - accuracy: 0.9772 - val_loss: 0.2884 - val_accuracy: 0.9284
    Epoch 58/200
    15/15 [==============================] - 1s 80ms/step - loss: 0.0621 - accuracy: 0.9769 - val_loss: 0.3007 - val_accuracy: 0.9268
    Epoch 59/200
    15/15 [==============================] - 1s 82ms/step - loss: 0.0592 - accuracy: 0.9768 - val_loss: 0.2847 - val_accuracy: 0.9324
    Epoch 60/200
    15/15 [==============================] - 1s 82ms/step - loss: 0.0561 - accuracy: 0.9787 - val_loss: 0.3094 - val_accuracy: 0.9298
    Epoch 61/200
    15/15 [==============================] - 1s 80ms/step - loss: 0.0470 - accuracy: 0.9821 - val_loss: 0.3001 - val_accuracy: 0.9306
    Epoch 62/200
    15/15 [==============================] - 1s 82ms/step - loss: 0.0496 - accuracy: 0.9814 - val_loss: 0.3146 - val_accuracy: 0.9313
    Epoch 63/200
    15/15 [==============================] - 1s 79ms/step - loss: 0.0522 - accuracy: 0.9804 - val_loss: 0.2977 - val_accuracy: 0.9314
    Epoch 64/200
    15/15 [==============================] - 1s 79ms/step - loss: 0.0442 - accuracy: 0.9825 - val_loss: 0.3215 - val_accuracy: 0.9296
    Epoch 65/200
    15/15 [==============================] - 1s 81ms/step - loss: 0.0462 - accuracy: 0.9829 - val_loss: 0.3029 - val_accuracy: 0.9299
    Epoch 66/200
    15/15 [==============================] - 1s 82ms/step - loss: 0.0472 - accuracy: 0.9827 - val_loss: 0.3406 - val_accuracy: 0.9288
    Epoch 67/200
    15/15 [==============================] - 1s 80ms/step - loss: 0.0448 - accuracy: 0.9836 - val_loss: 0.3225 - val_accuracy: 0.9308
    Epoch 68/200
    15/15 [==============================] - 1s 81ms/step - loss: 0.0422 - accuracy: 0.9841 - val_loss: 0.3263 - val_accuracy: 0.9291
    


```python
histories['VGG19_32'] = (history, model.evaluate(x_test_1, y_test_1, BATCH_SIZE))
```

    1/1 [==============================] - 2s 2s/step - loss: 0.2290 - accuracy: 0.9370
    

---


```python
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

with strategy.scope():
    model = factory.createMiniVGGModel(64)
    history = train_model(model)
```

    WARNING:tensorflow:TPU system grpc://10.31.158.74:8470 has already been initialized. Reinitializing the TPU can cause previously created variables on TPU to be lost.
    

    WARNING:tensorflow:TPU system grpc://10.31.158.74:8470 has already been initialized. Reinitializing the TPU can cause previously created variables on TPU to be lost.
    

    INFO:tensorflow:Initializing the TPU system: grpc://10.31.158.74:8470
    

    INFO:tensorflow:Initializing the TPU system: grpc://10.31.158.74:8470
    

    INFO:tensorflow:Clearing out eager caches
    

    INFO:tensorflow:Clearing out eager caches
    

    INFO:tensorflow:Finished initializing TPU system.
    

    INFO:tensorflow:Finished initializing TPU system.
    

    INFO:tensorflow:Found TPU system:
    

    INFO:tensorflow:Found TPU system:
    

    INFO:tensorflow:*** Num TPU Cores: 8
    

    INFO:tensorflow:*** Num TPU Cores: 8
    

    INFO:tensorflow:*** Num TPU Workers: 1
    

    INFO:tensorflow:*** Num TPU Workers: 1
    

    INFO:tensorflow:*** Num TPU Cores Per Worker: 8
    

    INFO:tensorflow:*** Num TPU Cores Per Worker: 8
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:CPU:0, CPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:CPU:0, CPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:CPU:0, CPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:CPU:0, CPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:0, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:0, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:1, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:1, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:2, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:2, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:3, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:3, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:4, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:4, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:5, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:5, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:6, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:6, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:7, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:7, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU_SYSTEM:0, TPU_SYSTEM, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU_SYSTEM:0, TPU_SYSTEM, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 0, 0)
    

    Model: "sequential_5"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    up_sampling3d_5 (UpSampling3 (None, 28, 28, 1)         0         
    _________________________________________________________________
    conv2d_60 (Conv2D)           (None, 28, 28, 64)        640       
    _________________________________________________________________
    batch_normalization_70 (Batc (None, 28, 28, 64)        112       
    _________________________________________________________________
    conv2d_61 (Conv2D)           (None, 28, 28, 64)        36928     
    _________________________________________________________________
    batch_normalization_71 (Batc (None, 28, 28, 64)        112       
    _________________________________________________________________
    max_pooling2d_20 (MaxPooling (None, 14, 14, 64)        0         
    _________________________________________________________________
    dropout_20 (Dropout)         (None, 14, 14, 64)        0         
    _________________________________________________________________
    conv2d_62 (Conv2D)           (None, 14, 14, 128)       73856     
    _________________________________________________________________
    batch_normalization_72 (Batc (None, 14, 14, 128)       56        
    _________________________________________________________________
    conv2d_63 (Conv2D)           (None, 14, 14, 128)       147584    
    _________________________________________________________________
    batch_normalization_73 (Batc (None, 14, 14, 128)       56        
    _________________________________________________________________
    max_pooling2d_21 (MaxPooling (None, 7, 7, 128)         0         
    _________________________________________________________________
    dropout_21 (Dropout)         (None, 7, 7, 128)         0         
    _________________________________________________________________
    conv2d_64 (Conv2D)           (None, 7, 7, 256)         295168    
    _________________________________________________________________
    batch_normalization_74 (Batc (None, 7, 7, 256)         28        
    _________________________________________________________________
    conv2d_65 (Conv2D)           (None, 7, 7, 256)         590080    
    _________________________________________________________________
    batch_normalization_75 (Batc (None, 7, 7, 256)         28        
    _________________________________________________________________
    conv2d_66 (Conv2D)           (None, 7, 7, 256)         590080    
    _________________________________________________________________
    batch_normalization_76 (Batc (None, 7, 7, 256)         28        
    _________________________________________________________________
    conv2d_67 (Conv2D)           (None, 7, 7, 256)         590080    
    _________________________________________________________________
    batch_normalization_77 (Batc (None, 7, 7, 256)         28        
    _________________________________________________________________
    max_pooling2d_22 (MaxPooling (None, 3, 3, 256)         0         
    _________________________________________________________________
    dropout_22 (Dropout)         (None, 3, 3, 256)         0         
    _________________________________________________________________
    conv2d_68 (Conv2D)           (None, 3, 3, 512)         1180160   
    _________________________________________________________________
    batch_normalization_78 (Batc (None, 3, 3, 512)         12        
    _________________________________________________________________
    conv2d_69 (Conv2D)           (None, 3, 3, 512)         2359808   
    _________________________________________________________________
    batch_normalization_79 (Batc (None, 3, 3, 512)         12        
    _________________________________________________________________
    conv2d_70 (Conv2D)           (None, 3, 3, 512)         2359808   
    _________________________________________________________________
    batch_normalization_80 (Batc (None, 3, 3, 512)         12        
    _________________________________________________________________
    conv2d_71 (Conv2D)           (None, 3, 3, 512)         2359808   
    _________________________________________________________________
    batch_normalization_81 (Batc (None, 3, 3, 512)         12        
    _________________________________________________________________
    max_pooling2d_23 (MaxPooling (None, 1, 1, 512)         0         
    _________________________________________________________________
    dropout_23 (Dropout)         (None, 1, 1, 512)         0         
    _________________________________________________________________
    flatten_5 (Flatten)          (None, 512)               0         
    _________________________________________________________________
    dense_15 (Dense)             (None, 512)               262656    
    _________________________________________________________________
    batch_normalization_82 (Batc (None, 512)               2048      
    _________________________________________________________________
    dense_16 (Dense)             (None, 512)               262656    
    _________________________________________________________________
    batch_normalization_83 (Batc (None, 512)               2048      
    _________________________________________________________________
    dense_17 (Dense)             (None, 10)                5130      
    =================================================================
    Total params: 11,119,034
    Trainable params: 11,116,738
    Non-trainable params: 2,296
    _________________________________________________________________
    Epoch 1/200
    15/15 [==============================] - 43s 2s/step - loss: 2.1510 - accuracy: 0.3452 - val_loss: 1.9289 - val_accuracy: 0.2408
    Epoch 2/200
    15/15 [==============================] - 2s 118ms/step - loss: 0.7127 - accuracy: 0.7297 - val_loss: 3.1997 - val_accuracy: 0.1334
    Epoch 3/200
    15/15 [==============================] - 2s 118ms/step - loss: 0.5751 - accuracy: 0.7838 - val_loss: 3.2515 - val_accuracy: 0.0993
    Epoch 4/200
    15/15 [==============================] - 2s 117ms/step - loss: 0.5137 - accuracy: 0.8082 - val_loss: 3.0022 - val_accuracy: 0.1066
    Epoch 5/200
    15/15 [==============================] - 2s 121ms/step - loss: 0.4632 - accuracy: 0.8268 - val_loss: 2.8594 - val_accuracy: 0.1374
    Epoch 6/200
    15/15 [==============================] - 2s 118ms/step - loss: 0.4223 - accuracy: 0.8418 - val_loss: 2.6877 - val_accuracy: 0.1831
    Epoch 7/200
    15/15 [==============================] - 2s 118ms/step - loss: 0.3866 - accuracy: 0.8551 - val_loss: 3.4796 - val_accuracy: 0.1733
    Epoch 8/200
    15/15 [==============================] - 2s 118ms/step - loss: 0.3558 - accuracy: 0.8671 - val_loss: 3.2986 - val_accuracy: 0.0986
    Epoch 9/200
    15/15 [==============================] - 2s 119ms/step - loss: 0.3256 - accuracy: 0.8777 - val_loss: 2.9048 - val_accuracy: 0.0336
    Epoch 10/200
    15/15 [==============================] - 2s 118ms/step - loss: 0.3046 - accuracy: 0.8869 - val_loss: 2.8676 - val_accuracy: 0.1024
    Epoch 11/200
    15/15 [==============================] - 2s 118ms/step - loss: 0.2804 - accuracy: 0.8984 - val_loss: 2.6008 - val_accuracy: 0.2020
    Epoch 12/200
    15/15 [==============================] - 2s 116ms/step - loss: 0.2656 - accuracy: 0.9021 - val_loss: 2.3834 - val_accuracy: 0.3154
    Epoch 13/200
    15/15 [==============================] - 2s 117ms/step - loss: 0.2503 - accuracy: 0.9090 - val_loss: 2.3958 - val_accuracy: 0.2610
    Epoch 14/200
    15/15 [==============================] - 2s 142ms/step - loss: 0.2335 - accuracy: 0.9138 - val_loss: 2.7149 - val_accuracy: 0.1264
    Epoch 15/200
    15/15 [==============================] - 2s 117ms/step - loss: 0.2260 - accuracy: 0.9166 - val_loss: 2.6027 - val_accuracy: 0.2435
    Epoch 16/200
    15/15 [==============================] - 2s 118ms/step - loss: 0.2164 - accuracy: 0.9185 - val_loss: 2.6427 - val_accuracy: 0.1974
    Epoch 17/200
    15/15 [==============================] - 2s 117ms/step - loss: 0.2015 - accuracy: 0.9257 - val_loss: 2.6884 - val_accuracy: 0.3071
    Epoch 18/200
    15/15 [==============================] - 2s 122ms/step - loss: 0.1901 - accuracy: 0.9298 - val_loss: 2.4228 - val_accuracy: 0.2650
    Epoch 19/200
    15/15 [==============================] - 2s 119ms/step - loss: 0.1850 - accuracy: 0.9304 - val_loss: 2.7339 - val_accuracy: 0.2544
    Epoch 20/200
    15/15 [==============================] - 2s 117ms/step - loss: 0.1838 - accuracy: 0.9307 - val_loss: 1.5574 - val_accuracy: 0.4418
    Epoch 21/200
    15/15 [==============================] - 2s 117ms/step - loss: 0.1733 - accuracy: 0.9377 - val_loss: 1.3369 - val_accuracy: 0.5000
    Epoch 22/200
    15/15 [==============================] - 2s 118ms/step - loss: 0.1621 - accuracy: 0.9396 - val_loss: 1.7079 - val_accuracy: 0.4620
    Epoch 23/200
    15/15 [==============================] - 2s 115ms/step - loss: 0.1530 - accuracy: 0.9446 - val_loss: 1.3987 - val_accuracy: 0.5229
    Epoch 24/200
    15/15 [==============================] - 2s 116ms/step - loss: 0.1448 - accuracy: 0.9463 - val_loss: 1.5279 - val_accuracy: 0.5093
    Epoch 25/200
    15/15 [==============================] - 2s 117ms/step - loss: 0.1431 - accuracy: 0.9469 - val_loss: 1.1812 - val_accuracy: 0.5830
    Epoch 26/200
    15/15 [==============================] - 2s 119ms/step - loss: 0.1362 - accuracy: 0.9492 - val_loss: 1.0448 - val_accuracy: 0.6384
    Epoch 27/200
    15/15 [==============================] - 2s 118ms/step - loss: 0.1271 - accuracy: 0.9524 - val_loss: 0.7727 - val_accuracy: 0.7378
    Epoch 28/200
    15/15 [==============================] - 2s 120ms/step - loss: 0.1249 - accuracy: 0.9523 - val_loss: 0.9178 - val_accuracy: 0.6991
    Epoch 29/200
    15/15 [==============================] - 2s 118ms/step - loss: 0.1132 - accuracy: 0.9575 - val_loss: 0.6253 - val_accuracy: 0.7851
    Epoch 30/200
    15/15 [==============================] - 2s 118ms/step - loss: 0.1135 - accuracy: 0.9577 - val_loss: 0.5432 - val_accuracy: 0.8205
    Epoch 31/200
    15/15 [==============================] - 2s 118ms/step - loss: 0.1041 - accuracy: 0.9607 - val_loss: 0.6760 - val_accuracy: 0.7851
    Epoch 32/200
    15/15 [==============================] - 2s 117ms/step - loss: 0.1083 - accuracy: 0.9605 - val_loss: 0.4196 - val_accuracy: 0.8574
    Epoch 33/200
    15/15 [==============================] - 2s 121ms/step - loss: 0.1012 - accuracy: 0.9619 - val_loss: 0.4294 - val_accuracy: 0.8571
    Epoch 34/200
    15/15 [==============================] - 2s 116ms/step - loss: 0.0964 - accuracy: 0.9634 - val_loss: 0.4638 - val_accuracy: 0.8594
    Epoch 35/200
    15/15 [==============================] - 2s 118ms/step - loss: 0.0908 - accuracy: 0.9666 - val_loss: 0.3566 - val_accuracy: 0.8914
    Epoch 36/200
    15/15 [==============================] - 2s 117ms/step - loss: 0.0860 - accuracy: 0.9673 - val_loss: 0.3918 - val_accuracy: 0.8836
    Epoch 37/200
    15/15 [==============================] - 2s 118ms/step - loss: 0.0811 - accuracy: 0.9698 - val_loss: 0.3042 - val_accuracy: 0.9071
    Epoch 38/200
    15/15 [==============================] - 2s 115ms/step - loss: 0.0782 - accuracy: 0.9714 - val_loss: 0.3153 - val_accuracy: 0.9046
    Epoch 39/200
    15/15 [==============================] - 2s 117ms/step - loss: 0.0728 - accuracy: 0.9733 - val_loss: 0.3423 - val_accuracy: 0.9138
    Epoch 40/200
    15/15 [==============================] - 2s 117ms/step - loss: 0.0714 - accuracy: 0.9729 - val_loss: 0.2839 - val_accuracy: 0.9151
    Epoch 41/200
    15/15 [==============================] - 2s 117ms/step - loss: 0.0692 - accuracy: 0.9736 - val_loss: 0.2826 - val_accuracy: 0.9198
    Epoch 42/200
    15/15 [==============================] - 2s 119ms/step - loss: 0.0623 - accuracy: 0.9767 - val_loss: 0.2803 - val_accuracy: 0.9234
    Epoch 43/200
    15/15 [==============================] - 2s 118ms/step - loss: 0.0587 - accuracy: 0.9777 - val_loss: 0.2826 - val_accuracy: 0.9265
    Epoch 44/200
    15/15 [==============================] - 2s 117ms/step - loss: 0.0602 - accuracy: 0.9778 - val_loss: 0.2529 - val_accuracy: 0.9311
    Epoch 45/200
    15/15 [==============================] - 2s 117ms/step - loss: 0.0587 - accuracy: 0.9788 - val_loss: 0.3087 - val_accuracy: 0.9240
    Epoch 46/200
    15/15 [==============================] - 2s 117ms/step - loss: 0.0528 - accuracy: 0.9802 - val_loss: 0.2829 - val_accuracy: 0.9273
    Epoch 47/200
    15/15 [==============================] - 2s 118ms/step - loss: 0.0489 - accuracy: 0.9817 - val_loss: 0.2978 - val_accuracy: 0.9240
    Epoch 48/200
    15/15 [==============================] - 2s 117ms/step - loss: 0.0556 - accuracy: 0.9788 - val_loss: 0.3086 - val_accuracy: 0.9288
    Epoch 49/200
    15/15 [==============================] - 2s 118ms/step - loss: 0.0455 - accuracy: 0.9830 - val_loss: 0.2880 - val_accuracy: 0.9298
    Epoch 50/200
    15/15 [==============================] - 2s 119ms/step - loss: 0.0424 - accuracy: 0.9843 - val_loss: 0.2999 - val_accuracy: 0.9345
    Epoch 51/200
    15/15 [==============================] - 2s 119ms/step - loss: 0.0434 - accuracy: 0.9845 - val_loss: 0.2772 - val_accuracy: 0.9354
    Epoch 52/200
    15/15 [==============================] - 2s 121ms/step - loss: 0.0379 - accuracy: 0.9865 - val_loss: 0.2997 - val_accuracy: 0.9341
    Epoch 53/200
    15/15 [==============================] - 2s 122ms/step - loss: 0.0342 - accuracy: 0.9877 - val_loss: 0.3029 - val_accuracy: 0.9335
    Epoch 54/200
    15/15 [==============================] - 2s 119ms/step - loss: 0.0346 - accuracy: 0.9877 - val_loss: 0.2916 - val_accuracy: 0.9353
    Epoch 55/200
    15/15 [==============================] - 2s 117ms/step - loss: 0.0348 - accuracy: 0.9875 - val_loss: 0.3180 - val_accuracy: 0.9343
    Epoch 56/200
    15/15 [==============================] - 2s 118ms/step - loss: 0.0336 - accuracy: 0.9881 - val_loss: 0.3493 - val_accuracy: 0.9284
    Epoch 57/200
    15/15 [==============================] - 2s 116ms/step - loss: 0.0314 - accuracy: 0.9884 - val_loss: 0.3542 - val_accuracy: 0.9300
    Epoch 58/200
    15/15 [==============================] - 2s 159ms/step - loss: 0.0337 - accuracy: 0.9873 - val_loss: 0.3263 - val_accuracy: 0.9330
    Epoch 59/200
    15/15 [==============================] - 2s 118ms/step - loss: 0.0310 - accuracy: 0.9886 - val_loss: 0.3227 - val_accuracy: 0.9364
    Epoch 60/200
    15/15 [==============================] - 2s 119ms/step - loss: 0.0268 - accuracy: 0.9900 - val_loss: 0.3243 - val_accuracy: 0.9330
    Epoch 61/200
    15/15 [==============================] - 2s 118ms/step - loss: 0.0273 - accuracy: 0.9905 - val_loss: 0.3317 - val_accuracy: 0.9344
    Epoch 62/200
    15/15 [==============================] - 2s 118ms/step - loss: 0.0256 - accuracy: 0.9909 - val_loss: 0.3586 - val_accuracy: 0.9269
    Epoch 63/200
    15/15 [==============================] - 2s 116ms/step - loss: 0.0256 - accuracy: 0.9906 - val_loss: 0.3653 - val_accuracy: 0.9314
    Epoch 64/200
    15/15 [==============================] - 2s 118ms/step - loss: 0.0224 - accuracy: 0.9919 - val_loss: 0.3441 - val_accuracy: 0.9323
    


```python
histories['VGG19_64'] = (history, model.evaluate(x_test_1, y_test_1, BATCH_SIZE))
```

    1/1 [==============================] - 2s 2s/step - loss: 0.2600 - accuracy: 0.9250
    

---


```python
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

with strategy.scope():
    model = factory.createMiniVGGModel(128)
    history = train_model(model)
```

    WARNING:tensorflow:TPU system grpc://10.31.158.74:8470 has already been initialized. Reinitializing the TPU can cause previously created variables on TPU to be lost.
    

    WARNING:tensorflow:TPU system grpc://10.31.158.74:8470 has already been initialized. Reinitializing the TPU can cause previously created variables on TPU to be lost.
    

    INFO:tensorflow:Initializing the TPU system: grpc://10.31.158.74:8470
    

    INFO:tensorflow:Initializing the TPU system: grpc://10.31.158.74:8470
    

    INFO:tensorflow:Clearing out eager caches
    

    INFO:tensorflow:Clearing out eager caches
    

    INFO:tensorflow:Finished initializing TPU system.
    

    INFO:tensorflow:Finished initializing TPU system.
    

    INFO:tensorflow:Found TPU system:
    

    INFO:tensorflow:Found TPU system:
    

    INFO:tensorflow:*** Num TPU Cores: 8
    

    INFO:tensorflow:*** Num TPU Cores: 8
    

    INFO:tensorflow:*** Num TPU Workers: 1
    

    INFO:tensorflow:*** Num TPU Workers: 1
    

    INFO:tensorflow:*** Num TPU Cores Per Worker: 8
    

    INFO:tensorflow:*** Num TPU Cores Per Worker: 8
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:CPU:0, CPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:CPU:0, CPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:CPU:0, CPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:CPU:0, CPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:0, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:0, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:1, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:1, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:2, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:2, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:3, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:3, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:4, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:4, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:5, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:5, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:6, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:6, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:7, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:7, TPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU_SYSTEM:0, TPU_SYSTEM, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU_SYSTEM:0, TPU_SYSTEM, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 0, 0)
    

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 0, 0)
    

    Model: "sequential_6"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    up_sampling3d_6 (UpSampling3 (None, 28, 28, 1)         0         
    _________________________________________________________________
    conv2d_72 (Conv2D)           (None, 28, 28, 128)       1280      
    _________________________________________________________________
    batch_normalization_84 (Batc (None, 28, 28, 128)       112       
    _________________________________________________________________
    conv2d_73 (Conv2D)           (None, 28, 28, 128)       147584    
    _________________________________________________________________
    batch_normalization_85 (Batc (None, 28, 28, 128)       112       
    _________________________________________________________________
    max_pooling2d_24 (MaxPooling (None, 14, 14, 128)       0         
    _________________________________________________________________
    dropout_24 (Dropout)         (None, 14, 14, 128)       0         
    _________________________________________________________________
    conv2d_74 (Conv2D)           (None, 14, 14, 256)       295168    
    _________________________________________________________________
    batch_normalization_86 (Batc (None, 14, 14, 256)       56        
    _________________________________________________________________
    conv2d_75 (Conv2D)           (None, 14, 14, 256)       590080    
    _________________________________________________________________
    batch_normalization_87 (Batc (None, 14, 14, 256)       56        
    _________________________________________________________________
    max_pooling2d_25 (MaxPooling (None, 7, 7, 256)         0         
    _________________________________________________________________
    dropout_25 (Dropout)         (None, 7, 7, 256)         0         
    _________________________________________________________________
    conv2d_76 (Conv2D)           (None, 7, 7, 512)         1180160   
    _________________________________________________________________
    batch_normalization_88 (Batc (None, 7, 7, 512)         28        
    _________________________________________________________________
    conv2d_77 (Conv2D)           (None, 7, 7, 512)         2359808   
    _________________________________________________________________
    batch_normalization_89 (Batc (None, 7, 7, 512)         28        
    _________________________________________________________________
    conv2d_78 (Conv2D)           (None, 7, 7, 512)         2359808   
    _________________________________________________________________
    batch_normalization_90 (Batc (None, 7, 7, 512)         28        
    _________________________________________________________________
    conv2d_79 (Conv2D)           (None, 7, 7, 512)         2359808   
    _________________________________________________________________
    batch_normalization_91 (Batc (None, 7, 7, 512)         28        
    _________________________________________________________________
    max_pooling2d_26 (MaxPooling (None, 3, 3, 512)         0         
    _________________________________________________________________
    dropout_26 (Dropout)         (None, 3, 3, 512)         0         
    _________________________________________________________________
    conv2d_80 (Conv2D)           (None, 3, 3, 1024)        4719616   
    _________________________________________________________________
    batch_normalization_92 (Batc (None, 3, 3, 1024)        12        
    _________________________________________________________________
    conv2d_81 (Conv2D)           (None, 3, 3, 1024)        9438208   
    _________________________________________________________________
    batch_normalization_93 (Batc (None, 3, 3, 1024)        12        
    _________________________________________________________________
    conv2d_82 (Conv2D)           (None, 3, 3, 1024)        9438208   
    _________________________________________________________________
    batch_normalization_94 (Batc (None, 3, 3, 1024)        12        
    _________________________________________________________________
    conv2d_83 (Conv2D)           (None, 3, 3, 1024)        9438208   
    _________________________________________________________________
    batch_normalization_95 (Batc (None, 3, 3, 1024)        12        
    _________________________________________________________________
    max_pooling2d_27 (MaxPooling (None, 1, 1, 1024)        0         
    _________________________________________________________________
    dropout_27 (Dropout)         (None, 1, 1, 1024)        0         
    _________________________________________________________________
    flatten_6 (Flatten)          (None, 1024)              0         
    _________________________________________________________________
    dense_18 (Dense)             (None, 1024)              1049600   
    _________________________________________________________________
    batch_normalization_96 (Batc (None, 1024)              4096      
    _________________________________________________________________
    dense_19 (Dense)             (None, 1024)              1049600   
    _________________________________________________________________
    batch_normalization_97 (Batc (None, 1024)              4096      
    _________________________________________________________________
    dense_20 (Dense)             (None, 10)                10250     
    =================================================================
    Total params: 44,446,074
    Trainable params: 44,441,730
    Non-trainable params: 4,344
    _________________________________________________________________
    Epoch 1/200
    15/15 [==============================] - 58s 2s/step - loss: 3.3322 - accuracy: 0.2095 - val_loss: 6657.1353 - val_accuracy: 0.1040
    Epoch 2/200
    15/15 [==============================] - 4s 237ms/step - loss: 1.2079 - accuracy: 0.5408 - val_loss: 7004.7500 - val_accuracy: 0.1011
    Epoch 3/200
    15/15 [==============================] - 4s 238ms/step - loss: 0.9198 - accuracy: 0.6505 - val_loss: 1557.5841 - val_accuracy: 0.1011
    Epoch 4/200
    15/15 [==============================] - 4s 236ms/step - loss: 0.8217 - accuracy: 0.6896 - val_loss: 696.1893 - val_accuracy: 0.1011
    Epoch 5/200
    15/15 [==============================] - 4s 237ms/step - loss: 0.7349 - accuracy: 0.7264 - val_loss: 223.7329 - val_accuracy: 0.1011
    Epoch 6/200
    15/15 [==============================] - 4s 237ms/step - loss: 0.6951 - accuracy: 0.7389 - val_loss: 70.2052 - val_accuracy: 0.1011
    Epoch 7/200
    15/15 [==============================] - 4s 238ms/step - loss: 0.6340 - accuracy: 0.7631 - val_loss: 30.7393 - val_accuracy: 0.1011
    Epoch 8/200
    15/15 [==============================] - 4s 238ms/step - loss: 0.6020 - accuracy: 0.7808 - val_loss: 13.4191 - val_accuracy: 0.1011
    Epoch 9/200
    15/15 [==============================] - 3s 236ms/step - loss: 0.5553 - accuracy: 0.7993 - val_loss: 6.7049 - val_accuracy: 0.1011
    Epoch 10/200
    15/15 [==============================] - 4s 237ms/step - loss: 0.5225 - accuracy: 0.8106 - val_loss: 4.7010 - val_accuracy: 0.1186
    Epoch 11/200
    15/15 [==============================] - 4s 238ms/step - loss: 0.4806 - accuracy: 0.8259 - val_loss: 4.8200 - val_accuracy: 0.0996
    Epoch 12/200
    15/15 [==============================] - 4s 238ms/step - loss: 0.4415 - accuracy: 0.8358 - val_loss: 4.7842 - val_accuracy: 0.1304
    Epoch 13/200
    15/15 [==============================] - 4s 237ms/step - loss: 0.4202 - accuracy: 0.8438 - val_loss: 4.8230 - val_accuracy: 0.1004
    Epoch 14/200
    15/15 [==============================] - 4s 240ms/step - loss: 0.3918 - accuracy: 0.8533 - val_loss: 3.6446 - val_accuracy: 0.2834
    Epoch 15/200
    15/15 [==============================] - 4s 237ms/step - loss: 0.3740 - accuracy: 0.8608 - val_loss: 6.5257 - val_accuracy: 0.1016
    Epoch 16/200
    15/15 [==============================] - 4s 238ms/step - loss: 0.3541 - accuracy: 0.8697 - val_loss: 8.5842 - val_accuracy: 0.0986
    Epoch 17/200
    15/15 [==============================] - 4s 241ms/step - loss: 0.3295 - accuracy: 0.8789 - val_loss: 8.0173 - val_accuracy: 0.0986
    Epoch 18/200
    15/15 [==============================] - 4s 240ms/step - loss: 0.3099 - accuracy: 0.8841 - val_loss: 7.5043 - val_accuracy: 0.0986
    Epoch 19/200
    15/15 [==============================] - 4s 239ms/step - loss: 0.2842 - accuracy: 0.8968 - val_loss: 7.3609 - val_accuracy: 0.0986
    Epoch 20/200
    15/15 [==============================] - 4s 238ms/step - loss: 0.2846 - accuracy: 0.8956 - val_loss: 7.7073 - val_accuracy: 0.0986
    Epoch 21/200
    15/15 [==============================] - 4s 238ms/step - loss: 0.2732 - accuracy: 0.8992 - val_loss: 8.3935 - val_accuracy: 0.0988
    Epoch 22/200
    15/15 [==============================] - 4s 238ms/step - loss: 0.2452 - accuracy: 0.9100 - val_loss: 9.0512 - val_accuracy: 0.0986
    Epoch 23/200
    15/15 [==============================] - 4s 237ms/step - loss: 0.2369 - accuracy: 0.9133 - val_loss: 6.0023 - val_accuracy: 0.1360
    Epoch 24/200
    15/15 [==============================] - 4s 237ms/step - loss: 0.2255 - accuracy: 0.9169 - val_loss: 5.8161 - val_accuracy: 0.1300
    Epoch 25/200
    15/15 [==============================] - 4s 237ms/step - loss: 0.2121 - accuracy: 0.9227 - val_loss: 5.4429 - val_accuracy: 0.1410
    Epoch 26/200
    15/15 [==============================] - 3s 235ms/step - loss: 0.2052 - accuracy: 0.9238 - val_loss: 4.9381 - val_accuracy: 0.1575
    Epoch 27/200
    15/15 [==============================] - 3s 236ms/step - loss: 0.1982 - accuracy: 0.9275 - val_loss: 4.0295 - val_accuracy: 0.1421
    Epoch 28/200
    15/15 [==============================] - 4s 238ms/step - loss: 0.1893 - accuracy: 0.9312 - val_loss: 3.0618 - val_accuracy: 0.2571
    Epoch 29/200
    15/15 [==============================] - 3s 236ms/step - loss: 0.1733 - accuracy: 0.9371 - val_loss: 2.9136 - val_accuracy: 0.2790
    Epoch 30/200
    15/15 [==============================] - 3s 236ms/step - loss: 0.1771 - accuracy: 0.9347 - val_loss: 3.0424 - val_accuracy: 0.3558
    Epoch 31/200
    15/15 [==============================] - 4s 238ms/step - loss: 0.1591 - accuracy: 0.9423 - val_loss: 2.4224 - val_accuracy: 0.4229
    Epoch 32/200
    15/15 [==============================] - 4s 238ms/step - loss: 0.1657 - accuracy: 0.9390 - val_loss: 2.4353 - val_accuracy: 0.4256
    Epoch 33/200
    15/15 [==============================] - 3s 236ms/step - loss: 0.1502 - accuracy: 0.9450 - val_loss: 2.3348 - val_accuracy: 0.4863
    Epoch 34/200
    15/15 [==============================] - 3s 236ms/step - loss: 0.1417 - accuracy: 0.9491 - val_loss: 1.7696 - val_accuracy: 0.6040
    Epoch 35/200
    15/15 [==============================] - 4s 237ms/step - loss: 0.1330 - accuracy: 0.9506 - val_loss: 2.8136 - val_accuracy: 0.4750
    Epoch 36/200
    15/15 [==============================] - 4s 239ms/step - loss: 0.1306 - accuracy: 0.9517 - val_loss: 2.0110 - val_accuracy: 0.5296
    Epoch 37/200
    15/15 [==============================] - 4s 237ms/step - loss: 0.1247 - accuracy: 0.9553 - val_loss: 1.6681 - val_accuracy: 0.5740
    Epoch 38/200
    15/15 [==============================] - 4s 239ms/step - loss: 0.1166 - accuracy: 0.9561 - val_loss: 0.7485 - val_accuracy: 0.7725
    Epoch 39/200
    15/15 [==============================] - 4s 239ms/step - loss: 0.1120 - accuracy: 0.9604 - val_loss: 1.1749 - val_accuracy: 0.6928
    Epoch 40/200
    15/15 [==============================] - 4s 238ms/step - loss: 0.1039 - accuracy: 0.9618 - val_loss: 0.4995 - val_accuracy: 0.8439
    Epoch 41/200
    15/15 [==============================] - 4s 238ms/step - loss: 0.1003 - accuracy: 0.9641 - val_loss: 0.3744 - val_accuracy: 0.8828
    Epoch 42/200
    15/15 [==============================] - 4s 237ms/step - loss: 0.1000 - accuracy: 0.9636 - val_loss: 0.4801 - val_accuracy: 0.8599
    Epoch 43/200
    15/15 [==============================] - 4s 239ms/step - loss: 0.0922 - accuracy: 0.9670 - val_loss: 0.4138 - val_accuracy: 0.8815
    Epoch 44/200
    15/15 [==============================] - 4s 239ms/step - loss: 0.0920 - accuracy: 0.9655 - val_loss: 0.3977 - val_accuracy: 0.8918
    Epoch 45/200
    15/15 [==============================] - 4s 242ms/step - loss: 0.0827 - accuracy: 0.9707 - val_loss: 0.4957 - val_accuracy: 0.8635
    Epoch 46/200
    15/15 [==============================] - 4s 243ms/step - loss: 0.0828 - accuracy: 0.9701 - val_loss: 0.3649 - val_accuracy: 0.9006
    Epoch 47/200
    15/15 [==============================] - 4s 241ms/step - loss: 0.0743 - accuracy: 0.9733 - val_loss: 0.3708 - val_accuracy: 0.8958
    Epoch 48/200
    15/15 [==============================] - 4s 242ms/step - loss: 0.0744 - accuracy: 0.9733 - val_loss: 0.3575 - val_accuracy: 0.9068
    Epoch 49/200
    15/15 [==============================] - 4s 239ms/step - loss: 0.0736 - accuracy: 0.9732 - val_loss: 0.2856 - val_accuracy: 0.9231
    Epoch 50/200
    15/15 [==============================] - 4s 240ms/step - loss: 0.0706 - accuracy: 0.9742 - val_loss: 0.3444 - val_accuracy: 0.9106
    Epoch 51/200
    15/15 [==============================] - 4s 240ms/step - loss: 0.0709 - accuracy: 0.9743 - val_loss: 0.3657 - val_accuracy: 0.9085
    Epoch 52/200
    15/15 [==============================] - 4s 240ms/step - loss: 0.0614 - accuracy: 0.9781 - val_loss: 0.3272 - val_accuracy: 0.9165
    Epoch 53/200
    15/15 [==============================] - 4s 242ms/step - loss: 0.0566 - accuracy: 0.9799 - val_loss: 0.3233 - val_accuracy: 0.9154
    Epoch 54/200
    15/15 [==============================] - 4s 240ms/step - loss: 0.0550 - accuracy: 0.9789 - val_loss: 0.3202 - val_accuracy: 0.9198
    Epoch 55/200
    15/15 [==============================] - 4s 240ms/step - loss: 0.0553 - accuracy: 0.9802 - val_loss: 0.3416 - val_accuracy: 0.9141
    Epoch 56/200
    15/15 [==============================] - 4s 242ms/step - loss: 0.0525 - accuracy: 0.9819 - val_loss: 0.2719 - val_accuracy: 0.9305
    Epoch 57/200
    15/15 [==============================] - 4s 247ms/step - loss: 0.0431 - accuracy: 0.9844 - val_loss: 0.3547 - val_accuracy: 0.9209
    Epoch 58/200
    15/15 [==============================] - 4s 241ms/step - loss: 0.0421 - accuracy: 0.9846 - val_loss: 0.3354 - val_accuracy: 0.9226
    Epoch 59/200
    15/15 [==============================] - 4s 242ms/step - loss: 0.0390 - accuracy: 0.9859 - val_loss: 0.3361 - val_accuracy: 0.9213
    Epoch 60/200
    15/15 [==============================] - 4s 241ms/step - loss: 0.0409 - accuracy: 0.9852 - val_loss: 0.3083 - val_accuracy: 0.9273
    Epoch 61/200
    15/15 [==============================] - 4s 242ms/step - loss: 0.0355 - accuracy: 0.9875 - val_loss: 0.2951 - val_accuracy: 0.9338
    Epoch 62/200
    15/15 [==============================] - 4s 294ms/step - loss: 0.0325 - accuracy: 0.9881 - val_loss: 0.2945 - val_accuracy: 0.9334
    Epoch 63/200
    15/15 [==============================] - 4s 242ms/step - loss: 0.0339 - accuracy: 0.9877 - val_loss: 0.3244 - val_accuracy: 0.9263
    Epoch 64/200
    15/15 [==============================] - 4s 239ms/step - loss: 0.0412 - accuracy: 0.9845 - val_loss: 0.3958 - val_accuracy: 0.9151
    Epoch 65/200
    15/15 [==============================] - 4s 238ms/step - loss: 0.0315 - accuracy: 0.9887 - val_loss: 0.3216 - val_accuracy: 0.9271
    Epoch 66/200
    15/15 [==============================] - 4s 239ms/step - loss: 0.0300 - accuracy: 0.9889 - val_loss: 0.3757 - val_accuracy: 0.9258
    Epoch 67/200
    15/15 [==============================] - 4s 239ms/step - loss: 0.0296 - accuracy: 0.9892 - val_loss: 0.4544 - val_accuracy: 0.9098
    Epoch 68/200
    15/15 [==============================] - 4s 240ms/step - loss: 0.0288 - accuracy: 0.9894 - val_loss: 0.3595 - val_accuracy: 0.9214
    Epoch 69/200
    15/15 [==============================] - 4s 240ms/step - loss: 0.0275 - accuracy: 0.9897 - val_loss: 0.4206 - val_accuracy: 0.9228
    Epoch 70/200
    15/15 [==============================] - 4s 243ms/step - loss: 0.0290 - accuracy: 0.9898 - val_loss: 0.3717 - val_accuracy: 0.9264
    Epoch 71/200
    15/15 [==============================] - 4s 240ms/step - loss: 0.0283 - accuracy: 0.9899 - val_loss: 0.3936 - val_accuracy: 0.9235
    Epoch 72/200
    15/15 [==============================] - 4s 239ms/step - loss: 0.0263 - accuracy: 0.9910 - val_loss: 0.3186 - val_accuracy: 0.9341
    Epoch 73/200
    15/15 [==============================] - 4s 238ms/step - loss: 0.0277 - accuracy: 0.9893 - val_loss: 0.3549 - val_accuracy: 0.9255
    Epoch 74/200
    15/15 [==============================] - 4s 240ms/step - loss: 0.0264 - accuracy: 0.9901 - val_loss: 0.3518 - val_accuracy: 0.9318
    Epoch 75/200
    15/15 [==============================] - 4s 239ms/step - loss: 0.0234 - accuracy: 0.9918 - val_loss: 0.3503 - val_accuracy: 0.9313
    Epoch 76/200
    15/15 [==============================] - 4s 240ms/step - loss: 0.0204 - accuracy: 0.9930 - val_loss: 0.3877 - val_accuracy: 0.9285
    


```python
histories['VGG19_128'] = (history, model.evaluate(x_test_1, y_test_1, BATCH_SIZE))
```

    1/1 [==============================] - 5s 5s/step - loss: 0.2482 - accuracy: 0.9345
    

---


```python
for model_name, (model_history, model_evaluation) in histories.items():
    print(model_name)
    visualize_history(model_history)
    print(model_evaluation)
```

    VGG19_32
    


![png](output_35_1.png)



![png](output_35_2.png)


    [0.22895169258117676, 0.937000036239624]
    VGG19_64
    


![png](output_35_4.png)



![png](output_35_5.png)


    [0.26001396775245667, 0.9250000715255737]
    VGG19_128
    


![png](output_35_7.png)



![png](output_35_8.png)


    [0.24820923805236816, 0.9345000386238098]
    

## V. Đánh giá chung

## VI. Tham khảo
