| Layer (type)    | Output Shape | Param # |  
| :-------------- | :-----------------------------: | --------------: |
| up_sampling3d_4(UpSampling3D) | (None, 28, 28, 1) |         0    |     
| conv2d_48 (Conv2D)           | (None, 28, 28, 32) |        320 | 
| batch_normalization_56 (Batch Normalization) | (None, 28, 28, 32) |        112       |
| conv2d_49 (Conv2D)          | (None, 28, 28, 32) |        9248      |
| batch_normalization_57 (Batch Normalization) | (None, 28, 28, 32) |        112       |
| max_pooling2d_16 (MaxPooling) | (None, 14, 14, 32) |        0         |
| dropout_16 (Dropout)        | (None, 14, 14, 32) |        0         |
| conv2d_50 (Conv2D)          | (None, 14, 14, 64) |        18496     |
| batch_normalization_58 (Batch Normalization) | (None, 14, 14, 64) |        56        |
| conv2d_51 (Conv2D)          | (None, 14, 14, 64) |        36928     |
| batch_normalization_59 (Batch Normalization) | (None, 14, 14, 64) |        56        |
| max_pooling2d_17 (MaxPooling) | (None, 7, 7, 64) |          0         |
| dropout_17 (Dropout)        | (None, 7, 7, 64) |          0         |
| conv2d_52 (Conv2D)          | (None, 7, 7, 128) |         73856     |
| batch_normalization_60 (Batch Normalization) | (None, 7, 7, 128) |         28        |
| conv2d_53 (Conv2D)          | (None, 7, 7, 128) |        147584    |
| batch_normalization_61 (Batch Normalization) | (None, 7, 7, 128) |         28        |
| conv2d_54 (Conv2D)          | (None, 7, 7, 128) |         147584    |
| batch_normalization_62 (Batch Normalization) | (None, 7, 7, 128) |         28        |
| conv2d_55 (Conv2D)          | (None, 7, 7, 128) |         147584    |
| batch_normalization_63 (Batch Normalization) | (None, 7, 7, 128) |         28        |
| max_pooling2d_18 (MaxPooling) | (None, 3, 3, 128) |         0         |
| dropout_18 (Dropout)        | (None, 3, 3, 128) |         0         |
| conv2d_56 (Conv2D)          | (None, 3, 3, 256) |         295168    |
| batch_normalization_64 (Batch Normalization) | (None, 3, 3, 256) |         12        |
| conv2d_57 (Conv2D)          | (None, 3, 3, 256) |         590080    |
| batch_normalization_65 (Batch Normalization) | (None, 3, 3, 256) |         12        |
| conv2d_58 (Conv2D)          | (None, 3, 3, 256) |         590080    |
| batch_normalization_66 (Batch Normalization) | (None, 3, 3, 256) |         12        |
| conv2d_59 (Conv2D)          | (None, 3, 3, 256) |         590080    |
|  batch_normalization_67 (Batch Normalization) | (None, 3, 3, 256) |         12        |
| max_pooling2d_19 (MaxPooling) | (None, 1, 1, 256) |         0         |
| dropout_19 (Dropout)        | (None, 1, 1, 256) |         0         |
| flatten_4 (Flatten)         | (None, 256) |               0         |
| dense_12 (Dense)            | (None, 256) |               65792     |
| batch_normalization_68 (Batch Normalization) | (None, 256) |               1024      |
| dense_13 (Dense)            | (None, 256) |               65792     |
| batch_normalization_69 (Batch Normalization) | (None, 256) |               1024      |
| dense_14 (Dense)            | (None, 10)               | 2570     | 
| Total params: || 2,783,706 |
| Trainable params: || 2,782,434 |
| Non-trainable params: || 1,272 |
________________________________________________________________