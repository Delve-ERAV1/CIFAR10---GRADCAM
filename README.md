# S11

# Training ResNet18 on CIFAR10 - GRADCAM

This assignment entails implementing grad-cam on a ResNet18 deep learning model on the CIFAR10 dataset and visualizing the results. We are going to follow the same code structure as in the reference repo: [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).

## Project Structure

The project is divided into several parts:

1. **Preparing the Data**
2. **Model Creation**
3. **Training and Test Loops**
4. **Visualization of Results**

### Preparing the Data

The data used for this project is the CIFAR10 dataset. It is split into training and testing subsets. The batch size is set to 128 for both the train and test loaders. Data augmentation is applied to the training data to improve model generalization. These data transformations are defined in the `utils.py` file.

### Model Creation

The model used for this project is ResNet18. The model definition is located in the `models` directory. The file `resnet.py` contains the ResNet model definitions. 

```
----------Model Summary----------
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 36, 36]           1,728
       BatchNorm2d-2           [-1, 64, 36, 36]             128
            Conv2d-3           [-1, 64, 36, 36]          36,864
       BatchNorm2d-4           [-1, 64, 36, 36]             128
            Conv2d-5           [-1, 64, 36, 36]          36,864
       BatchNorm2d-6           [-1, 64, 36, 36]             128
        BasicBlock-7           [-1, 64, 36, 36]               0
            Conv2d-8           [-1, 64, 36, 36]          36,864
       BatchNorm2d-9           [-1, 64, 36, 36]             128
           Conv2d-10           [-1, 64, 36, 36]          36,864
      BatchNorm2d-11           [-1, 64, 36, 36]             128
       BasicBlock-12           [-1, 64, 36, 36]               0
           Conv2d-13          [-1, 128, 19, 19]          73,728
      BatchNorm2d-14          [-1, 128, 19, 19]             256
           Conv2d-15          [-1, 128, 19, 19]         147,456
      BatchNorm2d-16          [-1, 128, 19, 19]             256
           Conv2d-17          [-1, 128, 19, 19]           8,192
      BatchNorm2d-18          [-1, 128, 19, 19]             256
       BasicBlock-19          [-1, 128, 19, 19]               0
           Conv2d-20          [-1, 128, 19, 19]         147,456
      BatchNorm2d-21          [-1, 128, 19, 19]             256
           Conv2d-22          [-1, 128, 19, 19]         147,456
      BatchNorm2d-23          [-1, 128, 19, 19]             256
       BasicBlock-24          [-1, 128, 19, 19]               0
           Conv2d-25          [-1, 256, 11, 11]         294,912
      BatchNorm2d-26          [-1, 256, 11, 11]             512
           Conv2d-27          [-1, 256, 11, 11]         589,824
      BatchNorm2d-28          [-1, 256, 11, 11]             512
           Conv2d-29          [-1, 256, 11, 11]          32,768
      BatchNorm2d-30          [-1, 256, 11, 11]             512
       BasicBlock-31          [-1, 256, 11, 11]               0
           Conv2d-32          [-1, 256, 11, 11]         589,824
      BatchNorm2d-33          [-1, 256, 11, 11]             512
           Conv2d-34          [-1, 256, 11, 11]         589,824
      BatchNorm2d-35          [-1, 256, 11, 11]             512
       BasicBlock-36          [-1, 256, 11, 11]               0
           Conv2d-37            [-1, 512, 7, 7]       1,179,648
      BatchNorm2d-38            [-1, 512, 7, 7]           1,024
           Conv2d-39            [-1, 512, 7, 7]       2,359,296
      BatchNorm2d-40            [-1, 512, 7, 7]           1,024
           Conv2d-41            [-1, 512, 7, 7]         131,072
      BatchNorm2d-42            [-1, 512, 7, 7]           1,024
       BasicBlock-43            [-1, 512, 7, 7]               0
           Conv2d-44            [-1, 512, 7, 7]       2,359,296
      BatchNorm2d-45            [-1, 512, 7, 7]           1,024
           Conv2d-46            [-1, 512, 7, 7]       2,359,296
      BatchNorm2d-47            [-1, 512, 7, 7]           1,024
       BasicBlock-48            [-1, 512, 7, 7]               0
           Linear-49                   [-1, 10]           5,130
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 16.96
Params size (MB): 42.63
Estimated Total Size (MB): 59.59
----------------------------------------------------------------
```

### Training and Test Loops

The model is trained for 20 epochs. The optimizer used for the training is Adam, and the learning rate scheduler used is OneCycleLR. The training and test loops are defined in the main script. 

```
----------Training Model----------
EPOCH: 0
Loss=1.6533868312835693 Batch_id=390 Accuracy=30.16: 100%|██████████| 391/391 [01:13<00:00,  5.35it/s]

Test set: Average loss: 0.0116, Accuracy: 4546/10000 (45.46%)

EPOCH: 1
Loss=1.6056445837020874 Batch_id=390 Accuracy=41.27: 100%|██████████| 391/391 [01:12<00:00,  5.37it/s]

Test set: Average loss: 0.0104, Accuracy: 5381/10000 (53.81%)

EPOCH: 2
Loss=1.3377693891525269 Batch_id=390 Accuracy=46.82: 100%|██████████| 391/391 [01:12<00:00,  5.37it/s]

Test set: Average loss: 0.0079, Accuracy: 6353/10000 (63.53%)

EPOCH: 3
Loss=1.367601752281189 Batch_id=390 Accuracy=50.67: 100%|██████████| 391/391 [01:12<00:00,  5.38it/s]

Test set: Average loss: 0.0077, Accuracy: 6510/10000 (65.10%)

EPOCH: 4
Loss=1.1784348487854004 Batch_id=390 Accuracy=53.88: 100%|██████████| 391/391 [01:12<00:00,  5.38it/s]

Test set: Average loss: 0.0086, Accuracy: 6517/10000 (65.17%)

EPOCH: 5
Loss=1.4694068431854248 Batch_id=390 Accuracy=56.38: 100%|██████████| 391/391 [01:12<00:00,  5.38it/s]

Test set: Average loss: 0.0068, Accuracy: 6964/10000 (69.64%)

EPOCH: 6
Loss=1.3115588426589966 Batch_id=390 Accuracy=58.52: 100%|██████████| 391/391 [01:12<00:00,  5.39it/s]

Test set: Average loss: 0.0066, Accuracy: 7115/10000 (71.15%)

EPOCH: 7
Loss=1.2128006219863892 Batch_id=390 Accuracy=59.92: 100%|██████████| 391/391 [01:12<00:00,  5.39it/s]

Test set: Average loss: 0.0064, Accuracy: 7259/10000 (72.59%)

EPOCH: 8
Loss=1.099895715713501 Batch_id=390 Accuracy=61.25: 100%|██████████| 391/391 [01:12<00:00,  5.38it/s]

Test set: Average loss: 0.0053, Accuracy: 7676/10000 (76.76%)

EPOCH: 9
Loss=0.7822802066802979 Batch_id=390 Accuracy=63.37: 100%|██████████| 391/391 [01:12<00:00,  5.39it/s]

Test set: Average loss: 0.0058, Accuracy: 7568/10000 (75.68%)

EPOCH: 10
Loss=0.9641550183296204 Batch_id=390 Accuracy=64.03: 100%|██████████| 391/391 [01:12<00:00,  5.38it/s]

Test set: Average loss: 0.0050, Accuracy: 7820/10000 (78.20%)

EPOCH: 11
Loss=1.183558702468872 Batch_id=390 Accuracy=64.81: 100%|██████████| 391/391 [01:12<00:00,  5.38it/s]

Test set: Average loss: 0.0047, Accuracy: 7951/10000 (79.51%)

EPOCH: 12
Loss=1.2491145133972168 Batch_id=390 Accuracy=65.48: 100%|██████████| 391/391 [01:12<00:00,  5.36it/s]

Test set: Average loss: 0.0046, Accuracy: 7963/10000 (79.63%)

EPOCH: 13
Loss=0.9477819204330444 Batch_id=390 Accuracy=66.42: 100%|██████████| 391/391 [01:12<00:00,  5.37it/s]

Test set: Average loss: 0.0054, Accuracy: 7755/10000 (77.55%)

EPOCH: 14
Loss=0.8960431218147278 Batch_id=390 Accuracy=67.17: 100%|██████████| 391/391 [01:12<00:00,  5.36it/s]

Test set: Average loss: 0.0044, Accuracy: 8168/10000 (81.68%)

EPOCH: 15
Loss=1.2009488344192505 Batch_id=390 Accuracy=67.52: 100%|██████████| 391/391 [01:13<00:00,  5.35it/s]

Test set: Average loss: 0.0042, Accuracy: 8146/10000 (81.46%)

EPOCH: 16
Loss=0.9159234762191772 Batch_id=390 Accuracy=68.12: 100%|██████████| 391/391 [01:13<00:00,  5.35it/s]

Test set: Average loss: 0.0048, Accuracy: 7989/10000 (79.89%)

EPOCH: 17
Loss=0.7118523120880127 Batch_id=390 Accuracy=69.03: 100%|██████████| 391/391 [01:12<00:00,  5.36it/s]

Test set: Average loss: 0.0043, Accuracy: 8262/10000 (82.62%)

EPOCH: 18
Loss=0.9092729687690735 Batch_id=390 Accuracy=69.50: 100%|██████████| 391/391 [01:12<00:00,  5.36it/s]

Test set: Average loss: 0.0036, Accuracy: 8431/10000 (84.31%)

EPOCH: 19
Loss=0.8162087202072144 Batch_id=390 Accuracy=69.93: 100%|██████████| 391/391 [01:12<00:00,  5.38it/s]

Test set: Average loss: 0.0048, Accuracy: 8162/10000 (81.62%)
```
### Visualization of Results

The results are visualized by plotting the loss curves for the train and test datasets, displaying a gallery of 10 misclassified images, and showing the GradCAM output on the 10 misclassified images. 

![image](https://github.com/cydal/Pytorch_CIFAR10_gradcam/assets/11761529/cc136c9d-503b-4b85-83b8-59603a258d5e)
![image](https://github.com/cydal/Pytorch_CIFAR10_gradcam/assets/11761529/fbed1879-0fe7-4a78-8804-5ab50f2b717e)
![image](https://github.com/cydal/Pytorch_CIFAR10_gradcam/assets/11761529/baaed97a-b68d-4af3-8ff8-60ebe9e0a92e)
![image](https://github.com/cydal/Pytorch_CIFAR10_gradcam/assets/11761529/8e673de7-d01a-4a93-a6b4-2c326c56b5d3)
![image](https://github.com/cydal/Pytorch_CIFAR10_gradcam/assets/11761529/133bdf98-c81e-46c1-bb3d-430ca01558e1)


## Dependencies

This project requires the following dependencies:

- torch
- torchvision
- numpy
- matplotlib

## Usage

To run this project, clone the repository and run the main script:

```bash
git clone https://github.com/cydal/Pytorch_CIFAR10_gradcam.git
%cd /content/Pytorch_CIFAR10_gradcam
%run main.py
```
