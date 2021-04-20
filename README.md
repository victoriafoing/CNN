# CNN

This repository implements a small version of the VGG16 Network in PyTorch with the following layers:

<img width="636" alt="Screenshot 2021-04-20 at 22 52 53" src="https://user-images.githubusercontent.com/28833172/115468588-33b93000-a22b-11eb-9d8a-a73cc10f1155.png">

The CNN model is trained on the CIFAR-10 dataset, which is an image collection with 10 classes:
![1_OSvbuPLy0PSM2nZ62SbtlQ](https://user-images.githubusercontent.com/28833172/115468927-b2ae6880-a22b-11eb-8ba9-7e3e4442fef8.png)

To train the CNN, run the python script train_convnet_pytorch and specify parameters.

```python train_convnet_pytorch.py --learning_rate 1e-4 --max_steps 5000 --batch_size 32 --eval_freq 500 --data_dir "./cifar10/cifar-10-batches-py"```

