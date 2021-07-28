### general settings ###

#path to dataset root directory 
train_data_root =  '/home/ansible/q474673/datasets/ucf_dataset/rgb'
#path to save the model
trained_model_path = './save_model/ucf101_resnet3D_checkpoint.pth'
#path to save the best model
trained_model_path_best = './save_model/ucf101_resnet3D_best.pth'


# hyper-parameters.
lr = 0.001
lr_decay = 0.1
epochs = 50
weight_decay = 1e-3
train_batch_size = 16
test_batch_size = 16
num_workers = 8
num_classes = 101
