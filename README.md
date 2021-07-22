# Setup
Run the following commands in your terminal

```
git clone https://github.com/IamMohitM/PointCloudToPatches.git
cd PointCloudToPatches
pip install -e .l
```

# Train with ModelNet
```
python src/models/train_model.py --dataset_path {}
--template_dir {template_direcotry}
--cuda
--checkpoint_dir {checkpoint_directory}
--batch_size {batch_size}
--process_data
```
The values in {} must be provided by the user. The dataset and templates are in the datasets folder.

Use arguments --no-cuda to train on CPU. 

## Acknowledgement

This project code is written with a combination of code from the following projects

 - [LearningPatches](https://github.com/dmsm/LearningPatches)
 - [Pointnet_Pointnet2_Pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
 - [DGCNN](https://github.com/WangYueFt/dgcnn)
 