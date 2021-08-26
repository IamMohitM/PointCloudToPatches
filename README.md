# Setup
Run the following commands in your terminal

```
git clone https://github.com/IamMohitM/PointCloudToPatches.git
cd PointCloudToPatches
pip install -e .
```

# Train
```

python src/models/train_model.py --encoder PointNet
 --batch_size 2
 --no_cuda
 --dataset_path 'dataset/modelnet40_normal_resampled'
 --checkpoint_dir 'checkpoints'
 --log_dir_suffix 'test'
 --template_dir 'dataset/templates/sphere24'

```
The above will make a checkpoint directory 'checkpoints' where checkpoints are saved and a 'checkpoints/summaries' which
 contains training and validation logs for tensorboard
 

One can explore the arguments with `python src/models/train_model.py --help`
 
 
# Generate Sketches

```
python src/scripts/reconstruct.py --pc_file "input.pts"
--output_file testing.pts
--file_type pts
--model_dir checkpoints
--template_dir dataset/templates/sphere24
```

reconstructs sketches from the input point cloud file

## Acknowledgement

This project code is written with a combination of code from the following projects

 - [LearningPatches](https://github.com/dmsm/LearningPatches)
 - [Pointnet_Pointnet2_Pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
 - [DGCNN](https://github.com/WangYueFt/dgcnn)
 