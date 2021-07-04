if __name__ == "__main__":
    with open("/Users/mo/Desktop/test2.obj", 'w') as f:
         with open("/Users/mo/Library/Mobile Documents/com~apple~CloudDocs/Surrey Essentials/Study/Learning Critical Edge Sets from 3D Shapes/Code.nosync/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled/bathtub/bathtub_0001.txt", 'r') as fr:
                 lines = fr.readlines()
                 for l in lines:
                         info = l.strip().split(',')
                         f.write(f"v {info[0]} {info[1]} {info[2]}\n")

                 for l in lines:
                         info = l.strip().split(',')
                         f.write(f"vn {info[3]} {info[4]} {info[5]}\n")