import argparse
import os
import pickle
import torch

import src.scripts.utils as utils
from src.models.PointnetModel import ReconstructionModel

#python src/scripts/reconstruct.py --pc_file {} --output_file
if __name__ == "__main__":
    parser = argparse.ArgumentParser('python src/scripts/reconstruct')
    parser.add_argument('--pc_file', type=str, required=True, help="Input Point Cloud File. The delimiter is \s which means the file should be formatted 'x y z'")
    parser.add_argument('--output_file', type=str, help="Output Sketch Path", default=None)
    parser.add_argument('--file_type', type=str, choices = ['pts', 'obj'], help='if pts generates sketches composed of point clouds. If obj generates Coons patches')
    parser.add_argument('--num_point', type=int, default=2048, help='Number of points')
    parser.add_argument('--use_uniform_sample', action='store_true', default=True, help='use uniform sampling')
    parser.add_argument('--template_dir', help="Path to Sphere templates")

    parser.set_defaults(num_worker_threads=8)

    parser.add_argument('--encoder', type=str, default="pointnet")

    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_dir', type=str,
                        help="Directory of where best_model.pth file is stored for the model inside"
                             " ModelCheckpoints directory")
    # parser.set_defaults(file_type='pts',
    #                    model_name='pretrained_modelnet40_2048_lr0.009_batch32_decay0.9_patience8_sphere24_ReduceOnPlateauScheduler_onlychamferplanar')


    args = parser.parse_args()
    template_used = os.path.basename(args.template_dir)
    template_dir = os.path.split(args.template_dir)[0]

    if args.output_file is None:
        base_name = os.path.basename(args.pc_file).split('.')[0]
        args.output_file = f"{base_name}_{args.encoder}_{template_used}.{args.file_type}"
    try:
        with open(f'{template_dir}/{template_used}_params.pkl', 'rb') as f:
            template_parameters = pickle.load(f)
    except:
        template_parameters = utils.load_template_parameters(args.template_dir)

    model = ReconstructionModel(args, len(template_parameters['initial_parameters']),
                                init=template_parameters['initial_parameters'])
    model.eval()

    # model_dir = os.path.join("/Users/mo/Library/Mobile Documents/com~apple~CloudDocs/Surrey Essentials/Study/Learning Critical Edge Sets from 3D Shapes/Code.nosync/PointCloudToPatches/ModelCheckpoints", args.model_name)

    checkpoint_path = os.path.join(args.model_dir, "best_model.pth")

    model = utils.load_pretrained(model, checkpoint_path)
    model.eval()
    with torch.no_grad():
        utils.reconstruct_file(model, args.num_point, template_parameters, args.pc_file, args.output_file,
                               args.file_type, delimiter=' ')
