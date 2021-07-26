import argparse
import os, torch
import numpy as np
import pickle

import src.scripts.utils as utils
from src.models.PointnetModel import ReconstructionModel
from src.models.interfaces import ReconstructionInterface
from torch.optim.lr_scheduler import CyclicLR

if __name__ == "__main__":
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--dataset_path', required=True, help='Path to dataset')
    parser.add_argument("--no-cuda", action="store_false", dest="cuda", help="Force CPU")
    parser.add_argument("--cuda", action="store_true", dest="cuda", help="Force GPU")
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--num_category', default=10, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Number of points')
    parser.add_argument('--decay_rate', type=float, default=0.5, help='decay rate')
    parser.add_argument('--decay_step', type=float, default=20, help='decay step')
    parser.add_argument('--use_normals', action='store_true', default=True, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=True, help='save data offline')
    parser.add_argument('--no_process_data', action='store_false', default=True, dest='process_data',
                        help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=True, help='use uniform sampling')
    parser.add_argument('--template_dir', help="Path to Sphere templates")

    parser.add_argument("--w_normals", type=float, default=0.008)
    parser.add_argument("--w_collision", type=float, default=1e-5)
    parser.add_argument("--sigma_collision", type=float, default=1e-6)
    parser.add_argument("--w_planar", type=float, default=2)
    parser.add_argument("--w_templatenormals", type=float, default=1e-4)
    parser.add_argument("--w_symmetry", type=float, default=1)
    parser.add_argument("--n_samples", type=int, default=7000)
    parser.add_argument("--n_views", type=int, default=4)
    parser.add_argument('--seperate-turbines',
                        dest='seperate_turbines', action='store_true')
    parser.add_argument('--no-seperate-turbines',
                        dest='seperate_turbines', action='store_false')
    parser.add_argument('--wheels', dest='wheels', action='store_true')
    parser.add_argument('--no-wheels', dest='wheels', action='store_false')
    parser.add_argument('--p2m', dest='p2m', action='store_true')
    parser.add_argument('--no-p2m', dest='p2m', action='store_false')
    parser.add_argument('--symmetries', dest='symmetries', action='store_true')
    parser.add_argument('--no-symmetries',
                        dest='symmetries', action='store_false')
    parser.set_defaults(seperate_turbines=False, wheels=False, p2m=False,
                        symmetries=False, num_worker_threads=8)

    parser.add_argument('--encoder', type=str, default="pointnet")
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.set_defaults(optimizer=None, scheduler=None, file_type='obj')

    args = parser.parse_args()

    template_parameters = utils.load_template_parameters(args)
    template_used = os.path.basename(args.template_dir)
    print(f'../../dataset/{template_used}_params.pkl')
    with open(f'../../dataset/{template_used}_params.pkl', 'rb') as f:
        template_parameters = pickle.load(f)

    model = ReconstructionModel(args, len(template_parameters['initial_parameters']),
                                init=template_parameters['initial_parameters'])
    model_dict = model.state_dict()
    model_dir = os.path.join("../../ModelCheckpoints",
                             'pretrained_modelnet40_1024_lr0.0001_batch16__sphere24_nocollision_cycliclrsheduler')
    checkpoint_path = os.path.join(model_dir, "best_model.pth")
    output_path = os.path.join(model_dir, f'reconstructed_patches_{args.file_type}')
    if args.file_type=='pts':
        extension = 'pts'
    else:
        extension='obj'
    os.makedirs(output_path, exist_ok=True)

    model = utils.load_pretrained(model, checkpoint_path)
    interface = ReconstructionInterface(model, args, template_parameters["vertex_idxs"],
                                        template_parameters["face_idxs"],
                                        template_parameters["junctions"],
                                        template_parameters["edge_data"],
                                        template_parameters["vertex_t"],
                                        template_parameters["adjacencies"],
                                        template_parameters["junction_order"],
                                        template_parameters["template_normals"],
                                        template_parameters["symmetries"])

    model.eval()

    filename = f"/Users/mo/Library/Mobile Documents/com~apple~CloudDocs/Surrey Essentials/Study/Learning Critical Edge Sets from 3D Shapes/Code.nosync/PointCloudToPatches/ignore_dir/02958343_shape.pts"

    shape_array = np.loadtxt(filename).astype(np.float32)
    output_file = os.path.join(f'/Users/mo/Library/Mobile Documents/com~apple~CloudDocs/Surrey Essentials/Study/Learning Critical Edge Sets from 3D Shapes/Code.nosync/PointCloudToPatches/ignore_dir',f'model_shapenet.{extension}')
    print(shape_array.shape)
    with torch.no_grad():
        utils.shape_reconstruct_file(model, template_parameters, shape_array, output_file, type=args.file_type)
