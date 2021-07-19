import argparse, torch
import ast

import numpy as np
import os

from collections import defaultdict
from src.data.dataset_prep import ModelNetDataLoader
from src.models.PointnetModel import ReconstructionModel
from src.models.interfaces import ReconstructionInterface
import src.scripts.utils as utils
import src.scripts.coons as coons
from src.data.dataset_prep import farthest_point_sample, pc_normalize


def read_pc_file(filename):
    point_set = np.loadtxt(filename, delimiter=',').astype(np.float32)
    return point_set


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

    args = parser.parse_args()

    test_dataset = ModelNetDataLoader(args.dataset_path, args, split='test', process_data=True, )
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=4)

    # for i, batch in enumerate(test_data_loader):
    #     print(batch[0].shape)
    #     break
    #     losses_dict = interface._compute_losses(batch, interface.forward(batch))
    #     loss = losses_dict['loss'].mean()
    #     chamfer_loss = losses_dict['chamfer_loss'].mean()
    #     print(f'Batch {i} - {loss}, {chamfer_loss}')
    #     loss_list.append(loss)
    #
    # print(f"Average Loss {torch.mean(loss_list)}")
    #

    line_data = [line.strip().split(' ') for line in open(
        os.path.join(args.template_dir, 'edges.txt'), 'r')]
    line_data = [(int(a), int(b), int(c), int(d)) for a, b, c, d in line_data]
    junction_order = [int(line.strip()) for line in open(
        os.path.join(args.template_dir, 'junction_order.txt'), 'r')]
    topology = ast.literal_eval(
        open(os.path.join(args.template_dir, 'topology.txt'), 'r').read())
    adjacencies = {}
    for i, l in enumerate(
            open(os.path.join(args.template_dir, 'adjacencies.txt'), 'r')):
        adj = {}
        for x in l.strip().split(','):
            if x != '':
                j, edge = x.strip().split(' edge ')
                adj[int(j)] = int(edge)
        adjacencies[i] = adj

    vertex_t = {}
    init_params = []
    junctions = {}
    vertex_idxs = np.zeros(
        [len(open(os.path.join(
            args.template_dir, 'vertices.txt'), 'r').readlines()), 3],
        dtype=np.int64)
    processed_vertices = []
    for i, l in enumerate(
            open(os.path.join(args.template_dir, 'vertices.txt'), 'r')):
        value = l.strip().split(' ')
        if value[0] == 'Junction':
            _, v0, v1, v2, v3, t_init = value  # what's the t_init value
            vertex_t[i] = len(init_params)
            init_params.append(utils.logit(float(t_init)))  # Why do the logit?
            junctions[i] = (int(v0), int(v1), int(v2), int(v3))
        elif value[0] == 'RegularVertex':
            _, a, b, c = value
            # Why do this?
            vertex_idxs[i] = [len(init_params), len(
                init_params) + 1, len(init_params) + 2]
            init_params.extend([float(a), float(b), float(c)])
            processed_vertices.append(i)

    edge_data_ = defaultdict(list)
    processed_edges = []
    for i in junction_order:
        processed_vertices.append(i)
        for a, b, c, d in line_data:
            # checking if a or d are regular vertex -
            if a in processed_vertices and \
                    d in processed_vertices and \
                    (a, b, c, d) not in processed_edges:
                edge_data_[i].append((a, b, c, d))
                processed_edges.append((a, b, c, d))
    edge_data = edge_data_

    face_idxs = np.empty([len(topology), 12])
    for i, patch in enumerate(topology):
        for j, k in enumerate(patch):
            face_idxs[i, j] = k
    face_idxs = torch.from_numpy(face_idxs.astype(np.int64))

    init_params = torch.tensor(init_params).squeeze()
    init_patches = utils.process_patches(
        init_params[None], vertex_idxs, face_idxs, edge_data, junctions,
        junction_order, vertex_t)[1][0]
    st = torch.empty(init_patches.shape[0], 1, 2).fill_(0.5).to(init_params)
    template_normals = coons.coons_normals(
        st[..., 0], st[..., 1], init_patches)
    symmetries = None

    model = ReconstructionModel(len(init_params),
                                init=init_params)
    model_dict = model.state_dict()
    checkpoint_dict = torch.load("../../ModelCheckpoints/best_val_loss_8_sphere54_1024_54_pretrained_0.9_8.pth",
                                 map_location='cpu')
    for k, v in checkpoint_dict['model_state_dict'].items():
        if k in model_dict.keys():
            model_dict[k] = v
    model.load_state_dict(model_dict)

    interface = ReconstructionInterface(model, args, vertex_idxs, face_idxs, junctions, edge_data, vertex_t,
        adjacencies, junction_order, template_normals, symmetries)

    # loss_list = []
    model.eval()


    with torch.no_grad():
        category = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]
        for cat in category:
            points = read_pc_file(f"/Users/mo/Library/Mobile Documents/com~apple~CloudDocs/Surrey Essentials/Study/Learning Critical Edge Sets from 3D Shapes/Code.nosync/PointCloudToPatches/dataset/modelnet40_normal_resampled/{cat}/{cat}_0111.txt")
            points = farthest_point_sample(points, 1024)
            points[:, 0:3] = pc_normalize(points[:, 0:3])
            print(points[None].shape)
            points = torch.from_numpy(points)
            points = points.transpose(1,0)
            params = model(points[None])

            _, patches = utils.process_patches(
                params, vertex_idxs, face_idxs, edge_data,
                junctions, junction_order, vertex_t)
            patches = patches.squeeze(0)
            utils.write_curves(f"/Users/mo/Library/Mobile Documents/com~apple~CloudDocs/Surrey Essentials/Study/Learning Critical Edge Sets from 3D Shapes/Code.nosync/PointCloudToPatches/ignore_dir/reconstrcuted_patches/vertices_{cat}_test.obj", patches)


