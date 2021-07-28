import ast
import datetime
import os
from collections import defaultdict

import numpy as np
import torch
import ttools
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset_prep import ModelNetDataLoader, farthest_point_sample, pc_normalize
from src.scripts.coons import coons_normals


def dot(a, b):
    """Dot product."""
    return torch.sum(a * b, dim=-1, keepdim=True)


def dot2(a):
    """Squared norm."""
    return dot(a, a)


def logit(x):
    """Inverse of softmax."""
    return np.log(x / (1 - x))


def batched_cdist_l2(x1, x2):
    """Compute batched l2 cdist."""
    x1_norm = x1.pow(2).sum(-1, keepdim=True)
    x2_norm = x2.pow(2).sum(-1, keepdim=True)
    # th.baddbmm Performs a batch matrix-matrix product of matrices in batch1 and batch2.
    # input is added to the final result.
    res = torch.baddbmm(
        x2_norm.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm).clamp_min_(1e-10).sqrt_()
    return res


def bboxes_intersect(points1, points2, dim=1):
    """Compute whether bounding boxes of two point clouds intersect."""
    min1 = points1.min(dim)[0]
    max1 = points1.max(dim)[0]
    min2 = points2.min(dim)[0]
    max2 = points2.max(dim)[0]
    center1 = (min1 + max1) / 2
    center2 = (min2 + max2) / 2
    size1 = max1 - min1
    size2 = max2 - min2
    return ((center1 - center2).abs() * 2 <= size1 + size2).all(-1)


def sub_bezier(t1, t2, params):
    """Compute control points for cubic Bezier curve between t1 and t2.

    t1 -- [batch_size]
    t2 -- [batch_size]
    params -- [batch_size, 4, 3]
    """

    def dB_dt(t):
        return params[:, 0] * (-3 * (1 - t) ** 2) + params[:, 1] * (3 * (1 - 4 * t + 3 * t ** 2)) \
               + params[:, 2] * (3 * (2 * t - 3 * t ** 2)) + params[:, 3] * (3 * t ** 2)

    t1 = t1[:, None]
    t2 = t2[:, None]
    sub_pts = torch.empty_like(params)
    sub_pts[:, 0] = bezier_sample(t1[:, :, None], params).squeeze(1)
    sub_pts[:, 3] = bezier_sample(t2[:, :, None], params).squeeze(1)
    sub_pts[:, 1] = (t2 - t1) * dB_dt(t1) / 3 + sub_pts[:, 0]
    sub_pts[:, 2] = sub_pts[:, 3] - (t2 - t1) * dB_dt(t2) / 3
    return sub_pts


def bezier_sample(t, params):
    """Sample points from cubic Bezier curves defined by params at t values."""
    A = params.new_tensor([[1, 0, 0, 0],
                           [-3, 3, 0, 0],
                           [3, -6, 3, 0],
                           [-1, 3, -3, 1]])

    t = t.pow(t.new_tensor([0, 1, 2, 3]))  # [n_samples, 4]

    # @ operator is equivalent to matmul
    points = t @ A @ params  # [..., n_samples, 3]
    return points


def coons_sample(s, t, params):
    """Sample points from Coons patch defined by params at s, t values.

    params -- [..., 12, 3]
    """
    sides = [params[..., :4, :], params[..., 3:7, :],
             params[..., 6:10, :], params[..., [9, 10, 11, 0], :]]
    corners = [params[..., [0], :], params[..., [3], :],
               params[..., [9], :], params[..., [6], :]]
    s = s[..., None]
    t = t[..., None]
    # bilinear interpolation
    B = corners[0] * (1 - s) * (1 - t) + corners[1] * s * (1 - t) + \
        corners[2] * (1 - s) * t + corners[3] * s * t  # [..., n_samples, 3]
    Lc = bezier_sample(s, sides[0]) * (1 - t) + bezier_sample(1 - s, sides[2]) * t  # linear interpolation
    Ld = bezier_sample(t, sides[1]) * s + bezier_sample(1 - t, sides[3]) * (1 - s)  # linear interpolation
    return Lc + Ld - B  # final patch


def extract_curves(params, n_points=50):
    """
    :param params: Patch Parameters
    :return: a tensor
    """
    s = torch.linspace(0, 1, n_points)
    sides = [params[..., :4, :], params[..., 3:7, :],
             params[..., 6:10, :], params[..., [9, 10, 11, 0], :]]

    curve_1 = bezier_sample(s[..., None], sides[0])
    curve_2 = bezier_sample(s[..., None], sides[1])
    curve_3 = bezier_sample(s[..., None], sides[2])
    curve_4 = bezier_sample(s[..., None], sides[3])

    return torch.stack((curve_1, curve_2, curve_3, curve_4), dim=1)


def process_patches(params, vertex_idxs, face_idxs, edge_data, junctions,
                    junction_order, vertex_t):
    """Process all junction curves to compute explicit patch control points."""
    vertices = params.clone()[:, vertex_idxs]

    for i in junction_order:
        edge = junctions[i]
        t = torch.sigmoid(params[:, vertex_t[i]])
        vertex = bezier_sample(t[:, None, None], vertices[:, edge]).squeeze(1)
        # vertices = vertices.clone()
        vertices[:, i] = vertex

        for a, b, c, d in edge_data[i]:
            if a not in junctions:  # if a is not, the d is def the junction (?)
                a, b, c, d = d, c, b, a

            edge = junctions[a]
            t_a = torch.sigmoid(params[:, vertex_t[a]])
            v0_a, _, _, v3_a = edge
            if d == v0_a:
                t_d = torch.zeros_like(t_a)
            elif d == v3_a:
                t_d = torch.ones_like(t_a)
            else:
                v0_d, _, _, v3_d = junctions[d]
                t_d = torch.sigmoid(params[:, vertex_t[d]])
                if v0_a == v0_d and v3_a == v3_d:
                    pass
                elif v0_a == v3_d and v3_a == v0_d:
                    t_d = 1 - t_d
                else:
                    edge = junctions[d]
                    if a == v0_d:
                        t_a = torch.zeros_like(t_d)
                    elif a == v3_d:
                        t_a = torch.ones_like(t_d)

            curve = sub_bezier(t_a, t_d, vertices[:, edge])[:, 1:-1]
            # vertices = vertices.clone()
            vertices[:, [b, c]] = curve

    patches = vertices[:, face_idxs]

    return vertices, patches


def load_modelnet(args):
    """
    Loads ModelNet Dataset
    :param args: Arguments with expected attributes - dataset_path, batch_size, process_data, num_worker_threads,
    use_normals, use_uniform_sample, num_point, num_category
    :return: torch Dataloader of training, validation and test dataset
    """
    train_dataset = ModelNetDataLoader(root=args.dataset_path, args=args, split='train', process_data=args.process_data)
    val_dataset = ModelNetDataLoader(root=args.dataset_path, args=args, split='val', process_data=args.process_data)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                    num_workers=args.num_worker_threads, drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_worker_threads)
    test_dataset = ModelNetDataLoader(root=args.dataset_path, args=args, split='test', process_data=args.process_data)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_worker_threads, drop_last=True)
    return train_data_loader, val_data_loader, test_data_loader


def load_template_parameters(args):
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
            init_params.append(logit(float(t_init)))  # Why do the logit?
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
    init_patches = process_patches(
        init_params[None], vertex_idxs, face_idxs, edge_data, junctions,
        junction_order, vertex_t)[1][0]
    st = torch.empty(init_patches.shape[0], 1, 2).fill_(0.5).to(init_params)
    template_normals = coons_normals(
        st[..., 0], st[..., 1], init_patches)
    symmetries = None

    template_params = {
        "vertex_idxs": vertex_idxs,
        "face_idxs": face_idxs,
        "junctions": junctions,
        "edge_data": edge_data,
        "vertex_t": vertex_t,
        "adjacencies": adjacencies,
        "junction_order": junction_order,
        "template_normals": template_normals,
        "symmetries": symmetries,
        "initial_patches": init_patches,
        "initial_parameters": init_params
    }

    return template_params


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
    return batch_pc


def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def write_points(patches, file):
    patch_vertices = extract_curves(patches, n_points=500).cpu().numpy()
    vertex_index = 1
    with open(file, 'w') as f:
        for patch in patch_vertices:
            for curves in patch:
                for x, y, z in curves:
                    f.write(f'{x} {y} {z}\n')


def write_curves(patches, file):
    patch_vertices = extract_curves(patches).cpu().numpy()
    line_list = []
    face_list = []
    vertex_index = 1
    with open(file, 'w') as f:
        for patch in patch_vertices:
            start_face_index = vertex_index
            for curves in patch:
                start_line_index = vertex_index
                for x, y, z in curves:
                    f.write(f'v {x} {y} {z}\n')
                    vertex_index += 1
                line_list.append(torch.arange(start_line_index, vertex_index))
            face_list.append(torch.arange(start_face_index, vertex_index))

        for line in line_list:
            f.write('l')
            for v in line:
                f.write(f' {v}')
            f.write('\n')

        for face in face_list:
            f.write('f')
            for v in face:
                f.write(f' {v}')
            f.write('\n')


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


def load_pretrained(model, checkpoint_path):
    if torch.cuda.is_available():
        device = None
    else:
        device = 'cpu'

    checkpoint_dict = torch.load(checkpoint_path, map_location=device)
    count = 0
    model_dict = model.state_dict()
    for k, v in checkpoint_dict['model_state_dict'].items():
        if k in model_dict.keys():
            count += 1
            model_dict[k] = v
    model.load_state_dict(model_dict)
    print(f'Loaded weights of {count} layers')
    return model


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_loss_on_dataset(interface, test_dataset):
    """

    :param interface: a ttools.training.ModelInterface object.
    :param test_dataset:
    :return:
    """
    with torch.no_grad():
        running_data = interface.init_validation()
        for batch_id, batch in enumerate(test_dataset):
            running_data = interface.validation_step(batch, running_data)
        return running_data


def read_pc_file(filename, delimiter=","):
    """
    Returns a numpy array of points in the filename. Expects ',' as delimeter
    :param filename: File Path
    :return: Numpy Array
    """
    point_set = np.loadtxt(filename, delimiter=delimiter).astype(np.float32)
    return point_set


def reconstruct_file(model, n_samples, template_params, pc_file, curve_path, type='pts', delimiter=','):
    """
    Writes a .obj file with vertex (v), line (l) and face (f) information extracted from output patches of the model
    :param model: Model to use for reconstructing Patches
    :param template_params: a dictionary of template parameters
    :param pc_file: The Point Cloud file to be used as input to the model
    :param curve_path: The output .obj path
    :return: None
    """
    points = read_pc_file(pc_file, delimiter)
    points = farthest_point_sample(points, n_samples)
    points[:, 0:3] = pc_normalize(points[:, 0:3])
    points = torch.from_numpy(points)
    points = points.transpose(1, 0)
    params = model(points[None])

    _, patches = process_patches(
        params, template_params['vertex_idxs'], template_params['face_idxs'], template_params['edge_data'],
        template_params['junctions'], template_params['junction_order'], template_params['vertex_t'])
    patches = patches.squeeze(0)
    if type == 'pts':
        write_points(patches, curve_path)
    else:
        write_curves(patches, curve_path)


def training_setup(args, model, interface_class, trainer_class, template_parameters, optimizer, scheduler,
                   log_directory_name=None):
    """
    Setups the training interface, callbacks, and
    :param args:
    :param model: An nn.Module type object
    :param interface_class: a ttools.training.ModelInterface class
    :param trainer_class:
    :param template_parameters: a dict - expected keys - vertex_idxs, face_idxs,
     junctions, edge_data, vertex_t, adjacencies, junction_order, template_normals, symmetries
    :param optimizer: torch.optim.Optimizer type
    :param scheduler: torch.optim.lr_scheduler._LRScheduler type
    :return:
    """

    args.optimizer = optimizer
    args.scheduler = scheduler

    interface = interface_class(model, args, template_parameters["vertex_idxs"],
                                template_parameters["face_idxs"],
                                template_parameters["junctions"],
                                template_parameters["edge_data"],
                                template_parameters["vertex_t"],
                                template_parameters["adjacencies"],
                                template_parameters["junction_order"],
                                template_parameters["template_normals"],
                                template_parameters["symmetries"])

    starting_epoch = 0
    check_pointer = ttools.Checkpointer(
        args.checkpoint_dir, model=model, optimizers=interface.optimizer)

    keys = ['loss', 'chamfer_loss', 'normals_loss', 'collision_loss',
            'planar_loss', 'template_normals_loss', 'learning_rate']

    if log_directory_name:
        train_log = f'training_log_{log_directory_name}'
        val_log = f'Validation_log_{log_directory_name}'
    else:
        train_log = datetime.datetime.now().strftime(
            'training_log_%d-%B-%y_%H-%M-%S')
        val_log = datetime.datetime.now().strftime(
            'validation_log_%d-%B-%y_%H-%M-%S')
    writer = SummaryWriter(
        os.path.join(args.checkpoint_dir, 'summaries',
                     train_log), flush_secs=1)
    val_writer = SummaryWriter(
        os.path.join(args.checkpoint_dir, 'summaries',
                     val_log),
        flush_secs=1)

    trainer = trainer_class(interface)
    trainer.add_callback(
        ttools.callbacks.TensorBoardLoggingCallback(keys=keys, writer=writer,

                                                    val_writer=val_writer,
                                                    frequency=3))
    trainer.add_callback(ttools.callbacks.ProgressBarCallback(keys=keys))
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(
        check_pointer, max_files=1, max_epochs=2))

    return trainer


def shape_reconstruct_file(model, template_params, points, curve_path, type='pts'):
    """
    Writes a .obj file with vertex (v), line (l) and face (f) information extracted from output patches of the model
    :param model: Model to use for reconstructing Patches
    :param template_params: a dictionary of template parameters
    :param pc_file: The Point Cloud file to be used as input to the model
    :param curve_path: The output .obj path
    :return: None
    """
    points = farthest_point_sample(points, 1024)
    points[:, 0:3] = pc_normalize(points[:, 0:3])
    points = torch.from_numpy(points)
    points = points.transpose(1, 0)

    params = model(points[None])

    _, patches = process_patches(
        params, template_params['vertex_idxs'], template_params['face_idxs'], template_params['edge_data'],
        template_params['junctions'], template_params['junction_order'], template_params['vertex_t'])
    patches = patches.squeeze(0)
    if type == 'pts':
        write_points(patches, curve_path)
    else:
        write_curves(patches, curve_path)
