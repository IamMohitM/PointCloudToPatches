import itertools

import numpy as np
import torch
from ttools.training import ModelInterface

from src.scripts import utils, loss_utils, coons


class ReconstructionInterface(ModelInterface):
    def __init__(self, model, args, vertex_idxs, face_idxs, junctions,
                 edge_data, vertex_t, adjacencies, junction_order,
                 template_normals, symmetries=None):
        self.model = model

        self.vertex_idxs = vertex_idxs
        self.face_idxs = face_idxs
        self.junctions = junctions
        self.edge_data = edge_data
        self.vertex_t = vertex_t
        self.junction_order = junction_order
        self.template_normals = template_normals[None]

        self.args = args

        self.n_samples_per_loop_side = int(
            np.ceil(np.sqrt(args.n_samples / face_idxs.shape[0])))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)

        # These represent the order of control points on the curve
        self.edge_idxs = [[0, 1, 2, 3], [3, 4, 5, 6],
                          [6, 7, 8, 9], [9, 10, 11, 0]]

        self.nonadjacent_patch_pairs = []
        self.adjacent_patch_pairs = []
        for i1, i2 in list(
                itertools.combinations(range(face_idxs.shape[0]), 2)):
            if args.wheels and (
                    i1 in [5, 11, 17, 23] or i2 in [5, 11, 17, 23]):
                continue
            if i2 in adjacencies[i1]:
                e1 = adjacencies[i1][i2]
                self.adjacent_patch_pairs.append((i1, i2, e1))
            else:
                self.nonadjacent_patch_pairs.append((i1, i2))

        self.d_points_to_tris = loss_utils.PointToTriangleDistance.apply

        p_edge0, p_edge1, p_edge2, p_edge3 = [], [], [], []
        for i in range(self.n_samples_per_loop_side):
            for j in range(self.n_samples_per_loop_side):
                n = self.n_samples_per_loop_side

                if i > 0:
                    p_edge0.append(i + j * self.n_samples_per_loop_side)
                if i < n - 1:
                    p_edge2.append(i + j * self.n_samples_per_loop_side)
                if j > 0:
                    p_edge3.append(i + j * self.n_samples_per_loop_side)
                if j < n - 1:
                    p_edge1.append(i + j * self.n_samples_per_loop_side)
        self.grid_point_edges = (p_edge0, p_edge1, p_edge2, p_edge3)

        self.triangulation = []
        t_edge0, t_edge1, t_edge2, t_edge3 = [], [], [], []
        for i in range(self.n_samples_per_loop_side - 1):
            for j in range(self.n_samples_per_loop_side - 1):
                n = self.n_samples_per_loop_side - 1

                if i > 1:
                    t_edge0.extend(
                        [len(self.triangulation), len(self.triangulation) + 1])
                if i < n - 2:
                    t_edge2.extend(
                        [len(self.triangulation), len(self.triangulation) + 1])
                if j > 1:
                    t_edge3.extend(
                        [len(self.triangulation), len(self.triangulation) + 1])
                if j < n - 2:
                    t_edge1.extend(
                        [len(self.triangulation), len(self.triangulation) + 1])

                self.triangulation.extend(
                    [[i + j * self.n_samples_per_loop_side,
                      i + (j + 1) * self.n_samples_per_loop_side,
                      i + (j + 1) * self.n_samples_per_loop_side + 1],
                     [i + j * self.n_samples_per_loop_side,
                      i + j * self.n_samples_per_loop_side + 1,
                      i + (j + 1) * self.n_samples_per_loop_side + 1]])
        self.triangulation_edges = (t_edge0, t_edge1, t_edge2, t_edge3)

        _loss = Loss(args, self.triangulation_edges, self.triangulation,
                     self.grid_point_edges, self.n_samples_per_loop_side,
                     self.d_points_to_tris, self.edge_idxs,
                     self.nonadjacent_patch_pairs, self.adjacent_patch_pairs,
                     self.template_normals, symmetries)
        self._compute_losses = torch.nn.DataParallel(
            _loss) if args.cuda else _loss

        if args.cuda:
            self.model.cuda()
            self._compute_losses.cuda()

    def forward(self, batch):
        points, target = batch
        # DATA AUGMENTATION
        point = points.data.numpy()
        points = utils.random_point_dropout(point)
        points[:, :, 0:3] = utils.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = utils.shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)
        points = points.transpose(2, 1)

        if self.args.cuda:
            points, target = points.cuda(), target.cuda()

        params = self.model(points)

        vertices, patches = utils.process_patches(
            params, self.vertex_idxs, self.face_idxs, self.edge_data,
            self.junctions, self.junction_order, self.vertex_t)

        st = torch.empty(patches.shape[0], patches.shape[1],
                         self.n_samples_per_loop_side ** 2, 2).uniform_().to(params)

        points = utils.coons_sample(st[..., 0], st[..., 1], patches)
        normals = utils.coons_normals(st[..., 0], st[..., 1], patches)
        mtds = coons.coons_mtds(st[..., 0], st[..., 1], patches)

        return {'patches': patches, 'points': points, 'normals': normals,
                'mtds': mtds, 'st': st, 'params': params, 'vertices': vertices}

    def training_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        losses_dict = self._compute_losses(batch, self.forward(batch))
        loss = losses_dict['loss']
        loss.mean().backward()
        self.optimizer.step()

        return {k: v.mean().item() for k, v in losses_dict.items()}

    def init_validation(self):
        losses = ['loss', 'chamfer_loss', 'normals_loss', 'collision_loss',
                  'planar_loss', 'template_normals_loss', 'symmetry_loss']
        ret = {loss: 0 for loss in losses}
        ret['count'] = 0
        return ret

    def validation_step(self, batch, running_data):
        self.model.eval()
        count = running_data['count']
        n = batch[0].shape[0]
        losses_dict = self._compute_losses(batch, self.forward(batch))
        loss = losses_dict['loss']
        chamfer_loss = losses_dict['chamfer_loss']
        normals_loss = losses_dict['normals_loss']
        collision_loss = losses_dict['collision_loss']
        planar_loss = losses_dict['planar_loss']
        template_normals_loss = losses_dict['template_normals_loss']
        symmetry_loss = losses_dict['symmetry_loss']
        return {
            'loss': (running_data['loss'] * count +
                     loss.mean().item() * n) / (count + n),
            'chamfer_loss': (running_data['chamfer_loss'] * count +
                             chamfer_loss.mean().item() * n) / (count + n),
            'normals_loss': (running_data['normals_loss'] * count +
                             normals_loss.mean().item() * n) / (count + n),
            'collision_loss': (running_data['collision_loss'] * count +
                               collision_loss.mean().item() * n) / (count + n),
            'planar_loss': (running_data['planar_loss'] * count +
                            planar_loss.mean().item() * n) / (count + n),
            'template_normals_loss': (running_data['template_normals_loss'] * count +
                                      template_normals_loss.mean().item() * n) / (count + n),
            'symmetry_loss': (running_data['symmetry_loss'] * count
                              + symmetry_loss.mean().item() * n) / (count + n),
            'count': count + n
        }


wheel_idxs = list(range(24))
no_wheel_idxs = list(range(24, 43))


class Loss(torch.nn.Module):
    def __init__(self, args, triangulation_edges, triangulation,
                 grid_point_edges, n_samples_per_loop_side, d_points_to_tris,
                 edge_idxs, nonadjacent_patch_pairs, adjacent_patch_pairs,
                 template_normals, symmetries):
        super(Loss, self).__init__()
        self.args = args
        self.triangulation_edges = triangulation_edges
        self.triangulation = triangulation
        self.grid_point_edges = grid_point_edges
        self.d_points_to_tris = d_points_to_tris
        self.edge_idxs = edge_idxs
        self.nonadjacent_patch_pairs = nonadjacent_patch_pairs
        self.adjacent_patch_pairs = adjacent_patch_pairs
        self.symmetries = symmetries

        linspace = torch.linspace(0, 1, n_samples_per_loop_side)
        s_grid, t_grid = torch.meshgrid(linspace, linspace)
        self.s_grid = torch.nn.Parameter(s_grid.flatten(), requires_grad=False)
        self.t_grid = torch.nn.Parameter(t_grid.flatten(), requires_grad=False)
        self.template_normals = torch.nn.Parameter(
            template_normals, requires_grad=False)

    def forward(self, batch, fwd_data):
        patches = fwd_data['patches']  # [b, n_patches, 12, 3]
        points = fwd_data['points']
        normals = fwd_data['normals']
        mtds = fwd_data['mtds']
        vertices = fwd_data['vertices']

        target_points = batch[0][:, :, :3].to(points)  # [b, n_points, 3]
        # target_points = batch['points'].to(points)  # [b, n_points, 3]
        target_normals = batch[0][:, :, 3:].to(normals)

        b, n_patches, _, _ = patches.shape

        st = fwd_data['st']

        symmetry_loss = patches.new_zeros(1)

        mtds = mtds.view(b, -1)

        chamferloss_a, chamferloss_b, normalsloss_a, normalsloss_b = \
            loss_utils.compute_chamfer_losses(
                points, normals, target_points, target_normals,
                self.args.w_normals > 0)
        chamferloss_a = torch.sum(mtds * chamferloss_a, dim=-1) / mtds.sum(-1)
        chamferloss_b = chamferloss_b.mean(1)
        chamfer_loss = ((chamferloss_a + chamferloss_b).mean() / 2).view(1)
        normalsloss_a = torch.sum(mtds * normalsloss_a, dim=-1) / mtds.sum(-1)
        normalsloss_b = normalsloss_b.mean(-1)
        normals_loss = ((normalsloss_a + normalsloss_b).mean() / 2).view(1)

        mtds = mtds.view(b, n_patches, -1)

        if self.args.w_templatenormals:
            template_normals_loss = torch.sum(
                (self.template_normals - normals) ** 2, dim=-1)
            template_normals_loss = torch.sum(
                mtds * template_normals_loss, dim=-1) / mtds.sum(-1)
        else:
            template_normals_loss = torch.zeros_like(chamfer_loss)

        del target_normals, normals, target_points

        if self.args.w_planar > 0:
            planar_loss = loss_utils.planar_patch_loss(st, points, mtds)
        else:
            planar_loss = torch.zeros_like(chamfer_loss)

        del points, mtds

        if self.args.w_collision > 0:
            collision_loss = chamfer_loss.new_zeros([b, 0])
            grid_points = utils.coons_sample(self.s_grid, self.t_grid, patches)
            triangles = grid_points[:, :, self.triangulation]

            i1s, i2s, e1s = zip(*self.adjacent_patch_pairs)
            points1 = grid_points[:, i1s]
            point_idxs = torch.tensor([self.grid_point_edges[e]
                                       for e in e1s]).to(points1.device)
            point_idxs = point_idxs[None, :, :, None].expand(b, -1, -1, 3)
            points1 = torch.gather(points1, 2, point_idxs)
            points2 = grid_points[:, i2s]

            triangles1 = triangles[:, i1s]
            triangle_idxs = torch.tensor(
                [self.triangulation_edges[e] for e in e1s]
            ).to(triangles1.device)
            triangle_idxs = triangle_idxs[None, :, :,
                            None, None].expand(b, -1, -1, 3, 3)
            triangles1 = torch.gather(triangles1, 2, triangle_idxs)
            triangles2 = triangles[:, i2s]

            idxs = utils.bboxes_intersect(
                points1, points2, dim=2).any(0).nonzero().squeeze(1)
            n_adjacent_intersections = idxs.shape[0]

            if n_adjacent_intersections > 0:
                points1 = points1[:, idxs].view([-1] + list(points1.shape[2:]))
                points2 = points2[:, idxs].view([-1] + list(points2.shape[2:]))
                triangles1 = triangles1[:, idxs].view(
                    [-1] + list(triangles1.shape[2:]))
                triangles2 = triangles2[:, idxs].view(
                    [-1] + list(triangles2.shape[2:]))
                d1 = self.d_points_to_tris(points1, triangles2)
                d2 = self.d_points_to_tris(points2, triangles1)
                d = torch.min(d1, d2).view(b, -1)
                collision_loss = torch.cat(
                    [collision_loss, torch.exp(-(d / self.args.sigma_collision) ** 2)],
                    dim=1)

            i1s, i2s = zip(*self.nonadjacent_patch_pairs)
            idxs = utils.bboxes_intersect(
                grid_points[:, i1s], grid_points[:, i2s], dim=2
            ).any(0).nonzero().squeeze(1)
            n_nonadjacent_intersections = idxs.shape[0]
            i1s = torch.tensor(i1s).to(grid_points.device)[idxs]
            i2s = torch.tensor(i2s).to(grid_points.device)[idxs]

            if n_nonadjacent_intersections > 0:
                points1 = grid_points[:, i1s].view(
                    [-1] + list(grid_points.shape[2:]))
                points2 = grid_points[:, i2s].view(
                    [-1] + list(grid_points.shape[2:]))
                triangles1 = triangles[:, i1s].view(
                    [-1] + list(triangles.shape[2:]))
                triangles2 = triangles[:, i2s].view(
                    [-1] + list(triangles.shape[2:]))
                d1 = self.d_points_to_tris(points1, triangles2)
                d2 = self.d_points_to_tris(points2, triangles1)
                d = torch.min(d1, d2).view(b, -1)
                collision_loss = torch.cat(
                    [collision_loss, torch.exp(-(d / self.args.sigma_collision) ** 2)],
                    dim=1)

            del triangles

            if n_adjacent_intersections + n_nonadjacent_intersections > 0:
                collision_loss = collision_loss.sum(-1).mean()
            else:
                collision_loss = torch.zeros_like(chamfer_loss)
        else:
            collision_loss = torch.zeros_like(chamfer_loss)

        loss = chamfer_loss + self.args.w_normals * normals_loss + \
               self.args.w_collision * collision_loss + \
               self.args.w_planar * planar_loss + \
               self.args.w_templatenormals * template_normals_loss + \
               self.args.w_symmetry * symmetry_loss

        loss_dict = {
            'loss': loss,
            'chamfer_loss': chamfer_loss,
            'normals_loss': normals_loss,
            'collision_loss': collision_loss,
            'planar_loss': planar_loss,
            'template_normals_loss': template_normals_loss,
            'symmetry_loss': symmetry_loss,
        }
        return loss_dict
