import torch
from src.scripts import utils


def d_points_to_tris(points, triangles):
    """Compute distance frome each point to the corresponding triangle.

    points -- [b, n, 3]
    triangles -- [b, n, 3, 3]
    """
    v21 = triangles[:, :, 1] - triangles[:, :, 0]
    v32 = triangles[:, :, 2] - triangles[:, :, 1]
    v13 = triangles[:, :, 0] - triangles[:, :, 2]
    p1 = points - triangles[:, :, 0]
    p2 = points - triangles[:, :, 1]
    p3 = points - triangles[:, :, 2]
    nor = torch.cross(v21, v13, dim=-1)

    cond = utils.dot(torch.cross(v21, nor, dim=-1), p1).sign() \
           + utils.dot(torch.cross(v32, nor, dim=-1), p2).sign() \
           + utils.dot(torch.cross(v13, nor, dim=-1), p3).sign() < 2
    cond = cond.float()
    result = cond * torch.stack([
        utils.dot2(v21 * torch.clamp(utils.dot(v21, p1) / utils.dot2(v21), 0, 1) - p1),
        utils.dot2(v32 * torch.clamp(utils.dot(v32, p2) / utils.dot2(v32), 0, 1) - p2),
        utils.dot2(v13 * torch.clamp(utils.dot(v13, p3) / utils.dot2(v13), 0, 1) - p3)
    ], dim=-1).min(-1)[0] + (1 - cond) * utils.dot(nor, p1) * utils.dot(nor, p1) / utils.dot2(nor)
    return result.squeeze(-1).min(-1)[0]


def compute_chamfer_losses(points, normals, target_points, target_normals,
                           compute_normals=True):
    """Compute area-weighted Chamfer and normals losses."""
    b = points.shape[0]
    points = points.view(b, -1, 3)

    # [b, n_total_samples, n_points]
    distances = utils.batched_cdist_l2(points, target_points)

    chamferloss_a, idx_a = distances.min(2)  # [b, n_total_samples]
    chamferloss_b, idx_b = distances.min(1)

    if compute_normals:
        normals = normals.view(b, -1, 3)

        # [b, n_total_samples, 1, 3]
        idx_a = idx_a[..., None, None].expand(-1, -1, -1, 3)
        nearest_target_normals = \
            target_normals[:, None].expand(list(distances.shape) + [3]) \
                .gather(index=idx_a, dim=2).squeeze(2)  # [b, n_total_samples, 3]

        # [b, 1, n_points, 3]
        idx_b = idx_b[..., None, :, None].expand(-1, -1, -1, 3)
        nearest_normals = \
            normals[:, :, None].expand(list(distances.shape) + [3]) \
                .gather(index=idx_b, dim=1).squeeze(1)  # [b, n_points, 3]

        normalsloss_a = torch.sum((nearest_target_normals - normals) ** 2, dim=-1)
        normalsloss_b = torch.sum((nearest_normals - target_normals) ** 2, dim=-1)
    else:
        normalsloss_a = torch.zeros_like(chamferloss_a)
        normalsloss_b = torch.zeros_like(chamferloss_b)

    return chamferloss_a, chamferloss_b, normalsloss_a, normalsloss_b


def planar_patch_loss(params, points, mtds):
    """Compute planar patch loss from control points, samples, and Jacobians.

    params -- [..., 2]
    points -- [..., 3]
    """
    X = torch.cat([params.new_ones(list(params.shape[:-1]) + [1]), params],
               dim=-1)
    b = torch.inverse(X.transpose(-1, -2) @ X) @ X.transpose(-1, -2) @ points
    distances = (X @ b - points).pow(2).sum(-1)
    return torch.sum(distances * mtds, dim=-1) / mtds.sum(-1)


class PointToTriangleDistance(torch.autograd.Function):
    """Autograd function for computing smallest point to triangle distance."""

    @staticmethod
    def forward(ctx, points, triangles):
        """Compute smallest distance between each point and triangle batch.

        points -- [batch_size, n_points, 3]
        triangles -- [batch_size, n_triangles, 3, 3]
        """
        b = points.shape[0]

        v21 = triangles[:, None, :, 1] - triangles[:, None, :, 0]
        v32 = triangles[:, None, :, 2] - triangles[:, None, :, 1]
        v13 = triangles[:, None, :, 0] - triangles[:, None, :, 2]
        p1 = points[:, :, None] - triangles[:, None, :, 0]
        p2 = points[:, :, None] - triangles[:, None, :, 1]
        p3 = points[:, :, None] - triangles[:, None, :, 2]
        nor = torch.cross(v21, v13, dim=-1)

        cond = utils.dot(torch.cross(v21, nor, dim=-1), p1).sign() \
               + utils.dot(torch.cross(v32, nor, dim=-1), p2).sign() \
               + utils.dot(torch.cross(v13, nor, dim=-1), p3).sign() < 2
        cond = cond.float()
        result = cond * torch.stack([
            utils.dot2(v21 * torch.clamp(utils.dot(v21, p1) / utils.dot2(v21), 0, 1) - p1),
            utils.dot2(v32 * torch.clamp(utils.dot(v32, p2) / utils.dot2(v32), 0, 1) - p2),
            utils.dot2(v13 * torch.clamp(utils.dot(v13, p3) / utils.dot2(v13), 0, 1) - p3)
        ], dim=-1).min(-1)[0] + (1 - cond) \
                 * utils.dot(nor, p1) * utils.dot(nor, p1) / utils.dot2(nor)
        result = result.squeeze(-1)

        _, nearest_tris_idxs = result.min(-1)  # [b, n_points]
        _, nearest_points_idxs = result.min(-2)  # [b, n_tris]
        ctx.save_for_backward(
            points, triangles, nearest_tris_idxs, nearest_points_idxs)

        return result.view(b, -1).min(-1)[0]

    @staticmethod
    def backward(ctx, grad_output):
        """Only consider the closest point-triangle pair for gradient."""
        points, triangles, nearest_tris_idxs, nearest_points_idxs = \
            ctx.saved_tensors
        grad_points = grad_tris = None

        if ctx.needs_input_grad[0]:
            idx = nearest_tris_idxs[..., None, None].expand(
                list(nearest_tris_idxs.shape) + [3, 3])
            nearest_tris = triangles.gather(index=idx, dim=1)
            with torch.enable_grad():
                distance = d_points_to_tris(points, nearest_tris)
                grad_points = torch.autograd.grad(outputs=distance, inputs=points,
                                                  grad_outputs=grad_output,
                                                  only_inputs=True)[0]
        if ctx.needs_input_grad[1]:
            idx = nearest_points_idxs[..., None].expand(
                list(nearest_points_idxs.shape) + [3])
            nearest_points = points.gather(index=idx, dim=1)
            with torch.enable_grad():
                distance = d_points_to_tris(nearest_points, triangles)
                grad_tris = torch.autograd.grad(outputs=distance,
                                                inputs=triangles,
                                                grad_outputs=grad_output,
                                                only_inputs=True)[0]

        return grad_points, grad_tris
