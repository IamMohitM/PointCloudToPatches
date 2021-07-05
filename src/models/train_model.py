import argparse
import datetime
import os

import torch
import ttools
from torch.utils.tensorboard import SummaryWriter

from src.models.PointnetModel import ReconstructionModel
from src.models.interfaces import ReconstructionInterface
from src.scripts import utils


def train(args):
    train_dataset, val_dataset = utils.load_modelnet(args)
    template_parameters = utils.load_template_parameters(args)
    model = ReconstructionModel(len(template_parameters['initial_parameters']),
                                init=template_parameters['initial_parameters'])
    interface = ReconstructionInterface(model, args, template_parameters["vertex_idxs"],
                                        template_parameters["face_idxs"],
                                        template_parameters["junctions"],
                                        template_parameters["edge_data"],
                                        template_parameters["vertex_t"],
                                        template_parameters["adjacencies"],
                                        template_parameters["junction_order"],
                                        template_parameters["template_normals"],
                                        template_parameters["symmetries"])

    starting_epoch = None

    if os.path.exists(os.path.join(args.checkpoint_dir, 'best_val_loss.pth')):
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'best_val_loss.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        interface.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        interface.best_val_loss = checkpoint['loss']
        interface.step_scheduler.load_state_dict(checkpoint['step_scheduler'])
        starting_epoch = checkpoint['epoch']

        print(
            f'Loading checkpoint and starting training from epoch {starting_epoch} and current learning rate \
            {interface.step_scheduler.get_lr()}')

    check_pointer = ttools.Checkpointer(
        args.checkpoint_dir, model=model, optimizers=interface.optimizer)

    # extras, _ = check_pointer.load_latest()

    keys = ['loss', 'chamfer_loss', 'normals_loss', 'collision_loss',
            'planar_loss', 'template_normals_loss', 'symmetry_loss']

    writer = SummaryWriter(
        os.path.join(args.checkpoint_dir, 'summaries',
                     datetime.datetime.now().strftime('training_log_%m%d%y_%H%M%S')),
        flush_secs=1)
    val_writer = SummaryWriter(
        os.path.join(args.checkpoint_dir, 'summaries',
                     datetime.datetime.now().strftime('validation_log_%m%d%y_%H%M%S')),
        flush_secs=1)

    trainer = ttools.Trainer(interface)
    trainer.add_callback(
        ttools.callbacks.TensorBoardLoggingCallback(keys=keys, writer=writer,
                                                    val_writer=val_writer,
                                                    frequency=3))
    trainer.add_callback(ttools.callbacks.ProgressBarCallback(keys=keys))
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(
        check_pointer, max_files=1, max_epochs=2))

    trainer.train(train_dataset, num_epochs=args.epoch,
                  val_dataloader=val_dataset, starting_epoch=starting_epoch)


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

    parser.add_argument("--checkpoint_dir", required=True,
                        help="Output directory where checkpoints are saved")
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

    train(args)
