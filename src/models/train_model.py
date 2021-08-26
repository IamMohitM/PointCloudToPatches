import argparse
import os

import torch

from src.models.PointnetModel import ReconstructionModel
from src.models.custom_trainers import ReduceOnPlateauTrainer
from src.models.interfaces import ReconstructionInterface
from src.scripts import utils


def train(args):
    train_dataset, val_dataset, _ = utils.load_modelnet(args)
    template_parameters = utils.load_template_parameters(args)

    model = ReconstructionModel(args, len(template_parameters['initial_parameters']),
                                init=template_parameters['initial_parameters'])

    if args.pretrained_model_path:
        model = utils.load_pretrained(model, args.pretrained_model_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.decay_step,
                                                           factor=args.decay_rate, verbose=True)

    trainer = utils.training_setup(args, model=model, interface_class=ReconstructionInterface,
                                   trainer_class=ReduceOnPlateauTrainer,
                                   template_parameters=template_parameters, optimizer=optimizer,
                                   scheduler=scheduler, log_directory_name=args.log_dir_name)

    starting_epoch = None
    checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model.pth')

    if os.path.exists(checkpoint_path):
        map_location =None if args.cuda else 'cpu'
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        trainer.interface.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.interface.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.interface.best_val_loss = checkpoint['loss']
        trainer.interface.scheduler.load_state_dict(checkpoint['scheduler'])
        starting_epoch = checkpoint['epoch']

        print(
            f'Loading checkpoint and starting training from epoch {starting_epoch} and current learning rate {utils.get_lr(trainer.interface.optimizer)}')
    else:
        print("No checkpoints")

    trainer.train(train_dataset, num_epochs=args.epoch,
                  val_dataloader=val_dataset, starting_epoch=starting_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('python src/scripts/train_model.py')
    parser.add_argument('--dataset_path', required=True,
                        help='Path to dataset - expected to contain modelnet{num_category}_train_new.txt, '
                             'modelnet{num_category}_train_new.txt, modelnet{num_category}_test_new.txt,'
                             ' modelnet{num_category}_val_new.txt')
    parser.add_argument("--no_cuda", action="store_false", dest="cuda", help="Force CPU")
    parser.add_argument("--cuda", action="store_true", dest="cuda", help="Force GPU")
    parser.add_argument('--pretrained_model_path', help="Path to Pretrained PointNet or EdgeConv Model -Must have ",
                        default=None)
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--num_category', default=10, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Number of points')
    parser.add_argument('--decay_rate', type=float, default=0.5, help='decay rate')
    parser.add_argument('--decay_step', type=float, default=10, help='decay step')
    parser.add_argument('--use_normals', action='store_true', default=True, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=True, help='save data offline')
    parser.add_argument('--no_process_data', action='store_false', default=True, dest='process_data',
                        help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=True,
                        help='use uniform sampling, default - True')
    parser.add_argument('--template_dir', required=True, help="Path to Sphere templates")

    parser.add_argument("--checkpoint_dir", required=True,
                        help="Output directory where checkpoints are saved")
    parser.add_argument("--w_normals", type=float, default=0.008, help='Weight for Normals loss. Set zero to ignore')
    parser.add_argument("--w_collision", type=float, default=1e-5,  help='Weight for Collission loss. Set zero to ignore')
    parser.add_argument("--sigma_collision", type=float, default=1e-6)
    parser.add_argument("--w_planar", type=float, default=2, help='Weight for Planarity loss. Set zero to ignore')
    parser.add_argument("--w_templatenormals", type=float, default=1e-4, help='Weight for Template Normals loss. Set zero to ignore')
    parser.add_argument("--w_symmetry", type=float, default=1)
    parser.add_argument("--n_samples", type=int, default=7000)
    parser.add_argument('--symmetries', dest='symmetries', action='store_true')
    parser.add_argument('--no-symmetries',
                        dest='symmetries', action='store_false')
    parser.add_argument('--wheels', dest='wheels', action='store_true')
    parser.add_argument('--no-wheels', dest='wheels', action='store_false')
    parser.add_argument('--log_dir_suffix', required=True, help='Suffix of Log directory name in checkpoint directory')

    parser.add_argument('--encoder', type=str, default="pointnet", help = "Type of Point Cloud Encoder - EdgeConv or PointNet")
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings for EdgeConv')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use with EdgeConv encoder')

    parser.set_defaults( num_worker_threads=8)

    args = parser.parse_args()

    train(args)
