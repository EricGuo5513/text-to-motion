from options.base_options import BaseOptions
import argparse

class TrainCompOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

        self.parser.add_argument('--max_sub_epoch', type=int, default=50, help='Maximum epoch while training a scheduled length')
        self.parser.add_argument('--tf_ratio', type=float, default=0.4, help='Teacher forcing ratio')

        self.parser.add_argument('--feat_bias', type=float, default=5, help='Scales for global motion features and foot contact')

        self.parser.add_argument('--early_stop_count', type=int, default=3, help='To early stop each subepoch in a scheduled length')

        self.parser.add_argument('--lambda_rec_mov', type=float, default=1, help='Weight factor for snippet reconstruction')
        self.parser.add_argument('--lambda_rec_mot', type=float, default=1, help='Weight factor for motion reconstruction')

        self.parser.add_argument('--lambda_kld', type=float, default=0.005, help='Weight factor for KL Divergence')

        self.parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')

        self.parser.add_argument('--is_continue', action="store_true", help='Is this trail continued from previous trail?')

        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress (by iteration)')
        self.parser.add_argument('--save_every_e', type=int, default=10, help='Frequency of saving models (by epoch)')
        self.parser.add_argument('--eval_every_e', type=int, default=5, help='Frequency of animation results (by epoch)')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of saving models (by iteration)')

        self.is_train = True


class TrainDecompOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--name', type=str, default="test", help='Name of this trial')
        self.parser.add_argument("--gpu_id", type=int, default=-1,
                                 help='GPU id')

        self.parser.add_argument('--dataset_name', type=str, default='kit', help='Dataset Name')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        self.parser.add_argument("--window_size", type=int, default=40, help="Length of motion clips for reconstruction")

        self.parser.add_argument('--dim_movement_enc_hidden', type=int, default=512,
                                 help='Dimension of hidden in AutoEncoder(encoder)')
        self.parser.add_argument('--dim_movement_dec_hidden', type=int, default=512,
                                 help='Dimension of hidden in AutoEncoder(decoder)')
        self.parser.add_argument('--dim_movement_latent', type=int, default=512, help='Dimension of motion snippet')

        self.parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

        self.parser.add_argument('--max_epoch', type=int, default=270, help='Training iterations')

        self.parser.add_argument('--feat_bias', type=float, default=5, help='Layers of GRU')

        self.parser.add_argument('--lambda_sparsity', type=float, default=0.001, help='Layers of GRU')
        self.parser.add_argument('--lambda_smooth', type=float, default=0.001, help='Layers of GRU')

        self.parser.add_argument('--lr', type=float, default=1e-4, help='Layers of GRU')

        self.parser.add_argument('--is_continue', action="store_true", help='Training iterations')

        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress')
        self.parser.add_argument('--save_every_e', type=int, default=10, help='Frequency of printing training progress')
        self.parser.add_argument('--eval_every_e', type=int, default=3, help='Frequency of printing training progress')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of printing training progress')

    def parse(self):
        self.opt = self.parser.parse_args()
        self.opt.is_train = True
        args = vars(self.opt)
        return self.opt


class TrainLenEstOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--name', type=str, default="test", help='Name of this trial')
        self.parser.add_argument("--gpu_id", type=int, default=-1,
                                 help='GPU id')

        self.parser.add_argument('--dataset_name', type=str, default='t2m', help='Dataset Name')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        self.parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

        self.parser.add_argument("--unit_length", type=int, default=4, help="Length of motion")
        self.parser.add_argument("--max_text_len", type=int, default=20, help="Length of motion")

        self.parser.add_argument('--max_epoch', type=int, default=300, help='Training iterations')
        self.parser.add_argument('--estimator_mod', type=str, default='bigru')
        self.parser.add_argument('--feat_bias', type=float, default=5, help='Layers of GRU')


        self.parser.add_argument('--lr', type=float, default=1e-4, help='Layers of GRU')

        self.parser.add_argument('--is_continue', action="store_true", help='Training iterations')

        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress')
        self.parser.add_argument('--save_every_e', type=int, default=5, help='Frequency of printing training progress')
        self.parser.add_argument('--eval_every_e', type=int, default=3, help='Frequency of printing training progress')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of printing training progress')

    def parse(self):
        self.opt = self.parser.parse_args()
        self.opt.is_train = True
        args = vars(self.opt)
        return self.opt

class TrainTexMotMatchOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--name', type=str, default="test", help='Name of this trial')
        self.parser.add_argument("--gpu_id", type=int, default=-1,
                                 help='GPU id')

        self.parser.add_argument('--dataset_name', type=str, default='t2m', help='Dataset Name')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--decomp_name', type=str, default="Decomp_SP001_SM001_H512", help='Name of this trial')

        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

        self.parser.add_argument("--unit_length", type=int, default=4, help="Length of motion")
        self.parser.add_argument("--max_text_len", type=int, default=20, help="Length of motion")

        self.parser.add_argument('--dim_movement_enc_hidden', type=int, default=512,
                                 help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--dim_movement_latent', type=int, default=512, help='Dimension of hidden unit in GRU')

        self.parser.add_argument('--dim_text_hidden', type=int, default=512, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--dim_motion_hidden', type=int, default=1024, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--dim_coemb_hidden', type=int, default=512, help='Dimension of hidden unit in GRU')

        self.parser.add_argument('--max_epoch', type=int, default=300, help='Training iterations')
        self.parser.add_argument('--estimator_mod', type=str, default='bigru')
        self.parser.add_argument('--feat_bias', type=float, default=5, help='Layers of GRU')
        self.parser.add_argument('--negative_margin', type=float, default=10.0)

        self.parser.add_argument('--lr', type=float, default=1e-4, help='Layers of GRU')

        self.parser.add_argument('--is_continue', action="store_true", help='Training iterations')

        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress')
        self.parser.add_argument('--save_every_e', type=int, default=5, help='Frequency of printing training progress')
        self.parser.add_argument('--eval_every_e', type=int, default=5, help='Frequency of printing training progress')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of printing training progress')

    def parse(self):
        self.opt = self.parser.parse_args()
        self.opt.is_train = True
        args = vars(self.opt)
        return self.opt