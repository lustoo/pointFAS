import argparse


class DefaultConfigs(object):
    def __init__(self):
        opt = self.get_config()
        # from opt
        self.iter_per_epoch = opt.iter_per_epoch
        self.seed = opt.seed
        self.batch_size = opt.batch_size
        self.model = opt.model  # resnet18 or maddg
        self.start_val = opt.start_val  # 开始验证操作的最初ep
        # SGD
        self.weight_decay = opt.weight_decay
        self.momentum = opt.momentum
        # learning rate
        self.init_lr = opt.init_lr
        self.lr_epoch_1 = opt.lr_epoch_1
        self.lr_epoch_2 = opt.lr_epoch_2
        # model
        self.pretrained = opt.pretrained
        # training parameters
        self.gpu = opt.gpu
        self.norm_flag = opt.norm_flag
        self.max_iter = opt.max_iter
        # lambda_triplet = 1
        # lambda_adreal = 0.5
        # test model name
        self.tgt_best_model_name = 'model_best_0.08_29.pth.tar'
        self.load = opt.load
        self.rand_vec = opt.rand_vec
        self.lambda_triplet = opt.lambda_triplet


        ''''''
        if opt.target == 'O':
            self.src1_data = 'casia'
            self.src2_data = 'replay'
            self.src3_data = 'msu'
            self.tgt_data = 'oulu'
            self.experiemnt = 'O_ICM'
            self.normalzation = 'IN'
        elif opt.target == 'M':
            self.src1_data = 'casia'
            self.src2_data = 'replay'
            self.src3_data = 'oulu'
            self.tgt_data = 'msu'
            self.experiemnt = 'M_CIO'
            self.normalzation = 'IN'
        elif opt.target == 'C':
            self.src1_data = 'oulu'
            self.src2_data = 'replay'
            self.src3_data = 'msu'
            self.tgt_data = 'casia'
            self.experiemnt = 'C_IOM'
            self.normalzation = 'IN'

        elif opt.target == 'I':
            self.src1_data = 'casia'
            self.src2_data = 'oulu'
            self.src3_data = 'msu'
            self.tgt_data = 'replay'
            self.experiemnt = 'I_OCM'
            self.normalzation = 'IN'

        ''''''

        self.src1_train_num_frames = 1
        self.src2_train_num_frames = 1
        self.src3_train_num_frames = 1
        self.tgt_test_num_frames = 2

        # paths information
        self.result_path = './experiment/' + self.experiemnt + '/'
        self.checkpoint_path = self.result_path + self.tgt_data + '_checkpoint/' + self.model + '/DGFANet/'
        self.best_model_path = self.result_path + self.tgt_data + '_checkpoint/' + self.model + '/best_model/'
        self.logs = self.result_path + 'logs/'

    def get_config(self):
        parser = argparse.ArgumentParser('_')
        parser.add_argument('--seed', type=int, default=666)
        parser.add_argument('--rand_vec', type=int, default=50)
        parser.add_argument('--weight_decay', type=float, default=5e-4)
        parser.add_argument('--lambda_triplet', type=float, default=1)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--init_lr', type=float, default=0.01)
        parser.add_argument('--lr_epoch_1', type=int, default=0)
        parser.add_argument('--lr_epoch_2', type=int, default=50)
        parser.add_argument('--pretrained', type=bool, default=True)
        parser.add_argument('--load', type=bool, default=False)
        parser.add_argument('--model', type=str, default='maddg')
        parser.add_argument('--gpu', type=str, default='0')
        parser.add_argument('--batch_size', type=int, default=10)
        parser.add_argument('--iter_per_epoch', type=int, default=10)
        parser.add_argument('--start_val', type=int, default=50)
        parser.add_argument('--norm_flag', type=bool, default=True)
        parser.add_argument('--max_iter', type=int, default=500000)
        parser.add_argument('--tgt_best_model_name', type=str, default='model_best_0.08_29.pth.tar')
        parser.add_argument('--target', type=str, required=True, choices=['O', 'M', 'I', 'C'])

        opt = parser.parse_args()
        print(opt)
        return opt


config = DefaultConfigs()


