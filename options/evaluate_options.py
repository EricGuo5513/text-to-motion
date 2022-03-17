from options.base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
        self.parser.add_argument('--start_mov_len', type=int, default=10)
        self.parser.add_argument('--est_length', action="store_true")


        self.parser.add_argument('--repeat_times', type=int, default=3)
        self.parser.add_argument('--split_file', type=str, default='test.txt')
        self.parser.add_argument('--text_file', type=str, default="./input.txt", help='Frequency of plot points')
        self.parser.add_argument('--which_epoch', type=str, default="latest", help='Frequency of plot points')
        self.parser.add_argument('--result_path', type=str, default="./eval_results/", help='Frequency of plot points')
        self.parser.add_argument('--num_results', type=int, default=40, help='Batch size of pose discriminator')
        self.parser.add_argument('--ext', type=str, default='default', help='Batch size of pose discriminator')

        self.is_train = False
