import os

from os.path import join as pjoin

from options.train_options import TrainLenEstOptions

from networks.modules import *
from networks.trainers import LengthEstTrainer, collate_fn
from data.dataset import Text2MotionDataset
from scripts.motion_process import *
from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizer, POS_enumerator
import torch.nn as nn
import codecs as cs


if __name__ == '__main__':
    parser = TrainLenEstOptions()
    opt = parser.parse()
    opt.is_train = False

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')

    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.max_motion_length = 196
        dim_pose = 263
        dim_word = 300
        dim_pos_ohot = len(POS_enumerator)
        num_classes = 200 // opt.unit_length

        mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
        std = np.load(pjoin(opt.data_root, 'Std.npy'))

        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, 'val.txt')
    else:
        raise KeyError('Dataset Does Not Exist')

    if opt.estimator_mod == 'bigru':
        estimator = MotionLenEstimatorBiGRU(dim_word, dim_pos_ohot, 512, num_classes)
    else:
        raise Exception('Estimator Mode is not Recognized!!!')


    checkpoints = torch.load(pjoin(opt.model_dir, 'latest.tar'))
    estimator.load_state_dict(checkpoints['estimator'])
    estimator.to(opt.device)
    estimator.eval()

    dataset = Text2MotionDataset(opt, mean, std, split_file, w_vectorizer)

    loader = DataLoader(dataset, batch_size=1, drop_last=True, num_workers=4,
                            shuffle=True, collate_fn=collate_fn, pin_memory=True)

    softmax = nn.Softmax(-1)
    save_path = pjoin('eval_results', opt.dataset_name, opt.name, 'test2')
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i > 40:
                break
            word_emb, pos_ohot, caption, cap_lens, _, m_lens = data
            word_emb = word_emb.detach().to(opt.device).float()
            pos_ohot = pos_ohot.detach().to(opt.device).float()

            pred_dis = estimator(word_emb, pos_ohot, cap_lens)
            pred_dis = softmax(pred_dis).squeeze().cpu().numpy()
            pred_list = list(pred_dis)
            pred_list = [str(item) for item in pred_list]
            pred_str = ' '.join(pred_list)
            with cs.open(pjoin(save_path, '%d.txt'%i), 'w') as f:
                f.write(caption[0] + '\n')
                f.write(pred_str + '\n')
                f.write(str(m_lens[0].item() // opt.unit_length))

            max_idxs = pred_dis.argsort()[-5:][::-1]
            max_values = pred_dis[max_idxs]

            print(caption)
            print(max_idxs)
            print(max_values)
            print(m_lens[0] // opt.unit_length)