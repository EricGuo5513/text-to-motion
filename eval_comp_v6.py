import os

from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.evaluate_options import TestOptions
from torch.utils.data import DataLoader
from utils.plot_script import *

from networks.modules import *
from networks.trainers import CompTrainerV6
from data.dataset import Text2MotionDataset, collate_fn
from scripts.motion_process import *
from utils.word_vectorizer import WordVectorizer, POS_enumerator

def plot_t2m(data, save_dir, captions):
    data = dataset.inv_transform(data)

    # print(ep_curves.shape)
    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = '%s_%02d'%(save_dir, i)
        plot_3d_motion(save_path + '.mp4', paramUtil.t2m_kinematic_chain, joint, title=caption, fps=20)


def loadDecompModel(opt):
    movement_enc = MovementConvEncoder(dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, dim_pose)

    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.decomp_name, 'model', 'latest.tar'),
                            map_location=opt.device)
    movement_enc.load_state_dict(checkpoint['movement_enc'])

    return movement_enc, movement_dec


def build_models(opt):
    if opt.text_enc_mod == 'bigru':
        text_encoder = TextEncoderBiGRU(word_size=dim_word,
                                        pos_size=dim_pos_ohot,
                                        hidden_size=opt.dim_text_hidden,
                                        device=opt.device)
        text_size = opt.dim_text_hidden * 2
    else:
        raise Exception("Text Encoder Mode not Recognized!!!")

    seq_prior = TextDecoder(text_size=text_size,
                            input_size=opt.dim_att_vec + opt.dim_movement_latent,
                            output_size=opt.dim_z,
                            hidden_size=opt.dim_pri_hidden,
                            n_layers=opt.n_layers_pri)


    seq_decoder = TextVAEDecoder(text_size=text_size,
                                 input_size=opt.dim_att_vec + opt.dim_z + opt.dim_movement_latent,
                                 output_size=opt.dim_movement_latent,
                                 hidden_size=opt.dim_dec_hidden,
                                 n_layers=opt.n_layers_dec)

    att_layer = AttLayer(query_dim=opt.dim_pos_hidden,
                         key_dim=text_size,
                         value_dim=opt.dim_att_vec)

    movement_enc = MovementConvEncoder(dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, dim_pose)

    # latent_dis = LatentDis(input_size=opt.dim_z * 2)

    # return text_encoder, text_decoder, att_layer, vae_pri, vae_dec, vae_pos, motion_dis, movement_dis, latent_dis
    return text_encoder, seq_prior, seq_decoder, att_layer, movement_enc, movement_dec



if __name__ == '__main__':
    parser = TestOptions()
    opt = parser.parse()
    opt.do_denoise = True

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    opt.result_dir = pjoin(opt.result_path, opt.dataset_name, opt.name, opt.ext)
    opt.joint_dir = pjoin(opt.result_dir, 'joints')
    opt.animation_dir = pjoin(opt.result_dir, 'animations')
    os.makedirs(opt.joint_dir, exist_ok=True)
    os.makedirs(opt.animation_dir, exist_ok=True)

    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        dim_pose = 263
        dim_word = 300
        dim_pos_ohot = len(POS_enumerator)
        num_classes = 200 // opt.unit_length

        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        std = np.load(pjoin(opt.meta_dir, 'std.npy'))

        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, opt.split_file)
        opt.max_motion_length = 196

    else:
        raise KeyError('Dataset Does Not Exist')


    text_enc, seq_pri, seq_dec, att_layer, mov_enc, mov_dec = build_models(opt)
    # mov_enc, mov_dec = loadDecompModel(opt)

    trainer = CompTrainerV6(opt, text_enc, seq_pri, seq_dec, att_layer, mov_dec, mov_enc=mov_enc)

    dataset = Text2MotionDataset(opt, mean, std, split_file, w_vectorizer)
    dataset.reset_max_len(opt.start_mov_len * opt.unit_length)
    epoch, it, sub_ep, schedule_len = trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))
    print('Loading model: Epoch %03d Schedule_len %03d'%(epoch, schedule_len))
    trainer.eval_mode()
    trainer.to(opt.device)
    # mov_enc.to(opt.device)
    # mov2_dec.to(opt.device)

    if opt.est_length:
        estimator = MotionLenEstimatorBiGRU(dim_word, dim_pos_ohot, 512, num_classes)
        checkpoints = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'length_est_bigru', 'model', 'latest.tar'))
        estimator.load_state_dict(checkpoints['estimator'])
        estimator.to(opt.device)
        estimator.eval()

    data_loader = DataLoader(dataset, batch_size=opt.batch_size, drop_last=True, num_workers=1,
                             shuffle=True, collate_fn=collate_fn)

    '''Generate Results'''
    print('Generate Results')
    result_dict = {}
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            print('%02d_%03d'%(i, opt.num_results))
            word_emb, pos_ohot, caption, cap_lens, motions, m_lens = data
            name = 'L%03dC%03d'%(m_lens[0], i)
            item_dict = {'caption': caption,
                         'length': m_lens[0],
                         'gt_motion': motions.numpy()}
            print(caption)

            if opt.est_length:
                word_emb, pos_ohot, caption, cap_lens, _, m_lens = data
                word_emb = word_emb.detach().to(opt.device).float()
                pos_ohot = pos_ohot.detach().to(opt.device).float()

                pred_dis = estimator(word_emb, pos_ohot, cap_lens)
                pred_dis = nn.Softmax(-1)(pred_dis).squeeze()

                # pred_dis_np = pred_dis.cpu().numpy()
                # max_idxs = pred_dis_np.argsort()[-5:][::-1]
                # max_values = pred_dis_np[max_idxs]
                # print(max_idxs)
                # print(max_values)
                # print(m_lens[0] // opt.unit_length)

            for t in range(opt.repeat_times):
                if opt.est_length:
                    length = torch.multinomial(pred_dis, opt.batch_size, replacement=True)
                    # print(length.item())
                    m_lens = length * opt.unit_length
                pred_motions, _, att_wgts = trainer.generate(word_emb, pos_ohot, cap_lens, m_lens, m_lens[0]//opt.unit_length, dim_pose)
                # trainer.forward(data, 0, m_lens[0]//opt.unit_length)
                # pred_motions = trainer.pred_motions.view(opt.batch_size, m_lens[0], -1)
                # ep_curves = trainer.ep_curve
                sub_dict = {}
                sub_dict['motion'] = pred_motions.cpu().numpy()
                sub_dict['att_wgts'] = att_wgts.cpu().numpy()
                item_dict['result_%02d'%t] = sub_dict
            result_dict[name] = item_dict
            if i > opt.num_results:
                break

    print('Animation Results')
    '''Animate Results'''
    for i, (key, item) in enumerate(result_dict.items()):
        print('%02d_%03d'%(i, opt.num_results))
        captions = item['caption']
        gt_motions = item['gt_motion']
        joint_save_path = pjoin(opt.joint_dir, key)
        animation_save_path = pjoin(opt.animation_dir, key)
        os.makedirs(joint_save_path, exist_ok=True)
        os.makedirs(animation_save_path, exist_ok=True)
        np.save(pjoin(joint_save_path, 'gt_motions.npy'), gt_motions)
        plot_t2m(gt_motions, pjoin(animation_save_path, 'gt_motion'), captions)
        for t in range(opt.repeat_times):
            sub_dict = item['result_%02d'%t]
            motion = sub_dict['motion']
            # att_wgts = sub_dict['att_wgts']
            np.save(pjoin(joint_save_path, 'gen_motion_%02d_L%03d.npy' % (t, motion.shape[1])), motion)
            # np.save(pjoin(joint_save_path, 'att_wgt_%02d_L%03d.npy' % (t, motion.shape[1])), att_wgts)
            plot_t2m(motion, pjoin(animation_save_path, 'gen_motion_%02d_L%03d' % (t, motion.shape[1])), captions)
