import os

from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.evaluate_options import TestOptions
from torch.utils.data import DataLoader
from utils.plot_script import *

from networks.modules import *
from networks.trainers import CompTrainerV5
from data.dataset import Text2MotionDataset, collate_fn
from scripts.motion_process import *
from utils.word_vectorizer import WordVectorizer, POS_enumerator

def plot_t2m(data, save_dir, captions, do_denoise=False, ep_curves=None):
    if do_denoise:
        data = data * dec_std + dec_mean
    else:
        data = dataset.inv_transform(data)

    # print(ep_curves.shape)
    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = '%s_%02d'%(save_dir, i)
        plot_3d_motion(save_path + '.mp4', paramUtil.t2m_kinematic_chain, joint, title=caption, fps=20)
        # print(ep_curve.shape)
        if ep_curves is not None:
            ep_curve = ep_curves[i]
            plt.plot(ep_curve)
            plt.title(caption)
            plt.savefig(save_path + '.png')
            plt.close()

def loadDecompModel(opt):
    movement_enc = MovementVAEEncoder(dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    movement_dec = MovementVAEDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, dim_pose,
                                      length=opt.unit_length)
    movement2_dec = MovementDecoder(opt.dim_movement_latent * 2, opt.dim_movement2_dec_hidden, dim_pose,
                                    length=opt.unit_length * 2)

    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.decomp_name, 'model', 'latest.tar'),
                            map_location=opt.device)
    movement_enc.load_state_dict(checkpoint['movement_enc'])
    movement_dec.load_state_dict(checkpoint['movement_dec'])
    movement2_dec.load_state_dict(checkpoint['movement2_dec'])

    return movement_enc, movement_dec, movement2_dec


def denoise(movement_enc, movement2_dec, motions):
    movement_enc.eval()
    movement2_dec.eval()
    denoised_motions = []

    motions = motions.cpu().numpy()
    motions = (motions * std + mean - dec_mean) / dec_std
    motions = torch.from_numpy(motions).to(opt.device).float()
    last_motion = motions[:, :opt.unit_length]
    steps = motions.shape[1] // opt.unit_length
    for i in range(1, steps):
        clips = motions[:, i * opt.unit_length: (i+1) * opt.unit_length]
        _, _, latent1 = movement_enc(last_motion[..., :-4])
        _, _, latent2 = movement_enc(clips[..., :-4])
        latent_com = torch.cat([latent1, latent2], dim=-1)
        de_motions = movement2_dec(latent_com)
        if i == 1:
            denoised_motions.append(de_motions)
        else:
            denoised_motions.append(de_motions[:, opt.unit_length:])
        last_motion = de_motions[:, opt.unit_length:].clone()
    denoised_motions = torch.cat(denoised_motions, dim=1)
    return denoised_motions


def build_models(opt):
    text_encoder = TextEncoderBiGRU(input_size=dim_word + dim_pos_ohot,
                                    hidden_size=opt.dim_text_hidden,
                                    device=opt.device)

    seq_prior = TextDecoder(input_size=opt.dim_att_vec + dim_pose - 4,
                               output_size=opt.dim_z,
                               hidden_size=opt.dim_pri_hidden,
                               n_layers=opt.n_layers_pri)

    seq_decoder = TextVAEDecoder(input_size=opt.dim_att_vec + opt.dim_z + dim_pose - 4,
                             output_size=dim_pose,
                             hidden_size=opt.dim_dec_hidden,
                             n_layers=opt.n_layers_dec)

    att_layer = AttLayer(query_dim=opt.dim_msd_hidden,
                         key_dim=opt.dim_text_hidden * 2,
                         value_dim=opt.dim_att_vec)

    # latent_dis = LatentDis(input_size=opt.dim_z * 2)

    # return text_encoder, text_decoder, att_layer, vae_pri, vae_dec, vae_pos, motion_dis, movement_dis, latent_dis
    return text_encoder, seq_prior, seq_decoder, att_layer



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
        opt.data_root = './dataset/pose_data_raw'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        dim_pose = 263
        dim_word = 300
        dim_pos_ohot = len(POS_enumerator)

        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        std = np.load(pjoin(opt.meta_dir, 'std.npy'))

        dec_mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.decomp_name, 'meta', 'mean.npy'))
        dec_std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.decomp_name, 'meta', 'std.npy'))

        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, opt.split_file)
    else:
        raise KeyError('Dataset Does Not Exist')


    text_enc, seq_pri, seq_dec, att_layer = build_models(opt)
    mov_enc, mov_dec, mov2_dec = loadDecompModel(opt)

    trainer = CompTrainerV5(opt, text_enc, seq_pri, seq_dec, att_layer)

    dataset = Text2MotionDataset(opt, mean, std, split_file, w_vectorizer)
    dataset.reset_max_len(opt.start_mov_len * opt.unit_length)
    epoch, it, sub_ep, schedule_len = trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))
    print('Loading model: Epoch %03d Schedule_len %03d'%(epoch, schedule_len))
    trainer.eval_mode()
    trainer.to(opt.device)
    mov_enc.to(opt.device)
    mov2_dec.to(opt.device)

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
            for t in range(opt.repeat_times):
                pred_motions, _, _ = trainer.generate(word_emb, pos_ohot, cap_lens, m_lens, dim_pose)
                # trainer.forward(data, 0, m_lens[0]//opt.unit_length)
                # pred_motions = trainer.pred_motions.view(opt.batch_size, m_lens[0], -1)
                # ep_curves = trainer.ep_curve
                sub_dict = {}
                if opt.do_denoise:
                    de_motions = denoise(mov_enc, mov2_dec, pred_motions)
                    sub_dict['de_motion'] = de_motions.cpu().numpy()
                sub_dict['motion'] = pred_motions.cpu().numpy()
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
        plot_t2m(gt_motions, pjoin(animation_save_path, 'gt_motion'), captions, do_denoise=False)
        for t in range(opt.repeat_times):
            sub_dict = item['result_%02d'%t]
            motion = sub_dict['motion']
            # np.save(pjoin(joint_save_path, 'gen_motion_%02d.npy'%t), motion)
            # plot_t2m(motion, pjoin(animation_save_path, 'gen_motion_%02d'%t), captions, do_denoise=False)
            if opt.do_denoise:
                de_motion = sub_dict['de_motion']
                np.save(pjoin(joint_save_path, 'gen_de_motion_%02d.npy' % t), de_motion)
                plot_t2m(de_motion, pjoin(animation_save_path, 'gen_de_motion_%02d'%t), captions, do_denoise=True)
