import os

from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.train_options import TrainCompOptions
from utils.plot_script import *

from networks.modules import *
from networks.trainers import CompTrainerV6
from data.dataset import Text2MotionDataset
from scripts.motion_process import *
from utils.word_vectorizer import WordVectorizer, POS_enumerator

def plot_t2m(data, save_dir, captions, ep_curves=None):
    data = train_dataset.inv_transform(data)
    # print(ep_curves.shape)
    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4'%(i))
        plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=fps, radius=radius)
        # print(ep_curve.shape)
        if ep_curves is not None:
            ep_curve = ep_curves[i]
            plt.plot(ep_curve)
            plt.title(caption)
            save_path = pjoin(save_dir, '%02d.png' % (i))
            plt.savefig(save_path)
            plt.close()


def loadDecompModel(opt):
    movement_enc = MovementConvEncoder(dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, dim_pose)

    if not opt.is_continue:
        checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.decomp_name, 'model', 'latest.tar'),
                                map_location=opt.device)
        movement_enc.load_state_dict(checkpoint['movement_enc'])
        movement_dec.load_state_dict(checkpoint['movement_dec'])

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

    seq_posterior = TextDecoder(text_size=text_size,
                                input_size=opt.dim_att_vec + opt.dim_movement_latent * 2,
                                output_size=opt.dim_z,
                                hidden_size=opt.dim_pos_hidden,
                                n_layers=opt.n_layers_pos)

    seq_decoder = TextVAEDecoder(text_size=text_size,
                                 input_size=opt.dim_att_vec + opt.dim_z + opt.dim_movement_latent,
                                 output_size=opt.dim_movement_latent,
                                 hidden_size=opt.dim_dec_hidden,
                                 n_layers=opt.n_layers_dec)

    att_layer = AttLayer(query_dim=opt.dim_pos_hidden,
                         key_dim=text_size,
                         value_dim=opt.dim_att_vec)

    # latent_dis = LatentDis(input_size=opt.dim_z * 2)

    # return text_encoder, text_decoder, att_layer, vae_pri, vae_dec, vae_pos, motion_dis, movement_dis, latent_dis
    return text_encoder, seq_prior, seq_posterior, seq_decoder, att_layer



if __name__ == '__main__':
    parser = TrainCompOptions()
    opt = parser.parse()

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        radius = 4
        fps = 20
        opt.max_motion_length = 196
        dim_pose = 263
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'kit':
        opt.data_root = './dataset/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.kit_kinematic_chain

    else:
        raise KeyError('Dataset Does Not Exist')

    dim_word = 300
    dim_pos_ohot = len(POS_enumerator)
    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')
    movement_enc, movement_dec = loadDecompModel(opt)

    text_encoder, seq_prior, seq_posterior, seq_decoder, att_layer = build_models(opt)
    print(text_encoder)
    print(seq_prior)
    print(seq_posterior)
    print(seq_decoder)
    print(att_layer)

    trainer = CompTrainerV6(opt, text_encoder, seq_prior, seq_decoder, att_layer, movement_dec,
                            mov_enc=movement_enc, seq_post=seq_posterior)

    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file, w_vectorizer)
    val_dataset = Text2MotionDataset(opt, mean, std, val_split_file, w_vectorizer)

    trainer.train(train_dataset, val_dataset, plot_t2m)