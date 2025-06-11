import os
import sys
import glob
import yaml
import json
import torch
import pprint
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from munch import munchify
from torchvision import transforms
from collections import OrderedDict
from models import VisDynamicsModel
from models_latentpred import VisLatentDynamicsModel
from dataset import NeuralPhysDataset
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger


def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def seed(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.if_cuda:
        torch.cuda.manual_seed(cfg.seed)
    # uncomment for strict reproducibility
    # torch.set_deterministic(True)


def model_rollout():
    config_filepath = str(sys.argv[2])
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)

    log_dir = '_'.join([cfg.log_dir,
                        cfg.dataset,
                        cfg.model_name,
                        str(cfg.seed)])

    model = VisDynamicsModel(lr=cfg.lr,
                             seed=cfg.seed,
                             if_cuda=cfg.if_cuda,
                             if_test=True,
                             gamma=cfg.gamma,
                             log_dir=log_dir,
                             train_batch=cfg.train_batch,
                             val_batch=cfg.val_batch,
                             test_batch=cfg.test_batch,
                             num_workers=cfg.num_workers,
                             model_name=cfg.model_name,
                             data_filepath=cfg.data_filepath,
                             dataset=cfg.dataset,
                             lr_schedule=cfg.lr_schedule)
    
    # load model
    if cfg.model_name == 'encoder-decoder' or cfg.model_name == 'encoder-decoder-64':
        checkpoint_filepath = str(sys.argv[3])
        checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath, '*.ckpt'))[0]
        ckpt = torch.load(checkpoint_filepath)
        model.load_state_dict(ckpt['state_dict'])
    
    if 'refine' in cfg.model_name:
        checkpoint_filepath = str(sys.argv[4])
        checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath, '*.ckpt'))[0]
        ckpt = torch.load(checkpoint_filepath)
        ckpt = rename_ckpt_for_multi_models(ckpt)
        model.model.load_state_dict(ckpt)

        high_dim_checkpoint_filepath = str(sys.argv[3])
        high_dim_checkpoint_filepath = glob.glob(os.path.join(high_dim_checkpoint_filepath, '*.ckpt'))[0]
        ckpt = torch.load(high_dim_checkpoint_filepath)
        ckpt = rename_ckpt_for_multi_models(ckpt)
        model.high_dim_model.load_state_dict(ckpt)

    model = model.to('cuda')
    model.eval()
    model.freeze()

    # get all the test video ids
    data_filepath_base = os.path.join(cfg.data_filepath, cfg.dataset)
    with open(os.path.join('datainfo', cfg.dataset, f'data_split_dict_{cfg.seed}.json'), 'r') as file:
        seq_dict = json.load(file)
    test_vid_ids = seq_dict['test']

    pred_len = int(sys.argv[6])
    long_term_folder = os.path.join(log_dir, 'prediction_long_term', 'model_rollout')
    loss_dict = {}

    if cfg.model_name == 'encoder-decoder' or cfg.model_name == 'encoder-decoder-64':
        for p_vid_idx in tqdm(test_vid_ids):
            vid_filepath = os.path.join(data_filepath_base, str(p_vid_idx))
            total_num_frames = len(os.listdir(vid_filepath))
            suf = os.listdir(vid_filepath)[0].split('.')[-1]
            data = None
            saved_folder = os.path.join(long_term_folder, str(p_vid_idx))
            mkdir(saved_folder)
            loss_lst = []
            for start_frame_idx in range(total_num_frames - 3):
                if start_frame_idx % 2 != 0:
                    continue
                # take the initial input from ground truth data
                if start_frame_idx % pred_len == 0:
                    data = [get_data(os.path.join(vid_filepath, f'{start_frame_idx}.{suf}')),
                            get_data(os.path.join(vid_filepath, f'{start_frame_idx+1}.{suf}'))]
                    data = (torch.cat(data, 2)).unsqueeze(0)
                # get the target
                target = [get_data(os.path.join(vid_filepath, f'{start_frame_idx+2}.{suf}')),
                          get_data(os.path.join(vid_filepath, f'{start_frame_idx+3}.{suf}'))]
                target = (torch.cat(target, 2)).unsqueeze(0)
                # feed into the model
                if cfg.model_name == 'encoder-decoder':
                    output, latent = model.model(data.cuda())
                if cfg.model_name == 'encoder-decoder-64':
                    output, latent = model.model(data.cuda(), data.cuda(), False)

                # compute loss
                loss_lst.append(float(model.loss_func(output, target.cuda()).cpu().detach().numpy()))
                
                # save (2', 3'), (4', 5'), ...

                np.save(os.path.join(saved_folder, f'{start_frame_idx+2}.{suf}'), output[0, :, :, :128].cpu().detach().numpy())
                np.save(os.path.join(saved_folder, f'{start_frame_idx+3}.{suf}'), output[0, :, :, 128:].cpu().detach().numpy())

                # the output becomes the input data in the next iteration
                data = torch.tensor(output.cpu().detach().numpy()).float()

            loss_dict[p_vid_idx] = loss_lst

        # save the test loss for all the testing videos
        with open(os.path.join(long_term_folder, 'test_loss.json'), 'w') as file:
            json.dump(loss_dict, file, indent=4)
    
    if 'refine' in cfg.model_name:
        for p_vid_idx in tqdm(test_vid_ids):
            vid_filepath = os.path.join(data_filepath_base, str(p_vid_idx))
            total_num_frames = len(os.listdir(vid_filepath))
            suf = os.listdir(vid_filepath)[0].split('.')[-1]
            data = None
            saved_folder = os.path.join(long_term_folder, str(p_vid_idx))
            mkdir(saved_folder)
            loss_lst = []
            for start_frame_idx in range(total_num_frames - 3):
                if start_frame_idx % 2 != 0:
                    continue
                # take the initial input from ground truth data
                if start_frame_idx % pred_len == 0:
                    data = [get_data(os.path.join(vid_filepath, f'{start_frame_idx}.{suf}')),
                            get_data(os.path.join(vid_filepath, f'{start_frame_idx+1}.{suf}'))]
                    data = (torch.cat(data, 2)).unsqueeze(0)
                # get the target
                target = [get_data(os.path.join(vid_filepath, f'{start_frame_idx+2}.{suf}')),
                          get_data(os.path.join(vid_filepath, f'{start_frame_idx+3}.{suf}'))]
                target = (torch.cat(target, 2)).unsqueeze(0)
                # feed into the model
                _, latent = model.high_dim_model(data.cuda(), data.cuda(), False)
                latent = latent.squeeze(-1).squeeze(-1)
                latent_reconstructed, latent_latent = model.model(latent)
                output, _ = model.high_dim_model(data.cuda(), latent_reconstructed.unsqueeze(2).unsqueeze(3), True)

                # compute loss
                loss_lst.append(float(model.loss_func(output, target.cuda()).cpu().detach().numpy()))

                # save (2', 3'), (4', 5'), ...
                np.save(os.path.join(saved_folder, f'{start_frame_idx+2}.{suf}'), output[0, :, :, :128].cpu().detach().numpy())
                np.save(os.path.join(saved_folder, f'{start_frame_idx+3}.{suf}'), output[0, :, :, 128:].cpu().detach().numpy())

                # the output becomes the input data in the next iteration
                data = torch.tensor(output.cpu().detach().numpy()).float()

            loss_dict[p_vid_idx] = loss_lst

        # save the test loss for all the testing videos
        with open(os.path.join(long_term_folder, 'test_loss.json'), 'w') as file:
            json.dump(loss_dict, file, indent=4)


def model_rollout_hybrid(step):
    config_filepath = str(sys.argv[2])
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)

    if 'refine' not in cfg.model_name:
        assert False, "the hybrid scheme is only supported with refine model..."

    log_dir = '_'.join([cfg.log_dir,
                        cfg.dataset,
                        cfg.model_name,
                        str(cfg.seed)])

    model = VisDynamicsModel(lr=cfg.lr,
                             seed=cfg.seed,
                             if_cuda=cfg.if_cuda,
                             if_test=True,
                             gamma=cfg.gamma,
                             log_dir=log_dir,
                             train_batch=cfg.train_batch,
                             val_batch=cfg.val_batch,
                             test_batch=cfg.test_batch,
                             num_workers=cfg.num_workers,
                             model_name=cfg.model_name,
                             data_filepath=cfg.data_filepath,
                             dataset=cfg.dataset,
                             lr_schedule=cfg.lr_schedule)

    # load model
    checkpoint_filepath = str(sys.argv[4])
    checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath, '*.ckpt'))[0]
    ckpt = torch.load(checkpoint_filepath)
    ckpt = rename_ckpt_for_multi_models(ckpt)
    model.model.load_state_dict(ckpt)

    high_dim_checkpoint_filepath = str(sys.argv[3])
    high_dim_checkpoint_filepath = glob.glob(os.path.join(high_dim_checkpoint_filepath, '*.ckpt'))[0]
    ckpt = torch.load(high_dim_checkpoint_filepath)
    ckpt = rename_ckpt_for_multi_models(ckpt)
    model.high_dim_model.load_state_dict(ckpt)

    model = model.to('cuda')
    model.eval()
    model.freeze()

    # get all the test video ids
    data_filepath_base = os.path.join(cfg.data_filepath, cfg.dataset)
    with open(os.path.join('../datainfo', cfg.dataset, f'data_split_dict_{cfg.seed}.json'), 'r') as file:
        seq_dict = json.load(file)
    test_vid_ids = seq_dict['test']

    pred_len = int(sys.argv[6])
    long_term_folder = os.path.join(log_dir, 'prediction_long_term', f'hybrid_rollout_{step}')
    loss_dict = {}

    for p_vid_idx in tqdm(test_vid_ids):
        vid_filepath = os.path.join(data_filepath_base, str(p_vid_idx))
        total_num_frames = len(os.listdir(vid_filepath))
        suf = os.listdir(vid_filepath)[0].split('.')[-1]
        data = None
        saved_folder = os.path.join(long_term_folder, str(p_vid_idx))
        mkdir(saved_folder)
        loss_lst = []
        for start_frame_idx in range(total_num_frames - 3):
            if start_frame_idx % 2 != 0:
                continue
            # take the initial input from ground truth data
            if start_frame_idx % pred_len == 0:
                data = [get_data(os.path.join(vid_filepath, f'{start_frame_idx}.{suf}')),
                        get_data(os.path.join(vid_filepath, f'{start_frame_idx+1}.{suf}'))]
                data = (torch.cat(data, 2)).unsqueeze(0)
            # get the target
            target = [get_data(os.path.join(vid_filepath, f'{start_frame_idx+2}.{suf}')),
                      get_data(os.path.join(vid_filepath, f'{start_frame_idx+3}.{suf}'))]
            target = (torch.cat(target, 2)).unsqueeze(0)
            # feed into the model
            if (start_frame_idx + 2) % (2 * step + 2) == 0:
                _, latent = model.high_dim_model(data.cuda(), data.cuda(), False)
                latent = latent.squeeze(-1).squeeze(-1)
                latent_reconstructed, latent_latent = model.model(latent)
                output, _ = model.high_dim_model(data.cuda(), latent_reconstructed.unsqueeze(2).unsqueeze(3), True)
            else:
                output, _ = model.high_dim_model(data.cuda(), data.cuda(), False)

            # compute loss
            loss_lst.append(float(model.loss_func(output, target.cuda()).cpu().detach().numpy()))

            # save (2', 3'), (4', 5'), ...
            img = tensor_to_img(output[0, :, :, :128])
            img.save(os.path.join(saved_folder, f'{start_frame_idx+2}.{suf}'))
            img = tensor_to_img(output[0, :, :, 128:])
            img.save(os.path.join(saved_folder, f'{start_frame_idx+3}.{suf}'))

            # the output becomes the input data in the next iteration
            data = torch.tensor(output.cpu().detach().numpy()).float()

        loss_dict[p_vid_idx] = loss_lst

    # save the test loss for all the testing videos
    with open(os.path.join(long_term_folder, 'test_loss.json'), 'w') as file:
        json.dump(loss_dict, file, indent=4)



def rename_ckpt_for_multi_models(ckpt):
    renamed_state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        if 'high_dim_model' in k:
            name = k.replace('high_dim_model.', '')
        else:
            name = k.replace('model.', '')
        renamed_state_dict[name] = v
    return renamed_state_dict

def get_data(filepath):
    data = np.load(filepath)
    data = (torch.tensor(data).unsqueeze(0).float() - (-3000)) / (9951 - (-3000))
    return data



if __name__ == '__main__':
    if str(sys.argv[1]) == 'model-rollout':
        model_rollout()
    elif 'hybrid' in str(sys.argv[1]):
        step = int(sys.argv[1].split('-')[-1])
        model_rollout_hybrid(step)
    else:
        assert False, "prediction scheme is not supported..."