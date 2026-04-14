from torch.nn.modules import loss
import os
import sys

from model.hash_model import DSPH as DSPH
from torch.nn import functional
from tqdm import tqdm
import torch.optim as optim
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import scipy.io as scio
# from hhf import HHF
from .base import TrainBase
from model.optimization import BertAdam
from utils import get_args, calc_neighbor, cosine_similarity, euclidean_similarity
from utils.calc_utils import calc_map_k_matrix as calc_map_k
# from utils.calc_utils import calc_map_k
from utils.utils import HyP
from dataset.dataloader import dataloader
import json
import os
# from ASL import AsymmetricLoss
import time
from copy import deepcopy

import torch
#from loguru import logger
# pip install pytorch-metric-learning
from pytorch_metric_learning import distances, reducers
from torch.optim import Adam
from utils import get_config
#from AdaTriplet.config import get_config
from AdaTriplet.losses import TripletCustomMarginLoss, LowerBoundLoss ,bit_var_loss
from AdaTriplet.methods import MetricLearningMethods
from AdaTriplet.miners.triplet_automargin_miner import TripletAutoParamsMiner
from AdaTriplet.networks import BackboneModel
import time
# from losses import contrastive_jaccard

from utils.CPF_loss import CPF
from DSH import DSHLoss
#from loss import RelaHashLoss
from relative_similarity import *
#from alex import AlexNet
#from timm.utils import AverageMeter
from pytorch_metric_learning import distances, reducers
from FAST_HPP import HouseHolder

class Trainer(TrainBase):

    def __init__(self,
                rank=2):
        args = get_args()
        super(Trainer, self).__init__(args, rank)

        self.best_epoch = -1
        self.best_avg_score = -1.0
        self.best_scale_results = {}

        self.logger.info(">>>>>> All Configuration Parameters:")
        args_dict = vars(self.args)
        
        max_key_length = max(len(key) for key in args_dict.keys())
        
        for key, value in args_dict.items():
            padded_key = key.ljust(max_key_length)
            self.logger.info(f">>>>>>> {padded_key}: {value}")

        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.run()

    def _init_model(self):
        self.logger.info("init model.")
        self.logger.info("ViT+GPT!")
        HashModel = DSPH
        self.model = HashModel(outputDim=self.args.output_dim, clipPath=self.args.clip_path,
                            writer=self.writer, logger=self.logger, is_train=self.args.is_train).to(self.rank)
        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))
        
        self.model.float()

        self.cpf_modules = {
            '8': CPF(embed_dim=8, n_classes=self.args.numclass, device=self.rank).to(self.rank),
            '16': CPF(embed_dim=16, n_classes=self.args.numclass, device=self.rank).to(self.rank),
            '32': CPF(embed_dim=32, n_classes=self.args.numclass, device=self.rank).to(self.rank),
            '64': CPF(embed_dim=64, n_classes=self.args.numclass, device=self.rank).to(self.rank),
            '128': CPF(embed_dim=128, n_classes=self.args.numclass, device=self.rank).to(self.rank),
        }

        self.rot_layers = {
            '8': HouseHolder(dim=8).to(self.rank),
            '16': HouseHolder(dim=16).to(self.rank),
            '32': HouseHolder(dim=32).to(self.rank),
            '64': HouseHolder(dim=64).to(self.rank),
            '128': HouseHolder(dim=128).to(self.rank),
        }

        rot_params = []
        for rot in self.rot_layers.values():
            rot_params.extend(rot.parameters())
        
        cpf_params = []
        for cpf in self.cpf_modules.values():
            cpf_params.extend(cpf.parameters())

        

        self.optimizer = BertAdam([
                    {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
                    {'params': self.model.image_multi.parameters(), 'lr': self.args.lr},                   
                    {'params': self.model.text_multi.parameters(), 'lr': self.args.lr},
                    {'params': rot_params, 'lr': self.args.clip_lr},
                    {'params': cpf_params, 'lr': 1e-5},  
                    ], lr=self.args.lr, warmup=self.args.warmup_proportion, schedule='warmup_cosine',
                    b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.args.epochs,
                    weight_decay=self.args.weight_decay, max_grad_norm=1.0)

       
        self.optimizer_loss = None
        self.total_time = 0

    def _init_dataset(self):
        self.config = get_config()
        self.logger.info("init dataset.")
        self.logger.info(f"Using {self.args.dataset} dataset.")
        self.args.index_file = os.path.join("./dataset", self.args.dataset, self.args.index_file)
        self.args.caption_file = os.path.join("./dataset", self.args.dataset, self.args.caption_file)
        self.args.label_file = os.path.join("./dataset", self.args.dataset, self.args.label_file)
        train_data, query_data, retrieval_data = dataloader(captionFile=self.args.caption_file, 
                                        indexFile=self.args.index_file, 
                                        labelFile=self.args.label_file, 
                                        maxWords=self.args.max_words,
                                        imageResolution=self.args.resolution,
                                        query_num=self.args.query_num,
                                        train_num=self.args.train_num,
                                        seed=self.args.seed)
        self.train_labels = train_data.get_all_label().to(self.rank)
        self.query_labels = query_data.get_all_label()
        self.retrieval_labels = retrieval_data.get_all_label()
        self.args.retrieval_num = len(self.retrieval_labels)
        self.logger.info(f"query shape: {self.query_labels.shape}")
        self.logger.info(f"retrieval shape: {self.retrieval_labels.shape}")
        self.train_loader = DataLoader(
                dataset=train_data,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                shuffle=True
            )
        self.query_loader = DataLoader(
                dataset=query_data,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                shuffle=True
            )
        self.retrieval_loader = DataLoader(
                dataset=retrieval_data,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                shuffle=True
            )


    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info(">>>>>> epochs: %d/%d"%(epoch, self.args.epochs))
        total_loss = 0.0  
        scale_losses = {s: 0.0 for s in ['8', '16', '32', '64', '128']}  
        
        distance = distances.CosineSimilarity()
        reducer = reducers.ThresholdReducer(low=0)
        mining_func = TripletAutoParamsMiner(
            distance=distance,
            margin_init=self.config.margin_m_loss,
            beta_init=self.config.margin_beta,
            type_of_triplets=self.config.type_of_triplets,
            k=self.config.k_param_automargin,
            k_n=self.config.k_n_param_autobeta,
            k_p=self.config.k_p_param_autobeta,
            mode=self.config.automargin_mode,
        )
        loss_matching_func = TripletCustomMarginLoss(margin=self.config.margin_m_loss, distance=distance, reducer=reducer)
        loss_id_func = LowerBoundLoss()
        # ##

        for iteration , (image, text, label, index) in enumerate(self.train_loader):
            start_time = time.time()
            self.global_step += 1
            image.float()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            label = label.to(self.rank, non_blocking=True).float()

            img_final, txt_final, img_scales, txt_scales = self.model(image, text)
            
            batch_total_loss = 0.0
            
            for scale in ['8', '16', '32', '64', '128']:
                img_scale = img_scales[scale]
                txt_scale = txt_scales[scale]
                
                cpf_loss = self.cpf_modules[scale](img_scale, txt_scale, label)
                
                criterion = bit_var_loss()
                method = MetricLearningMethods(self.config, mining_func, loss_matching=loss_matching_func,
                                            loss_identity=loss_id_func)
                t_img_loss = method.calculate_total_loss(img_scale, label, epoch_id=epoch, batch_id=iteration)
                t_text_loss = method.calculate_total_loss(txt_scale, label, epoch_id=epoch, batch_id=iteration)
                
                
                rot_layer = self.rot_layers[scale]
                img_rot = F.normalize(rot_layer(img_scale.T).T)  
                text_rot = F.normalize(rot_layer(txt_scale.T).T)

                q_img_loss = criterion(img_rot)
                q_text_loss = criterion(text_rot)
                
                lossLQ = t_text_loss + q_text_loss + q_img_loss + t_img_loss
                scale_total_loss = cpf_loss + lossLQ
                
                scale_losses[scale] += scale_total_loss.item()
                
                batch_total_loss += scale_total_loss
            
            total_loss += batch_total_loss.item()
            
            self.optimizer.zero_grad()
            batch_total_loss.backward()  
            self.optimizer.step()

            self.total_time += time.time() - start_time
            ##

        avg_total_loss = total_loss / len(self.train_loader)
        lr_info = '-'.join([str('%.9f'%itm) for itm in sorted(list(set(self.optimizer.get_lr())))])
        
        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}] Total loss: {avg_total_loss:.6f}, LR: {lr_info}, Time: {self.total_time}")
        for scale in ['8', '16', '32', '64', '128']:
            avg_scale_loss = scale_losses[scale] / len(self.train_loader)
            self.logger.info(f">>>>>> Scale {scale} bits loss: {avg_scale_loss:.6f}")


    def train(self):
        self.logger.info("Start train.")
        
        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)
            self.valid(epoch)  
            self.save_model(epoch)
        
        self.logger.info(f">>>>>>> FINISHED >>>>>> Best model at epoch {self.best_epoch} with avg score: {self.best_avg_score:.6f}")
        
        for scale in ['16', '32', '64']:
            result = self.best_scale_results.get(scale, {'i2t': 0, 't2i': 0})
            self.logger.info(f"Scale {scale} bits | I2T: {result['i2t']:.6f}, T2I: {result['t2i']:.6f}")
            

    def get_code(self, data_loader, length: int):
        img_buffers = {s: torch.empty(length, int(s)).to(self.rank) for s in ['8', '16', '32', '64', '128']}
        txt_buffers = {s: torch.empty(length, int(s)).to(self.rank) for s in ['8', '16', '32', '64', '128']}
        
        encoder_time = 0
        for image, text, label, index in tqdm(data_loader):
            start_encoder_time = time.time()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            index = index.numpy()
            
            _, _, img_scales, txt_scales = self.model(image, text)
            
            for scale in ['8', '16', '32', '64', '128']:
                img_scale = img_scales[scale]
                txt_scale = txt_scales[scale]
                
                img_scale = torch.sign(img_scale)
                txt_scale = torch.sign(txt_scale)
                
                img_buffers[scale][index, :] = img_scale.data
                txt_buffers[scale][index, :] = txt_scale.data
            
            encoder_time = time.time() - start_encoder_time
        
        return img_buffers, txt_buffers, encoder_time


    def test(self, mode_name="i2t"):
        if self.args.pretrained == "":
            raise RuntimeError("test step must load a model! please set the --pretrained argument.")
        
        self.logger.info(f"Loading pretrained model: {self.args.pretrained}")
        self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))
        
        self.change_state(mode="valid")
        save_dir = os.path.join(self.args.save_dir, "test_results")
        os.makedirs(save_dir, exist_ok=True)
        
        query_img_scales, query_txt_scales, _ = self.get_code(self.query_loader, self.args.query_num)
        retrieval_img_scales, retrieval_txt_scales, _ = self.get_code(self.retrieval_loader, self.args.retrieval_num)
        
        for scale in ['16', '32', '64']:
            self.logger.info(f">>>>>> Calculating mAP for {scale}-bit hashes")
            
            q_img = query_img_scales[scale]
            q_txt = query_txt_scales[scale]
            r_img = retrieval_img_scales[scale]
            r_txt = retrieval_txt_scales[scale]
            
            mAPi2t = calc_map_k(q_img, r_txt, self.query_labels, self.retrieval_labels, 5000, self.rank)
            mAPt2i = calc_map_k(q_txt, r_img, self.query_labels, self.retrieval_labels, 5000, self.rank)
            mAPi2i = calc_map_k(q_img, r_img, self.query_labels, self.retrieval_labels, 5000, self.rank)
            mAPt2t = calc_map_k(q_txt, r_txt, self.query_labels, self.retrieval_labels, 5000, self.rank)
            
            self.logger.info(f">>>>>> Scale {scale} bits | MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}")
            
            q_img_np = q_img.cpu().detach().numpy()
            q_txt_np = q_txt.cpu().detach().numpy()
            r_img_np = r_img.cpu().detach().numpy()
            r_txt_np = r_txt.cpu().detach().numpy()
            q_l_np = self.query_labels.numpy()
            r_l_np = self.retrieval_labels.numpy()

            result_dict = {
                'q_img': q_img_np,
                'q_txt': q_txt_np,
                'r_img': r_img_np,
                'r_txt': r_txt_np,
                'q_l': q_l_np,
                'r_l': r_l_np
            }
            
            mat_file = os.path.join(save_dir, f"{scale}bits-{self.args.dataset}-{mode_name}.mat")
            scio.savemat(mat_file, result_dict)
            self.logger.info(f">>>>>> Saved {scale}-bit hashes to {mat_file}")
        
        self.logger.info(">>>>>> Test completed for scales 16, 32, 64 bits")



    def valid(self, epoch):
            self.logger.info("Valid.")
            self.change_state(mode="valid")
            query_img_scales, query_txt_scales, q_encoder_time = self.get_code(self.query_loader, self.args.query_num)
            retrieval_img_scales, retrieval_txt_scales, r_encoder_time = self.get_code(self.retrieval_loader, self.args.retrieval_num)
            
            scale_results = {}
            avg_score = 0.0  
            
            # 计算所有尺度的mAP
            for scale in ['8', '16', '32', '64', '128']:
                q_img = query_img_scales[scale]
                q_txt = query_txt_scales[scale]
                r_img = retrieval_img_scales[scale]
                r_txt = retrieval_txt_scales[scale]
                
                map_i2t = calc_map_k(q_img, r_txt, self.query_labels, self.retrieval_labels, 5000, self.rank)
                map_t2i = calc_map_k(q_txt, r_img, self.query_labels, self.retrieval_labels, 5000, self.rank)
                
                scale_results[scale] = {'i2t': map_i2t, 't2i': map_t2i}
                
                if scale in ['16', '32', '64']:
                    avg_score += (map_i2t + map_t2i) / 2
            
            avg_score /= 3
            
            if avg_score > self.best_avg_score:
                self.best_epoch = epoch
                self.best_avg_score = avg_score
                self.best_scale_results = scale_results
                self.save_all_scales(query_img_scales, query_txt_scales, 
                                retrieval_img_scales, retrieval_txt_scales)
            
            output_scale = str(self.args.output_dim)
            self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}], MAP(i->t): {scale_results[output_scale]['i2t']}, MAP(t->i): {scale_results[output_scale]['t2i']}")
            self.logger.info(f">>>>>> 16/32/64 Avg: {avg_score:.6f}, Best Avg: {self.best_avg_score:.6f} (Epoch {self.best_epoch})")
            
            return scale_results

    def save_all_scales(self, query_img_scales, query_txt_scales, 
                       retrieval_img_scales, retrieval_txt_scales):
        save_dir = os.path.join(self.args.save_dir, "best_model_scales")
        os.makedirs(save_dir, exist_ok=True)
        
        for scale in ['8', '16', '32', '64', '128']:
            q_img = query_img_scales[scale].cpu().detach().numpy()
            q_txt = query_txt_scales[scale].cpu().detach().numpy()
            r_img = retrieval_img_scales[scale].cpu().detach().numpy()
            r_txt = retrieval_txt_scales[scale].cpu().detach().numpy()
            q_l = self.query_labels.numpy()
            r_l = self.retrieval_labels.numpy()
            
            result_dict = {
                'q_img': q_img,
                'q_txt': q_txt,
                'r_img': r_img,
                'r_txt': r_txt,
                'q_l': q_l,
                'r_l': r_l
            }
            
            scio.savemat(os.path.join(save_dir, f"{scale}bits-best.mat"), result_dict)
        
        self.logger.info(f">>>>>> Saved all scales for best model at epoch {self.best_epoch}")


