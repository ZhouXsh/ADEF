import logging
from random import randint
import sys
import torch
import torch.nn as nn
import os
import pickle
import numpy as np
import torch.optim as optim
from torch.utils import data
from tensorboardX import SummaryWriter
sys.path.append(os.path.dirname(os.path.abspath("../")))
from src.dataset import infinite_data_loader
from src.scheduler import GradualWarmupScheduler
from src.modules.common import PositionalEncoding

torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import os.path as osp
 
from rich.progress import track

from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.utils.camera import get_rotation_matrix
from src.utils.video import images2video, add_audio_to_video
from src.utils.io import load_image_rgb, resize_to_limit
from src.utils.helper import mkdir, basename, dct2device
from src.utils.rprint import rlog as log
# from src.ADEF_wrapper import ADEFWrapper
import tyro
from src.modules.emotion_level_classifier import EmotionTransformer as Classifier

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


emo_list = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']
emo_label = ['ang',  'con',  'dis',  'fea',  'hap',  'neu',  'sad',  'sur']

# 同分布
class DiT_Emo_Dataset(data.Dataset):
    def __init__(self, root_dir='/mnt/disk2/zhouxishi/JoyVASA/src/prepare_data', gt_motion_filename="front_motions.pkl", dit_motion_filename="front_dit_motions.pkl"):
        self.template_dict = pickle.load(open('/mnt/disk2/zhouxishi/JoyVASA/src/my_prepare/joyvasa_motion_template.pkl', 'rb'))
        self.gt_motion_data = pickle.load(open(os.path.join(root_dir, gt_motion_filename), "rb"))
        self.dit_motion_data = pickle.load(open(os.path.join(root_dir, dit_motion_filename), "rb"))
        print("load all motion data done...")
        self.eps = 1e-9
        
        txt_path = os.path.join(root_dir, "front_train.txt")
        with open(txt_path, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file.readlines()]
            self.all_data = [{
                "video_name": line,
                "audio_name": line[:-4]+'.wav'
            } for line in lines]
        self.all_num = 100

    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, index):
        metadata = self.all_data[index] 
        # /W037/front/angry/level_3/W037_front_angry_level_3_034.mp4
        emotype = metadata['audio_name'].split('/')[-3]   # angry
        emo_index = torch.tensor(emo_list.index(emotype), dtype=torch.long)  # (1,)

        level_ = int(metadata['audio_name'].split('/')[-2].split('_')[-1])-1   # 0~2
        emo_level = torch.tensor(level_, dtype=torch.long)  # (1,)
    
        template_dict = self.template_dict         # 同分布

        gt_motions = self.gt_motion_data[metadata["audio_name"]]
        dit_motions = self.dit_motion_data[metadata["audio_name"]]

        min_frames = min(gt_motions['n_frames'], dit_motions['n_frames'])
        
        gt_list, dit_list = [],[]
        for i in range(min_frames):
            gt_motion_exp = gt_motions['motion'][i]['exp'] * 1.5
            dit_motion_exp = dit_motions['motion'][i]['exp']           # (1,21,3)

            normalized_exp11 = (gt_motion_exp.flatten() - template_dict["mean_exp"]) / (template_dict["std_exp"] + self.eps)
            normalized_exp22 = (dit_motion_exp.flatten() - template_dict["mean_exp"]) / (template_dict["std_exp"] + self.eps)

            gt_motion_exp = torch.tensor(normalized_exp11, dtype=torch.float32)           # (63)
            dit_motion_exp = torch.tensor(normalized_exp22, dtype=torch.float32)          # (63)

            gt_list.append(gt_motion_exp)
            dit_list.append(dit_motion_exp)

        gt_motion_exps =  torch.stack(gt_list, dim=0)      # (min_frames, 63)
        dit_motion_exps =  torch.stack(dit_list, dim=0)    # (min_frames, 63)

        while gt_motion_exps.shape[0] < self.all_num + 2:
            gt_motion_exps = torch.cat([gt_motion_exps, gt_motion_exps], dim=0)
            dit_motion_exps = torch.cat([dit_motion_exps, dit_motion_exps], dim=0)

        end_frame = randint(self.all_num, gt_motion_exps.shape[0] - 1)
        start_frame = end_frame - self.all_num

        dit_prev = dit_motion_exps[start_frame:end_frame]
        gt_prev = gt_motion_exps[start_frame:end_frame]

        return dit_prev, gt_prev, emo_index, emo_level

class EmotionTransformer(nn.Module):
    def __init__(self, input_dim=63, emotion_dim=8, embed_dim=512, num_heads=8, ff_dim=512*4, num_layers=6):
        super(EmotionTransformer, self).__init__()
        
        self.embed_dim = embed_dim
        
        # 运动参数嵌入层 (B, L, 63) -> (B, L, D)
        self.motion_embedding = nn.Linear(input_dim, embed_dim)
        
        # 位置编码
        # self.positional_encoding = nn.Parameter(torch.zeros(1, 100, embed_dim))  # 假设最大 L=100
        self.PE = PositionalEncoding(self.embed_dim)

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Transformer Decoder
        decoder_layers = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)

        self.emotion_embedding = nn.Embedding(emotion_dim, embed_dim)  # (B, 1, 8) -> (B, 1, D)
        self.emotion_level_embedding = nn.Embedding(3, embed_dim) 

        # 融合情感类别+等级（拼接后映射）
        self.emotion_fusion = nn.Linear(embed_dim * 2, embed_dim)

        # 输出层 (B, 1, D) -> (B, 1, 63)
        self.output_layer = nn.Linear(embed_dim, input_dim)

    def forward(self, motion_seq, emotion, level):
        """
        motion_seq: (B, L, 63)  无表情运动序列
        emotion: (B,)  One-hot 情感信息
        """

        emotion = self.emotion_embedding(emotion)    # (B, )  ->  (B, D)
        level = self.emotion_level_embedding(level)  # (B, )  ->  (B, D)

        emotion_embed = torch.cat([emotion, level], dim=-1)        # (B, D*2)
        emotion_embed = self.emotion_fusion(emotion_embed).unsqueeze(1)  # (B, 1, D)

        # 运动参数嵌入 + 位置编码
        motion_embed = self.motion_embedding(motion_seq)  # (B, L, D)
        motion_embed = self.PE(motion_embed)

        # 经过 Transformer Encoder
        encoded_features = self.encoder(motion_embed)  # (B, L, D)

        # 经过 Transformer Decoder
        decoded_output = self.decoder(encoded_features, emotion_embed)  # (B, 1, D)

        # 输出运动参数 (B, 1, D) -> (B, 1, 63)
        output = self.output_layer(decoded_output)

        return output

def transf_train():
    # 训练超参数
    num_epochs = 120000
    learning_rate = 1e-4

    warm_iter = 12000
    decay_iter = 120000
    batch_size = 64
    # 同分布
    log_dir = f'./0511_64_120000_gt15增强'
    
    writer = SummaryWriter(log_dir)   # 路径

    # 初始化模型
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = EmotionTransformer().to(device)

    # 0417 情感分类器    0426高dim且同分布
    emo_classifier = Classifier().to(device)
    emo_classifier.load_state_dict(torch.load(f'/mnt/disk2/zhouxishi/JoyVASA/pretrained_weights/ADEF/emo_classifier/emo_level_classifier.pth', map_location=device))
    emo_classifier.eval()
    criterion = torch.nn.CrossEntropyLoss()

    # 定义损失函数
    mse_loss = nn.MSELoss()  # 主要损失，约束运动参数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    after_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, decay_iter, learning_rate * 0.02)
    scheduler = GradualWarmupScheduler(optimizer, 1, warm_iter, after_scheduler)

    train_dataset = DiT_Emo_Dataset()
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader = infinite_data_loader(train_loader)   # 将数据加载器（train_loader）转换为一个无限循环的迭代器

    model.train()
    loss_log = {
        'total_loss': [],
        'mse_loss': []
        ,'emo_loss':[]
        ,'level_loss': []
    }

    for epoch in range(num_epochs):
        dit_prev, _, gt_prev, _, emo_index, emo_level = next(train_loader)
        dit_prev = dit_prev.to(device)     # （B,100,63）
        # dit_cur = dit_cur.to(device)      # （B,100,63）
        emo_index = emo_index.to(device)     # （B,）
        gt_prev = gt_prev.to(device)       # （B,100,63）
        # gt_cur = gt_cur.to(device)       # （B,100,63）
        emo_level = emo_level.to(device)

        optimizer.zero_grad()

        pred = model(dit_prev, emo_index, emo_level)     #   (B, 1, 63)

        B, L, _ = dit_prev.shape
        dit_prev = dit_prev + pred.expand(-1, L, -1)   # (B, 1, 63)  -> (B, L, 63)

        # === 计算各个损失 ===
        loss_mse = mse_loss(dit_prev, gt_prev)  # (B, 100, 63)

        pred_emo, pred_level = emo_classifier(dit_prev)   # (N,100,63)  -> (N,8)
        loss_emo = criterion(pred_emo, emo_index)
        loss_level = criterion(pred_level, emo_level)

        # === 计算总损失 ===
        total_loss = loss_mse + loss_emo + loss_level

        # 反向传播
        total_loss.backward()
        optimizer.step()

        # Inside your training loop, after computing the losses:
        # Logging - Append current batch losses
        loss_log['total_loss'].append(total_loss.mean().item())
        loss_log['mse_loss'].append(loss_mse.mean().item())
        loss_log['emo_loss'].append(loss_emo.mean().item())
        loss_log['level_loss'].append(loss_emo.mean().item())

        # Create description string for logging
        description = f'Iter: {epoch}\t Train loss: [Total: {np.mean(loss_log["total_loss"]):.3e}'
        description += f", MSE: {np.mean(loss_log['mse_loss']):.3e}"
        description += f", Emo: {np.mean(loss_log['emo_loss']):.3e}"
        description += f", Level: {np.mean(loss_log['level_loss']):.3e}"
        description += ']'
        logging.info(description)

        # Write to tensorboard
        if epoch % 50 == 0 and writer is not None:
            writer.add_scalar('train/total_loss', np.mean(loss_log['total_loss']), epoch)
            writer.add_scalar('train/mse_loss', np.mean(loss_log['mse_loss']), epoch)
            writer.add_scalar('train/emo_loss', np.mean(loss_log['emo_loss']), epoch)
            writer.add_scalar('train/level_loss', np.mean(loss_log['level_loss']), epoch)
            writer.add_scalar('opt/lr', optimizer.param_groups[0]['lr'], epoch)
            
            # Clear the loss log for next interval
            for key in loss_log.keys():
                loss_log[key].clear()

        # update learning rate  更新学习率
        if scheduler is not None and epoch < warm_iter + decay_iter:   # 调度器用于更新学习率  区分：优化器optimizor
            scheduler.step()

        # 打印训练信息
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.6f}")

    print("训练完成！")
    save_dir = f"{log_dir}/ckpt.pth"
    torch.save(model.state_dict(), save_dir)

def infer(emo_le = 2):
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)
    inf_cfg = partial_fields(InferenceConfig, args.__dict__)    # 从args.__dict__选取InferenceConfig对应的字段 组成的字典
    adef_wrapper = ADEFWrapper(inference_cfg=inf_cfg)       # 核心功能包装器
    device = adef_wrapper.device          # cuda:0

    transf_model = EmotionTransformer().to(device)
    enhancer_p = '/mnt/disk2/zhouxishi/ADEF/pretrained_weights/ADEF/emo_enhancer/emo_enhancer.pth'
    transf_model_data = torch.load(enhancer_p, map_location=device)
    transf_model.load_state_dict(transf_model_data, strict=False)   # ['model']
    transf_model.eval()
    templates_dict = pickle.load(open('/mnt/disk2/zhouxishi/JoyVASA/src/my_prepare/joyvasa_motion_template.pkl', 'rb'))

    args.output_dir = f'/mnt/disk2/zhouxishi/JoyVASA/测试效果/eval/level{emo_le+1}'
    emo_level = torch.tensor(emo_le, dtype=torch.long).unsqueeze(0).to(device)
    image = '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/first_frame/M003_front_happy_level_3_027.png'
    audio = '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/videos/M003/front/happy/level_3/M003_front_happy_level_3_027.wav'
    test_datas = [(image, audio)]

    image_list = ['/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/first_frame/M003_front_angry_level_1_001.png',
                  '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/first_frame/M003_front_contempt_level_1_001.png',
                  '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/first_frame/M003_front_disgusted_level_1_001.png',
                  '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/first_frame/M003_front_fear_level_1_001.png',
                  '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/first_frame/M003_front_happy_level_1_001.png',
                  '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/first_frame/M003_front_neutral_level_1_001.png',
                  '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/first_frame/M003_front_sad_level_1_001.png',
                  '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/first_frame/M003_front_surprised_level_1_001.png',]

    audio_list = ['/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/videos/M003/front/angry/level_3/M003_front_angry_level_3_001.wav',
                  '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/videos/M003/front/contempt/level_3/M003_front_contempt_level_3_001.wav',
                  '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/videos/M003/front/disgusted/level_3/M003_front_disgusted_level_3_001.wav',
                  '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/videos/M003/front/fear/level_3/M003_front_fear_level_3_001.wav',
                  '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/videos/M003/front/happy/level_3/M003_front_happy_level_3_001.wav',
                  '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/videos/M003/front/neutral/level_1/M003_front_neutral_level_1_001.wav',
                  '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/videos/M003/front/sad/level_3/M003_front_sad_level_3_001.wav',
                  '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/videos/M003/front/surprised/level_3/M003_front_surprised_level_3_001.wav',]
    # test_datas = [(image_list[i],audio_list[i]) for i in range(len(audio_list))]

    for image,audio in test_datas:
    # for audio in audio_list:
        args.audio = audio
    ######## load reference image  加载参考图像########
        img_rgb = load_image_rgb(image)       # (h,w,3)   (336,336,3)
        # 调整图像的大小，使最大尺寸不超过max_dim，图像的宽度和高度是n（division）的倍数。
        img_rgb = resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)       # (h,w,3)   (336,336,3)
        log(f"Load reference image from {image}")
        source_rgb_lst = [img_rgb]    # 源rgb图像 列表，source_rgb_lst[0]就是img_rgb    len==1
        
    #########  原本实现 #########################
        driving_template_dct = adef_wrapper.gen_motion_sequence(args)
        n_frames = driving_template_dct['n_frames']                        # （音频对应的）总帧数

        emo_list = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']
        emotype = audio.split('/')[-3]
        emo_id = int(emo_list.index(emotype))
        print(emo_id)
        # template_dict = templates_dict[emo_id]
        template_dict = templates_dict
        batch, least = n_frames//100, n_frames % 100        # 先写死100帧

        ori_exp = []
        for i in range(n_frames):
            exp = driving_template_dct['motion'][i]['exp']   # (1,21,3)
            exp = (exp.flatten() - template_dict["mean_exp"]) / (template_dict["std_exp"] + 1e-9)   # (63)
            exp = torch.tensor(exp, dtype=torch.float32).to(device)  # (63)
            ori_exp.append(exp)
        if least > 0:
            batch += 1
            addi = 100 - least
            for i in range(addi):
                ori_exp.append(ori_exp[-1])
        ori_exp =  torch.stack(ori_exp, dim=0).to(device)      # (batch * 100, 63)
        
        ## (n,1,63)
        # dit_cur = ori_exp[0:100].unsqueeze(0).to(device)       # (1, 100, 63)  
        # emo_index = torch.tensor([emo_id], dtype=torch.long).to(device)     # (1)
        # pred_delta_exp = transf_model(dit_cur,emo_index)    # (1, 1, 63)
        # pred_delta_exp = pred_delta_exp.cpu().detach() * template_dict["std_exp"] + template_dict["mean_exp"]    # [63]
        # pred_delta_exp = pred_delta_exp.reshape(21, 3).unsqueeze(0)                 #   (1,21,3)

        ## (n,100,63)
        res_exp = []
        for i in range(batch):
            dit_cur = ori_exp[i*100:(i+1)*100].unsqueeze(0).to(device)       # (1, 100, 63)  
            emo_index = torch.tensor([emo_id], dtype=torch.long).to(device)     # (1)
            pred_delta_exp = transf_model(dit_cur,emo_index,emo_level)    # (1, 1, 63)
            res_exp.append(pred_delta_exp)          # (1, 100, 63)
        res_exp = torch.stack(res_exp, dim=1).squeeze()     # (1, batch*100, 63) ->  (batch*100, 63)       # (b,25,63) -  (b*25,63)  -  (b*25,21,3) -  (batch*25, 1, 21, 3)
        exp_all = torch.zeros(1,21,3)
        for i in range(n_frames):
            # 反归一化
            exp =  res_exp[i].cpu().detach() * template_dict["std_exp"] + template_dict["mean_exp"]    # [63]
            exp = exp.reshape(21, 3).unsqueeze(0)                 #   (1,21,3)
            exp_all += exp
        exp_all = exp_all / n_frames

        for i in range(n_frames):
            # driving_template_dct['motion'][i]['exp'] += 1 * pred_delta_exp.numpy() #  (1, 21, 3)
            driving_template_dct['motion'][i]['exp'] += 1 * exp_all.numpy()  #  (1, 21, 3)
        
        I_p_lst = []                                      # 生成的视频的图像列表，还没有粘贴到原图空间
        R_d_0, x_d_0_info = None, None             # 第一帧的旋转矩阵（R_d_0）和相应的运动信息（x_d_0_info） ？？？？？？？？

    ######## process source info 处理源参考图像相关的信息 ########
        img_crop_256x256 = cv2.resize(source_rgb_lst[0], (256, 256))  # force to resize to 256x256  强制调整大小为256x256   （256,256,3）
        
        I_s = adef_wrapper.prepare_source(img_crop_256x256)         # 处理后的图像  H x W x 3  ->  B x 3 x H x W
        x_s_info = adef_wrapper.get_kp_info(I_s)                    # 参考图像的计算隐式关键点时相关的信息
        x_c_s = x_s_info['kp']                                                    # 源图像的规范关键点Xc,s      B x 21 x 3 
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])    # 旋转矩阵R         (B, 3, 3)
        f_s = adef_wrapper.extract_feature_3d(I_s)                  # 参考图像的外观特征（预训练的F）   [1, 32, 16, 64, 64]
        x_s = adef_wrapper.transform_keypoint(x_s_info)             # 計算源隐式关键点Xs,k       (bs, k, 3)  B x 21 x 3 

    ######## animate 动画化（逐帧处理音频）########
        for i in track(range(n_frames), description='🚀Animating Image with Generated Motions...', total=n_frames):
            x_d_i_info = driving_template_dct['motion'][i]   # 包括该帧的exp, scale, R, t, pitch, yaw, roll
            x_d_i_info = dct2device(x_d_i_info, device)
            
            # R  旋转矩阵
            R_d_i = x_d_i_info['R'] if 'R' in x_d_i_info.keys() else x_d_i_info['R_d']  # compatible with previous keys 与以前的键兼容
            if i == 0:  # cache the first frame     缓存第一帧
                R_d_0 = R_d_i                      # 第一帧的旋转矩阵
                x_d_0_info = x_d_i_info.copy()     # 第一帧的exp, scale, R, t, pitch, yaw, roll信息
            if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":       # "exp", "pose", "lip", "eyes", "all"
                R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s  # Rd_i * Rd_0 * Rs    (B=1, 3, 3)
            else:        # "exp", "lip", "eyes"
                R_new = R_s           # 直接copy参考图像的旋转矩阵 (B=1, 3, 3)

            # delta  表情变化 （大道至简）
            delta_new = x_d_i_info['exp']

            # scale  缩放
            scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])   # （1,1）

            # translation  头部平移
            t_new = x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])    # 原图平移+相对首帧平移    （1,3）
            t_new[..., 2].fill_(0)  # zero tz        z坐标变为0

            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new         # 计算该帧的隐式关键点   （1,21,3）

            # 扭曲解码（图像生成）：外观特征，源隐式关键点，当前帧驱动隐式关键点
            out = adef_wrapper.warp_decode(f_s, x_s, x_d_i_new)    
            I_p_i = adef_wrapper.parse_output(out['out'])[0]  # 512x512x3, uint8   生成的图像（可显示的那种0~255）
            I_p_lst.append(I_p_i)         # 图像列表（还没粘贴回完整原图空间）

        # save the animated result       保存结果 
        mkdir(args.output_dir)
        temp_video = osp.join(args.output_dir, f'{basename(image)}_{basename(audio)}_temp.mp4')
        images2video(I_p_lst, wfp=temp_video, fps=inf_cfg.output_fps)
        final_video = osp.join(args.output_dir, f'{basename(image)}_{basename(audio)}.mp4')
        add_audio_to_video(temp_video, audio, final_video)     # 添加音轨
    return None

if __name__ == "__main__":
    # transf_train()
    infer()
