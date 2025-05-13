import pickle
import numpy as np

emo_list = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']
emo_label = ['ang',  'con',  'dis',  'fea',  'hap',  'neu',  'sad',  'sur']

# 各个emo单独分布
def all_cal(data_root, emo_index=0):
    emo_type = emo_list[emo_index]

    scale_list = []  # 所有音频的 所有帧的 scale
    R_list = []
    pitch_list = []
    yaw_list = []
    roll_list = []
    t_list = []
    exp_list = []
    
    motions = pickle.load(open(data_root, 'rb'))

    # 遍历所有音频文件，提取运动数据
    audio_names = motions.keys()
    for audio_name in audio_names:          # 所有音频   （还没有   level——up）
        if emo_type not in audio_name:   # 只计算特定情感的
            continue
        motion_data = motions[audio_name]     # eg：motions_data_1
        seq_len = motion_data["n_frames"]     # 运动数据的帧数
        for frame_idx in range(seq_len):      # 遍历每一帧，读取运动参数
            scale_list.append(motion_data['motion'][frame_idx]["scale"].flatten())
            R_list.append(motion_data['motion'][frame_idx]["R"].flatten())
            t_list.append(motion_data['motion'][frame_idx]["t"].flatten())
            exp_list.append(motion_data['motion'][frame_idx]["exp"].flatten())
            pitch_list.append(motion_data['motion'][frame_idx]["pitch"].flatten())
            yaw_list.append(motion_data['motion'][frame_idx]["yaw"].flatten())
            roll_list.append(motion_data['motion'][frame_idx]["roll"].flatten())
    
    # 转换为 NumPy 数组
    scale_array = np.array(scale_list)  # (811, 1)
    R_array = np.array(R_list)          # (811, 9)
    t_array = np.array(t_list)          # (811, 3)
    exp_array = np.array(exp_list)      # (811, 63)
    pitch_array = np.array(pitch_list)  # (811, 1)
    yaw_array = np.array(yaw_list)      # (811, 1)
    roll_array = np.array(roll_list)    # (811, 1)
    print(scale_array.shape, R_array.shape, t_array.shape, exp_array.shape, pitch_array.shape, yaw_array.shape, roll_array.shape)

    # 处理 lip 和 eyes 特征   处理 c_lip_lst（嘴唇形状数据）和 c_eyes_lst（眼睛形状数据）。
    lip_lst_array = np.array([data.flatten() for data in motion_data['c_lip_lst']]).astype(np.float32)
    eyes_lst_array = np.array([data.flatten() for data in motion_data['c_eyes_lst']]).astype(np.float32)
    print(f"lip_aray: {lip_lst_array.shape}, eyes_lst_array: {eyes_lst_array.shape}")

    # abs max  计算各类统计信息 的 绝对最大值
    abs_max_scale = np.max(abs(scale_array), axis=0)
    abs_max_R = np.max(abs(R_array), axis=0)
    abs_max_t = np.max(abs(t_array), axis=0)
    abs_max_exp = np.max(abs(exp_array), axis=0)
    abs_max_pitch = np.max(abs(pitch_array), axis=0)
    abs_max_yaw = np.max(abs(yaw_array), axis=0)
    abs_max_roll = np.max(abs(roll_array), axis=0)
    abs_max_lip = np.max(abs(lip_lst_array), axis=0)
    abs_max_eyes = np.max(abs(eyes_lst_array), axis=0)
    print("absmax: ", abs_max_scale.shape, abs_max_R.shape, abs_max_t.shape, abs_max_exp.shape, abs_max_pitch.shape, abs_max_pitch.shape, abs_max_roll.shape, abs_max_lip.shape, abs_max_eyes.shape)

    # max 计算最大值
    max_scale = np.max(scale_array, axis=0)
    max_R = np.max(R_array, axis=0)
    max_t = np.max(t_array, axis=0)
    max_exp = np.max(exp_array, axis=0)
    max_pitch = np.max(pitch_array, axis=0)
    max_yaw = np.max(yaw_array, axis=0)
    max_roll = np.max(roll_array, axis=0)
    max_lip = np.max(lip_lst_array, axis=0)
    max_eyes = np.max(eyes_lst_array, axis=0)
    print("max: ", max_scale.shape, max_R.shape, max_t.shape, max_exp.shape, max_pitch.shape, max_pitch.shape, max_roll.shape, max_lip.shape, max_eyes.shape)

    # min 计算最小值
    min_scale = np.min(scale_array, axis=0)
    min_R = np.min(R_array, axis=0)
    min_t = np.min(t_array, axis=0)
    min_exp = np.min(exp_array, axis=0)
    min_pitch = np.min(pitch_array, axis=0)
    min_yaw = np.min(yaw_array, axis=0)
    min_roll = np.min(roll_array, axis=0)
    min_lip = np.min(lip_lst_array, axis=0)
    min_eyes = np.min(eyes_lst_array, axis=0)
    print("min: ", min_scale.shape, min_R.shape, min_t.shape, min_exp.shape, min_pitch.shape, min_pitch.shape, min_roll.shape, min_lip.shape, min_eyes.shape)

    # mean 计算均值
    mean_scale = np.mean(scale_array, axis=0)
    mean_R = np.mean(R_array, axis=0)
    mean_t = np.mean(t_array, axis=0)
    mean_exp = np.mean(exp_array, axis=0)     # (n_frames, 63)  -> (63)
    mean_pitch = np.mean(pitch_array, axis=0)
    mean_yaw = np.mean(yaw_array, axis=0)
    mean_roll = np.mean(roll_array, axis=0)
    mean_lip = np.mean(lip_lst_array, axis=0)
    mean_eyes = np.mean(eyes_lst_array, axis=0)
    print("mean: ", mean_scale.shape, mean_R.shape, mean_t.shape, mean_exp.shape, mean_pitch.shape, mean_yaw.shape, mean_roll.shape, mean_lip.shape, mean_eyes.shape)

    # std 计算标准差
    std_scale = np.std(scale_array, axis=0)
    std_R = np.std(R_array, axis=0)
    std_t = np.std(t_array, axis=0)
    std_exp = np.std(exp_array, axis=0)
    std_pitch = np.std(pitch_array, axis=0)
    std_yaw = np.std(yaw_array, axis=0)
    std_roll = np.std(roll_array, axis=0)
    std_lip = np.std(lip_lst_array, axis=0)
    std_eyes = np.std(eyes_lst_array, axis=0)
    print("std: ", std_scale.shape, std_R.shape, std_t.shape, std_exp.shape, std_pitch.shape, std_yaw.shape, std_roll.shape, std_lip.shape, std_eyes.shape)

    # 保存统计信息
    motion_template = {
        "mean_scale": mean_scale,      # 均值
        "mean_R": mean_R,
        "mean_t": mean_t,
        "mean_exp": mean_exp,
        "mean_pitch": mean_pitch,
        "mean_yaw": mean_yaw,
        "mean_roll": mean_roll,
        "mean_lip": mean_lip,
        "mean_eyes": mean_eyes,
        "std_scale": std_scale,        # 标准差
        "std_R": std_R,
        "std_t": std_t,
        "std_exp": std_exp, 
        "std_pitch": std_pitch, 
        "std_yaw": std_yaw, 
        "std_roll": std_roll, 
        "std_lip": std_lip,
        "std_eyes": std_eyes,
        "max_scale": max_scale,          # 最大值
        "max_R": max_R,
        "max_t": max_t,
        "max_exp": max_exp,
        "max_pitch": max_pitch,
        "max_yaw": max_yaw,
        "max_roll": max_roll,
        "max_lip": max_lip,
        "max_eyes": max_eyes,
        "min_scale": min_scale,          # 最小值
        "min_R": min_R,
        "min_t": min_t,
        "min_exp": min_exp,
        "min_pitch": min_pitch,
        "min_yaw": min_yaw,
        "min_roll": min_roll,
        "min_lip": min_lip,
        "min_eyes": min_eyes,
        "abs_max_scale": abs_max_scale,   # 绝对最大值
        "abs_max_R": abs_max_R,
        "abs_max_t": abs_max_t,
        "abs_max_exp": abs_max_exp,
        "abs_max_pitch": abs_max_pitch,
        "abs_max_yaw": abs_max_yaw,
        "abs_max_roll": abs_max_roll,
        "abs_max_lip": abs_max_lip,
        "abs_max_eyes": abs_max_eyes,
    }
    return motion_template

# # 不level——up
Emotion_template = {}
for i in range(8):
    Emotion_template[i] = all_cal(emo_index=i)
pickle.dump(Emotion_template, open(f"emotion_template.pkl", 'wb'))
# # 将计算出的统计信息打包并保存到 front_all_motion_template.pkl，供后续训练或分析使用。


# 0425 把所有的emo相同分布
def all_same_cal(data_root):

    scale_list = []  # 所有音频的 所有帧的 scale
    R_list = []
    pitch_list = []
    yaw_list = []
    roll_list = []
    t_list = []
    exp_list = []
    
    motions = pickle.load(open(data_root, 'rb'))

    # 遍历所有音频文件，提取运动数据
    audio_names = motions.keys()
    for audio_name in audio_names:          # 所有音频   （还没有   level——up）
        motion_data = motions[audio_name]     # eg：motions_data_1
        seq_len = motion_data["n_frames"]     # 运动数据的帧数
        for frame_idx in range(seq_len):      # 遍历每一帧，读取运动参数
            scale_list.append(motion_data['motion'][frame_idx]["scale"].flatten())
            R_list.append(motion_data['motion'][frame_idx]["R"].flatten())
            t_list.append(motion_data['motion'][frame_idx]["t"].flatten())
            exp_list.append(motion_data['motion'][frame_idx]["exp"].flatten())
            pitch_list.append(motion_data['motion'][frame_idx]["pitch"].flatten())
            yaw_list.append(motion_data['motion'][frame_idx]["yaw"].flatten())
            roll_list.append(motion_data['motion'][frame_idx]["roll"].flatten())
    
    # 转换为 NumPy 数组
    scale_array = np.array(scale_list)  # (811, 1)
    R_array = np.array(R_list)          # (811, 9)
    t_array = np.array(t_list)          # (811, 3)
    exp_array = np.array(exp_list)      # (811, 63)
    pitch_array = np.array(pitch_list)  # (811, 1)
    yaw_array = np.array(yaw_list)      # (811, 1)
    roll_array = np.array(roll_list)    # (811, 1)
    print(scale_array.shape, R_array.shape, t_array.shape, exp_array.shape, pitch_array.shape, yaw_array.shape, roll_array.shape)

    # 处理 lip 和 eyes 特征   处理 c_lip_lst（嘴唇形状数据）和 c_eyes_lst（眼睛形状数据）。
    lip_lst_array = np.array([data.flatten() for data in motion_data['c_lip_lst']]).astype(np.float32)
    eyes_lst_array = np.array([data.flatten() for data in motion_data['c_eyes_lst']]).astype(np.float32)
    print(f"lip_aray: {lip_lst_array.shape}, eyes_lst_array: {eyes_lst_array.shape}")

    # abs max  计算各类统计信息 的 绝对最大值
    abs_max_scale = np.max(abs(scale_array), axis=0)
    abs_max_R = np.max(abs(R_array), axis=0)
    abs_max_t = np.max(abs(t_array), axis=0)
    abs_max_exp = np.max(abs(exp_array), axis=0)
    abs_max_pitch = np.max(abs(pitch_array), axis=0)
    abs_max_yaw = np.max(abs(yaw_array), axis=0)
    abs_max_roll = np.max(abs(roll_array), axis=0)
    abs_max_lip = np.max(abs(lip_lst_array), axis=0)
    abs_max_eyes = np.max(abs(eyes_lst_array), axis=0)
    print("absmax: ", abs_max_scale.shape, abs_max_R.shape, abs_max_t.shape, abs_max_exp.shape, abs_max_pitch.shape, abs_max_pitch.shape, abs_max_roll.shape, abs_max_lip.shape, abs_max_eyes.shape)

    # max 计算最大值
    max_scale = np.max(scale_array, axis=0)
    max_R = np.max(R_array, axis=0)
    max_t = np.max(t_array, axis=0)
    max_exp = np.max(exp_array, axis=0)
    max_pitch = np.max(pitch_array, axis=0)
    max_yaw = np.max(yaw_array, axis=0)
    max_roll = np.max(roll_array, axis=0)
    max_lip = np.max(lip_lst_array, axis=0)
    max_eyes = np.max(eyes_lst_array, axis=0)
    print("max: ", max_scale.shape, max_R.shape, max_t.shape, max_exp.shape, max_pitch.shape, max_pitch.shape, max_roll.shape, max_lip.shape, max_eyes.shape)

    # min 计算最小值
    min_scale = np.min(scale_array, axis=0)
    min_R = np.min(R_array, axis=0)
    min_t = np.min(t_array, axis=0)
    min_exp = np.min(exp_array, axis=0)
    min_pitch = np.min(pitch_array, axis=0)
    min_yaw = np.min(yaw_array, axis=0)
    min_roll = np.min(roll_array, axis=0)
    min_lip = np.min(lip_lst_array, axis=0)
    min_eyes = np.min(eyes_lst_array, axis=0)
    print("min: ", min_scale.shape, min_R.shape, min_t.shape, min_exp.shape, min_pitch.shape, min_pitch.shape, min_roll.shape, min_lip.shape, min_eyes.shape)

    # mean 计算均值
    mean_scale = np.mean(scale_array, axis=0)
    mean_R = np.mean(R_array, axis=0)
    mean_t = np.mean(t_array, axis=0)
    mean_exp = np.mean(exp_array, axis=0)     # (n_frames, 63)  -> (63)
    mean_pitch = np.mean(pitch_array, axis=0)
    mean_yaw = np.mean(yaw_array, axis=0)
    mean_roll = np.mean(roll_array, axis=0)
    mean_lip = np.mean(lip_lst_array, axis=0)
    mean_eyes = np.mean(eyes_lst_array, axis=0)
    print("mean: ", mean_scale.shape, mean_R.shape, mean_t.shape, mean_exp.shape, mean_pitch.shape, mean_yaw.shape, mean_roll.shape, mean_lip.shape, mean_eyes.shape)

    # std 计算标准差
    std_scale = np.std(scale_array, axis=0)
    std_R = np.std(R_array, axis=0)
    std_t = np.std(t_array, axis=0)
    std_exp = np.std(exp_array, axis=0)
    std_pitch = np.std(pitch_array, axis=0)
    std_yaw = np.std(yaw_array, axis=0)
    std_roll = np.std(roll_array, axis=0)
    std_lip = np.std(lip_lst_array, axis=0)
    std_eyes = np.std(eyes_lst_array, axis=0)
    print("std: ", std_scale.shape, std_R.shape, std_t.shape, std_exp.shape, std_pitch.shape, std_yaw.shape, std_roll.shape, std_lip.shape, std_eyes.shape)

    # 保存统计信息
    motion_template = {
        "mean_scale": mean_scale,      # 均值
        "mean_R": mean_R,
        "mean_t": mean_t,
        "mean_exp": mean_exp,
        "mean_pitch": mean_pitch,
        "mean_yaw": mean_yaw,
        "mean_roll": mean_roll,
        "mean_lip": mean_lip,
        "mean_eyes": mean_eyes,
        "std_scale": std_scale,        # 标准差
        "std_R": std_R,
        "std_t": std_t,
        "std_exp": std_exp, 
        "std_pitch": std_pitch, 
        "std_yaw": std_yaw, 
        "std_roll": std_roll, 
        "std_lip": std_lip,
        "std_eyes": std_eyes,
        "max_scale": max_scale,          # 最大值
        "max_R": max_R,
        "max_t": max_t,
        "max_exp": max_exp,
        "max_pitch": max_pitch,
        "max_yaw": max_yaw,
        "max_roll": max_roll,
        "max_lip": max_lip,
        "max_eyes": max_eyes,
        "min_scale": min_scale,          # 最小值
        "min_R": min_R,
        "min_t": min_t,
        "min_exp": min_exp,
        "min_pitch": min_pitch,
        "min_yaw": min_yaw,
        "min_roll": min_roll,
        "min_lip": min_lip,
        "min_eyes": min_eyes,
        "abs_max_scale": abs_max_scale,   # 绝对最大值
        "abs_max_R": abs_max_R,
        "abs_max_t": abs_max_t,
        "abs_max_exp": abs_max_exp,
        "abs_max_pitch": abs_max_pitch,
        "abs_max_yaw": abs_max_yaw,
        "abs_max_roll": abs_max_roll,
        "abs_max_lip": abs_max_lip,
        "abs_max_eyes": abs_max_eyes,
    }
    return motion_template

same_template = all_same_cal(f'front_all_motions.pkl')
pickle.dump(same_template, open(f"motion_template.pkl", 'wb'))
