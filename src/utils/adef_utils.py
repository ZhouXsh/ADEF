'''emotion2vec'''

'''
Using the finetuned emotion recognization model

rec_result contains {'feats', 'labels', 'scores'}
	extract_embedding=False: 9-class emotions with scores
	extract_embedding=True: 9-class emotions with scores, along with features

9-class emotions: 
iic/emotion2vec_plus_seed, iic/emotion2vec_plus_base, iic/emotion2vec_plus_large (May. 2024 release)
iic/emotion2vec_base_finetuned (Jan. 2024 release)
    0: angry
    1: disgusted
    2: fearful
    3: happy
    4: neutral
    5: other
    6: sad
    7: surprised
    8: unknown
'''
from funasr import AutoModel

model_id = "iic/emotion2vec_plus_large"

model = AutoModel(
    model=model_id,
    hub="ms",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
)

def extract_emo2vec(wav_file,output_dir):
    model.generate(wav_file, output_dir=output_dir, granularity="utterance", extract_embedding=True)
    '''
    key: file_name
    labels: emotion type  ['生气/angry', '厌恶/disgusted', '恐惧/fearful', '开心/happy', '中立/neutral', '其他/other', '难过/sad', '吃惊/surprised', '<unk>']
    scores: possibility of each emotion
    feats: emo_vector       shape:(1024,)    # (768,)
    '''

def single():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # extract_emo2vec('/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD25/raw_videos/M003/front/angry/level_3/M003_front_angry_level_3_001.wav', '.')

    np_path = '/mnt/disk2/zhouxishi/ADEF/src/modules/M003_front_angry_level_3_001.npy'

    emo_vec = torch.tensor(np.load(np_path)).unsqueeze(0).to(device)  # [1, 1024]

    a2e_model = AudioEmotionClassifierModel().to(device)
    dict = torch.load('/mnt/disk2/zhouxishi/ADEF/pretrained_weights/ADEF/audio2emo/audio2emo.pth',map_location=device)
    a2e_model.load_state_dict(dict)
    a2e_model.eval()

    outputs = a2e_model(emo_vec)
    print(outputs)

    argmax = outputs.argmax(dim=1).item()
    print(argmax)