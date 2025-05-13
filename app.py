import os
import tyro
import subprocess
import gradio as gr
import os.path as osp
import platform
from src.utils.helper import load_description
from src.gradio_pipeline import GradioPipeline, GradioPipelineAnimal
from src.config.crop_config import CropConfig
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig

if platform.system() == "Windows":
    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

# 检查ffmpeg
def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
if osp.exists(ffmpeg_dir):
    os.environ["PATH"] += (os.pathsep + ffmpeg_dir)
if not fast_check_ffmpeg():
    raise ImportError(
        "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
    )

# set tyro theme  设定tyro主题
tyro.extras.set_accent_color("bright_cyan")
args = tyro.cli(ArgumentConfig)

# specify configs for inference  指定推理的配置
inference_cfg = partial_fields(InferenceConfig, args.__dict__)  # use attribute of args to initial InferenceConfig
crop_cfg = partial_fields(CropConfig, args.__dict__)  # use attribute of args to initial CropConfig

############# Functions #################
if args.gradio_temp_dir not in (None, ''):
    os.environ["GRADIO_TEMP_DIR"] = args.gradio_temp_dir
    os.makedirs(args.gradio_temp_dir, exist_ok=True)

gradio_pipeline_human = GradioPipeline(
    inference_cfg=inference_cfg,
    crop_cfg=crop_cfg,
    args=args
)
def gpu_wrapped_execute_a2v(*args, **kwargs):
    print("args: ", args, args[5])
    # if args[5] == "animal":
    #     return None
    # else:
        
    return gradio_pipeline_human.execute_a2v(*args, **kwargs)

################# GUI ################
title_md = "assets/gradio/gradio_title.md"            # gradio标题渲染
example_reference_dir = "assets/myTest/image"        # 参考图像 目录
example_audio_dir = "assets/myTest/audio"          # 音频 目录

data_examples_a2v = [
    [osp.join(example_reference_dir, "大嘴生气.jpg"), osp.join(example_audio_dir, "0D_7PGxpV1M_6_7.m4a"), True, 2.8],
    [osp.join(example_reference_dir, "开心侧脸.jpg"), osp.join(example_audio_dir, "Af5hc1y22NA_0_0.m4a"), True, 2.8],
    [osp.join(example_reference_dir, "恐惧侧脸.jpg"), osp.join(example_audio_dir, "_cyL5vt3pMc_2_0.m4a"), True, 2.8],
    [osp.join(example_reference_dir, "悲伤.jpg"), osp.join(example_audio_dir, "LojCtuv1WiE_10_0.m4a"), True, 2.8],
    [osp.join(example_reference_dir, "白人女.jpg"), osp.join(example_audio_dir, "tTXd90pVjCg_7_0.m4a"), True, 2.8],
    [osp.join(example_reference_dir, "白人男.jpg"), osp.join(example_audio_dir, "60mHAO_qiUo_11_0.m4a"), True, 2.8],
    [osp.join(example_reference_dir, "证件女.jpg"), osp.join(example_audio_dir, "CvrulAgUdMU_8_0.m4a"), True, 2.8],
    [osp.join(example_reference_dir, "证件男.jpg"), osp.join(example_audio_dir, "-DqsmgME7yw_0_0.m4a"), True, 2.8],
    [osp.join(example_reference_dir, "黑人女.jpg"), osp.join(example_audio_dir, "P1266YbTBhg_3_2.m4a"), True, 2.8],
    [osp.join(example_reference_dir, "黑人男.jpg"), osp.join(example_audio_dir, "Vj1UuBG5RN8_27_0.m4a"), True, 2.8],
]

# 整个gradio渲染作为demo
with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Plus Jakarta Sans")])) as demo:
    gr.HTML(load_description(title_md))     # gradio标题渲染
    
    # Inputs & Outputs   # step1： 输入输出 渲染
    gr.Markdown(load_description("assets/gradio/gradio_description_upload.md"))
    with gr.Row():      # 行
        with gr.Accordion(open=True, label="🖼️ Reference Image"):   # 参考图像输入框
            input_image = gr.Image(type="filepath", width=512, label="Reference Image")
            gr.Examples(
                examples = [
                    [osp.join(example_reference_dir, "大嘴生气.jpg")],
                    [osp.join(example_reference_dir, "开心侧脸.jpg")],
                    [osp.join(example_reference_dir, "恐惧侧脸.jpg")],
                    [osp.join(example_reference_dir, "悲伤.jpg")],
                    [osp.join(example_reference_dir, "白人女.jpg")],
                    [osp.join(example_reference_dir, "白人男.jpg")],
                    [osp.join(example_reference_dir, "证件女.jpg")],
                    [osp.join(example_reference_dir, "证件男.jpg")],
                    [osp.join(example_reference_dir, "黑人女.jpg")],
                    [osp.join(example_reference_dir, "黑人男.jpg")],
                ],
                inputs=[input_image],
                cache_examples=False,
            )
        with gr.Accordion(open=True, label="🎵 Input Audio"):       # 音频 输入框
            input_audio = gr.Audio(type="filepath", label="Input Audio")
            gr.Examples(
                examples = [
                    [osp.join(example_audio_dir, "0D_7PGxpV1M_6_7.m4a")],
                    [osp.join(example_audio_dir, "Af5hc1y22NA_0_0.m4a")],
                    [osp.join(example_audio_dir, "_cyL5vt3pMc_2_0.m4a")],
                    [osp.join(example_audio_dir, "LojCtuv1WiE_10_0.m4a")],
                    [osp.join(example_audio_dir, "tTXd90pVjCg_7_0.m4a")],
                    [osp.join(example_audio_dir, "60mHAO_qiUo_11_0.m4a")],
                    [osp.join(example_audio_dir, "CvrulAgUdMU_8_0.m4a")],
                    [osp.join(example_audio_dir, "-DqsmgME7yw_0_0.m4a")],
                    [osp.join(example_audio_dir, "P1266YbTBhg_3_2.m4a")],
                    [osp.join(example_audio_dir, "Vj1UuBG5RN8_27_0.m4a")],
                ],
                inputs=[input_audio],
                cache_examples=False,
            )
        with gr.Accordion(open=True, label="🎬 Output Video",):     # 生成视频 输出框
            output_video = gr.Video(autoplay=False, interactive=False, width=512)


    # Configs          step 2：修改参数 渲染
    gr.Markdown(load_description("assets/gradio/gradio_description_configuration.md"))

    with gr.Column():   # 列
        with gr.Accordion(open=True, label="Key Animation Options"):
            with gr.Row():
                # animation_mode =gr.Radio(['human', 'animal'], value="human", label="Animation Mode") 
                flag_do_crop_input = gr.Checkbox(value=True, label="do crop (image)")
                cfg_scale = gr.Number(value=2.8, label="cfg_scale", minimum=0.0, maximum=10.0, step=0.5)
        with gr.Accordion(open=False, label="Optional Animation Options"):
            with gr.Row():
                driving_option_input = gr.Radio(['expression-friendly', 'pose-friendly'], value="pose-friendly", label="driving option")
                driving_multiplier = gr.Number(value=1.0, label="driving multiplier", minimum=0.0, maximum=2.0, step=0.02)
            with gr.Row():
                flag_normalize_lip = gr.Checkbox(value=True, label="normalize lip")
                flag_relative_motion = gr.Checkbox(value=True, label="relative motion",info="nmjj")    # 相对运动 （相对鼻子的运动）
                flag_remap_input = gr.Checkbox(value=False, label="paste-back")
                flag_stitching_input = gr.Checkbox(value=False, label="stitching")
        with gr.Accordion(open=False, label="Optional Options for Reference Image"):
            with gr.Row():
                scale = gr.Number(value=3.0, label="image crop scale", minimum=1.8, maximum=4.0, step=0.05)
                vx_ratio = gr.Number(value=0.0, label="image crop x", minimum=-0.5, maximum=0.5, step=0.01)
                vy_ratio = gr.Number(value=-0.125, label="image crop y", minimum=-0.5, maximum=0.5, step=0.01)

    # Generate      step 3：生成
    gr.Markdown(load_description("assets/gradio/gradio_description_generate.md"))   # 生成渲染
    with gr.Row():
        process_button_generate = gr.Button("🚀 Generate", variant="primary")   # generate按钮
    

    # Examples   给的样例，不必理会
    gr.Examples(
        examples=data_examples_a2v,     # 加载参考图像列表
        inputs=[input_image,
                input_audio,
                # animation_mode, 
                flag_do_crop_input,
                cfg_scale,
                ],
        outputs=[output_video],
        cache_examples=False
    )

    # Binding Functions for Buttons  按钮绑定功能
    generation_func = gpu_wrapped_execute_a2v
    process_button_generate.click(                # 点击generate按钮
        fn=generation_func,    # 点击后执行的函数
        inputs=[
            input_image,        # 视频输入框
            input_audio,        # 音频输入框
            flag_normalize_lip,        # True
            flag_relative_motion,      # True
            driving_multiplier,        # 1 全局倍率
            driving_option_input,      # 'expression-friendly'  ~~~driving_option ： “expression-friendly” 将使用全局倍率来调整驱动动作，并可用于源为人像的情况。
            flag_do_crop_input,        # False    ~~~flag_do_crop     是否将源肖像或视频裁剪到 面部裁剪空间
            scale,                     # 比例越大，人脸面积比越小      2.3
            vx_ratio,                  # 在裁剪空间中将 脸 向左或向右移动的比率  0
            vy_ratio,                  # 在裁剪空间中向上或向下移动 脸 的比率  -0.125
            flag_stitching_input,      # False     ~~~flag_stitching   如果头部运动较小，建议为True，如果头部运动较大或源图像是动物，建议为False
            flag_remap_input,          # False     ~~~flag_pasteback   是否将动画面部裁剪从面部裁剪空间 粘贴/缝合到 原始图像空间
            cfg_scale,                 # CFG的因子         4
        ],
        outputs=[
            output_video,    # 显示到 -> 生成视频输出框
        ],
        show_progress=True   # 进度条
    )

demo.launch(       # 启动gradio
    server_port=args.server_port,
    share=args.share,
    server_name=args.server_name
)