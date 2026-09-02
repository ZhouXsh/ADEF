#  coding:utf-8
import subprocess
import os
import sys
import threading


def main(frame_dir, face_dir, n_thread):

    threads = []
    # function
    func_path = './face_alignment_code/lib/face_align_cuda.py'
    # Model
    predictor_path      = './face_alignment_code/lib/shape_predictor_5_face_landmarks.dat'
    cnn_face_detector   = './face_alignment_code/lib/mmod_human_face_detector.dat'

    for frame_file in os.listdir(frame_dir):
        frame_root_folder = os.path.join(frame_dir, frame_file)
        face_root_folder = frame_root_folder.replace(frame_dir, face_dir)
        if os.path.isdir(frame_root_folder):
            makefile(face_root_folder)
            threads.append(threadFun(frame2face, (func_path, predictor_path, frame_root_folder, face_root_folder, cnn_face_detector)))
    run_threads(threads, n_thread)
    print('all is over')


def makefile(file_dir):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)


def run_threads(threads, n_thread):
    used_thread = []
    for num, new_thread in enumerate(threads):
        print('thread index: {:}'.format(num), end=' \t')
        new_thread.start()
        used_thread.append(new_thread)

        if num % n_thread == 0:
            for old_thread in used_thread:
                old_thread.join()
            used_thread = []


class threadFun(threading.Thread):
    def __init__(self, func, args):
        super(threadFun, self).__init__()
        self.fun = func
        self.args = args
    def run(self):
        self.fun(*self.args)


# python ./lib/face_align_cuda.py ./lib/shape_predictor_5_face_landmarks.dat /data4/AFEW/train_frame/Neutral/002020400 /data4/AFEW/train_face/Neutral/002020400 ./lib/mmod_human_face_detector.dat 0

def frame2face(func_path, predictor_path, image_root_folder, save_root_folder, cnn_face_detector, gpu_id=0):

    # Use the *current* Python interpreter so dlib/torch resolve via the venv,
    # and forward PYTHONPATH because the relative imports inside
    # `face_align_cuda.py` need `code/` on sys.path.
    here = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.dirname(os.path.dirname(here))   # .../evaluation_eat/code
    env = os.environ.copy()
    env["PYTHONPATH"] = code_dir + os.pathsep + env.get("PYTHONPATH", "")
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, func_path, predictor_path,
        image_root_folder, save_root_folder,
        cnn_face_detector, str(gpu_id),
    ]
    print('{:}'.format(image_root_folder))
    # print('cmd:', ' '.join(cmd))
    subprocess.call(cmd, shell=False, env=env)


if __name__ == '__main__':
    frame_dir_train_afew = '/data4/AFEW/train_frame/'
    face_dir_train_afew  = '/data4/AFEW/train_face/'
    frame_dir_val_afew = '/data4/test_data_for_emotion-fan/test2_frame/'
    face_dir_val_afew  = '/data4/test_data_for_emotion-fan/test2_face/'
    # main(frame_dir_train_afew, face_dir_train_afew, n_thread=20)
    main(frame_dir_val_afew, face_dir_val_afew, n_thread=1)