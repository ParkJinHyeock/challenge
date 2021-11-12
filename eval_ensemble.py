from glob import glob
import tensorflow as tf
import json

from transforms import *
from utils import *
from data_utils import *

from sj_train import get_model, ARGS, random_merge_aug, stereo_mono, stft_filter, label_downsample_model
from metrics import Challenge_Metric, output_to_metric, get_er, evaluate, evaluate_ensemble
import os
import numpy as np

def minmax_log_on_mel(mel, labels=None):
    # batch-wise pre-processing
    axis = tuple(range(1, len(mel.shape)))

    # MIN-MAX
    mel_max = tf.math.reduce_max(mel, axis=axis, keepdims=True)
    mel_min = tf.math.reduce_min(mel, axis=axis, keepdims=True)
    mel = safe_div(mel-mel_min, mel_max-mel_min)

    # LOG
    mel = tf.math.log(mel + EPSILON)

    if labels is not None:
        return mel, labels
    return mel


def second2frame(seconds: list, frame_num, resolution):
    # seconds = [[class, start, end], ...]
    frames = np.zeros([frame_num, 3], dtype=np.float32)
    for second in seconds:
        class_num = second[0]
        start = int(np.round(second[1] * resolution))
        end = int(np.round(second[2] * resolution))
        frames[start:end,class_num] += 1
    return tf.convert_to_tensor(frames, dtype=tf.float32)



if __name__ == "__main__":
    config = ARGS()
    config.args.add_argument('--verbose', help='verbose', type=bool, default=True)
    config.args.add_argument('--p', help='parsing name', action='store_true')
    config.args.add_argument('--path', type=str, default='')
    config = config.get()
    model_name_list = [glob('./saved_model/*')[1]]
    model_list = []
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    preds_list = []
    wav_list = sorted(glob(config.path + '/*.wav'))
    preds_all_list = []
    if config.p:
        for item in model_name_list:
            name = os.path.basename(item).split('.h5')[0]  
            parsed_name = name.split('_')
            if parsed_name[0][0] not in ('B', 'v'):
                parsed_name = parsed_name[1:]
            if parsed_name[0] == 'vad':
                config.model_type = 'vad'
                config.model = 1
            else:
                config.model = int(parsed_name[0][-1])
            config.v = int(parsed_name[1][-1])
            config.n_mels = int(parsed_name[6][3:])
            config.n_chan = int(parsed_name[7][-1])
            config.n_frame = int(parsed_name[9].split('framelen')[-1])
            model = get_model(config)
            model.load_weights(os.path.join('./saved_model', f'{name}.h5'))
            pred_list = []
            for path in wav_list:
                pred_list.append(evaluate_ensemble(config, model, path))
            preds_all_list.append(pred_list) 
    ensemble_ = []
    for i in range(len(wav_list)):
        ensemble_temp = tf.zeros((len(preds_all_list[0][i]),3))
        for item in preds_all_list:
            ensemble_temp = ensemble_temp + item[i]
        ensemble_temp = ensemble_temp / len(preds_all_list)
        ensemble_.append(ensemble_temp)
    with open('sample_answer.json') as f:
        answer_gt = json.load(f)
    answer_gt = answer_gt['task2_answer']
    sr = 16000
    hop = 256
    metric = Challenge_Metric()
    smoothing_kernel_size = int(0.5 * sr) // hop # 0.5초 길이의 kernel
    final_score =[]
    for ensemble, path in zip(ensemble_, wav_list):
        ensemble = tf.keras.layers.AveragePooling1D(smoothing_kernel_size, 1, padding='same')(ensemble[tf.newaxis, ...])[0]
        ensemble = tf.keras.layers.MaxPooling1D(smoothing_kernel_size * 4, 1, padding='same')(ensemble[tf.newaxis, ...])[0]
        ensemble = tf.cast(ensemble >= 0.5, tf.float32)
        cls0, cls1, cls2 = metric.get_start_end_frame(ensemble)
        answer_gt_temp = tf.convert_to_tensor(answer_gt[os.path.basename(path)[:-4]])
        answer_predict = output_to_metric(hop, sr)(cls0, cls1, cls2)
        print(answer_predict)
        er = get_er(answer_gt_temp, answer_predict)
        final_score.append(er)
    print(f'Final score is {np.mean(final_score)}')