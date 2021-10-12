import unidecode
import re
from nltk import ngrams
import numpy as np


def remove_accent(text):
    return unidecode.unidecode(text)

def extract_phrases(text):
    return re.findall(r'\w[\w ]+', text)

# gen 5-grams
def gen_ngrams(words, n=5):
    return ngrams(words.split(), n)


def compute_accuracy(ground_truth, predictions, mode='full_sequence'):
    """
    Computes accuracy
    :param ground_truth:
    :param predictions:
    :param display: Whether to print values to stdout
    :param mode: if 'per_char' is selected then
                 single_label_accuracy = correct_predicted_char_nums_of_single_sample / single_label_char_nums
                 avg_label_accuracy = sum(single_label_accuracy) / label_nums
                 if 'full_sequence' is selected then
                 single_label_accuracy = 1 if the prediction result is exactly the same as label else 0
                 avg_label_accuracy = sum(single_label_accuracy) / label_nums
    :return: avg_label_accuracy
    """
    if mode == 'per_char':

        accuracy = []

        for index, label in enumerate(ground_truth):
            prediction = predictions[index]
            total_count = len(label)
            correct_count = 0
            try:
                for i, tmp in enumerate(label):
                    if tmp == prediction[i]:
                        correct_count += 1
            except IndexError:
                continue
            finally:
                try:
                    accuracy.append(correct_count / total_count)
                except ZeroDivisionError:
                    if len(prediction) == 0:
                        accuracy.append(1)
                    else:
                        accuracy.append(0)
        avg_accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
    elif mode == 'full_sequence':
        try:
            correct_count = 0
            for index, label in enumerate(ground_truth):
                prediction = predictions[index]
                if np.all(prediction == label):
                    correct_count += 1
            avg_accuracy = correct_count / len(ground_truth)
        except ZeroDivisionError:
            if not predictions:
                avg_accuracy = 1
            else:
                avg_accuracy = 0

    elif mode == 'word':
        accuracy = []

        for index, label in enumerate(ground_truth):
            prediction = predictions[index]

            gt_word_list = label.split(' ')
            pred_word_list = prediction.split(' ')

            for i, gt_w in enumerate(gt_word_list):
                if i > len(pred_word_list) - 1:
                    accuracy.append(0)
                    continue

                pred_w = pred_word_list[i]

                if pred_w == gt_w:
                    accuracy.append(1)
                else:
                    accuracy.append(0)

        avg_accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
    else:
        raise NotImplementedError('Other accuracy compute mode has not been implemented')

    return avg_accuracy


def batch_to_device(text, tgt_input, tgt_output, tgt_padding_mask, device):
    text = text.to(device, non_blocking=True)
    tgt_input = tgt_input.to(device, non_blocking=True)
    tgt_output = tgt_output.to(device, non_blocking=True)
    tgt_padding_mask = tgt_padding_mask.to(device, non_blocking=True)

    return text, tgt_input, tgt_output, tgt_padding_mask

def get_bucket(w):
    bucket_size = (w // 5) * 5

    return bucket_size