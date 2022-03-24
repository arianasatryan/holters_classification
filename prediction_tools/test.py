import os
import sys
import json
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix,\
    classification_report
from data_utils import train_val_test_split, dataframe_creator, data_generator


with open(sys.argv[1], "r") as f:
    config = json.load(f)


def test_model(config):
    # load data generator
    test_ecg_info = dataframe_creator[config['data_config']['source']](config)
    if not isinstance(config['data_config']['train_test_split_ratio'], type(None)) \
            and 0 < config['data_config']['train_test_split_ratio'] < 1:
        _, _, test_ecg_info = train_val_test_split(test_ecg_info,
                                                   test_ratio=config['data_config']['train_test_split_ratio'],
                                                   val_ratio=config['data_config']['train_test_split_ratio'],
                                                   SEED=config['SEED'])

    Generator = data_generator[config['data_config']['source']]
    test_gen = Generator(ecg_info=test_ecg_info,
                         batch_size=config['training_config']['batch_size'],
                         record_length=config['data_config']['ecg_length'],
                         frequency=config['data_config']['resampled_frequency'],
                         leads=config['data_config']['leads'],
                         pad_by=config['data_config']['pad_mode'],
                         scale_by=config['data_config']['scale_by'])

    # load model and predict on test generator
    model = load_model(config['model_path'], compile=False)
    print("START TESTING...")
    y_pred = model.predict(test_gen)

    # get labels as there might be artefact records which were discarded
    y_test_labels = test_gen.get_labels()
    y_pred = apply_threshold(y_pred)

    return y_test_labels, y_pred


def get_metrics(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    return {
        'accuracy': accuracy_score(y_test, y_pred).round(3),
        'recall': recall_score(y_test, y_pred, average="binary", pos_label=1).round(3),
        'precision': precision_score(y_test, y_pred, average="binary", pos_label=1).round(3),
        'f1-score': f1_score(y_test, y_pred, average="binary", pos_label=1).round(3),
    }, classification_report(y_test, y_pred, labels=labels)


def apply_threshold(y_pred):
    y_pred_labels = []
    for pred in y_pred:
        pred_label = []
        for sample in pred:
            label = 1 if sample >= 0.5 else 0
            pred_label.append(label)
        y_pred_labels.append(pred_label)
    return np.array(y_pred_labels)


y_test_labels, y_pred = test_model(config)

# compute model's quality and save results
labels = [1, 0] if len(list(config['merge_map'].keys())) == 1 else list(config['merge_map'].keys())
metrics, report = get_metrics(y_test_labels, y_pred, labels)

with open(os.path.join(config['results_path'], '{}_quality.json'.format(config['experiment_name'])), 'w') as f:
    json.dump(metrics, f, indent=4)

with open(os.path.join(config['results_path'], '{}_metrics_per_label.txt'.format(config['experiment_name'])), 'w') as f:
    f.write(report, labels=labels)

