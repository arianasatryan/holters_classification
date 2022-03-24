from tensorflow.keras.models import load_model
from dataset import Record
import numpy as np
import sys
import json

with open(sys.argv[1], "r") as f:
    config = json.load(f)

record_path = sys.argv[2]


def predict(config, record_path):
    # loading data
    predict_record = Record(record_length=config['data_config']['ecg_length'],
                            frequency=config['data_config']['resampled_frequency'],
                            json_file=record_path,
                            scale_by=config['data_config']['scale_by']).get_record()

    # load pretrained model
    model = load_model(config['model_path'], compile=False)
    y_pred = model.predict(predict_record)

    # get labels as there might be artefact records which were discarded
    y_pred = apply_threshold(y_pred)
    print("Result:", np.squeeze(y_pred))
    return np.squeeze(y_pred)


def apply_threshold(y_pred, threshold=0.5):
    y_pred_labels = []
    for pred in y_pred:
        pred_label = []
        for sample in pred:
            label = 1 if sample >= threshold else 0
            pred_label.append(label)
        y_pred_labels.append(pred_label)
    return np.array(y_pred_labels)


predict(config, record_path)


