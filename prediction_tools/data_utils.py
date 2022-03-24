import ast
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from datetime import timedelta, datetime


def correct_types(df, task_type):
    df['ecg_shape'] = df['ecg_shape'].apply(lambda x: ast.literal_eval(x))
    if task_type in df.columns:
        df[task_type] = df[task_type].apply(lambda x: ast.literal_eval(x))
    if 'startdate' in df.columns:
        df['startdate'] = df['startdate'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    if 'enddate' in df.columns:
        df['enddate'] = df['enddate'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    if 'intervals' in df.columns:
        df['intervals'] = df['intervals'].apply(lambda l: [datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
                                                           for x in ast.literal_eval(l)])
    return df


def create_dataframe(config, target_col=True):
    print("CREATING DATAFRAME...")
    data_config = config['data_config']
    ecg_info = pd.DataFrame()
    for directory in data_config[data_config['source']]:
        ecg_info = pd.concat([pd.read_csv(directory)])

    ecg_info = correct_types(ecg_info, config['task_type'])
    one_hot = make_onehot(ecg_info, config['task_type'], config['prediction_classes'])
    if 'merge_map' in config.keys():
        one_hot = merge_columns(df=one_hot, merge_map=config['merge_map'])

    if target_col:
        ecg_info['target'] = [l[0] for l in one_hot.values.tolist()]

    # ecg_info.to_csv(f'{data_config["source"]}current.csv', index=False)
    return ecg_info


def create_dataframe_for_holters(config, target_col=True):
    print("CREATING HOLTER'S DATAFRAME...")
    data_config = config['data_config']
    ecg_info = pd.DataFrame()
    for directory in data_config[data_config['source']]:
        ecg_info = pd.concat([pd.read_csv(directory)])

    ecg_info = correct_types(ecg_info, config['task_type'])
    df_cuts = pd.DataFrame(columns=ecg_info.columns)
    for i, row in ecg_info.iterrows():
        cuts_count = row['ecg_shape'][1] // (data_config['ecg_length'] * data_config['resampled_frequency'])
        for start_sec in range(0, cuts_count):

            # check if the cut is artefact
            record = np.load(row['fpath'])["arr_0"]
            row['split_number'] = int(start_sec)
            is_artefact = sum(np.apply_along_axis(lambda x: len(set(x)) == 1, 0, record))
            if is_artefact:
                continue

            cut_start = row['startdate'] + timedelta(seconds=int(start_sec) * data_config['ecg_length'])
            cut_end = cut_start + timedelta(seconds=data_config['ecg_length'])
            if target_col:
                row['target'] = int(get_label(cut_start, cut_end, row['intervals'], overlap=data_config['overlap']))
            df_cuts = pd.concat([df_cuts, row])

    # df_cuts.to_csv(f'{data_config["source"]}current.csv')
    return df_cuts


def preprocess_holters_info(df, sec, freq, overlap, get_labels=False):
    points = sec * freq
    df_cuts = pd.DataFrame(columns=df.columns)
    for i, row in df.iterrows():
        cuts_count = row['ecg_shape'][1] // points
        for start_sec in range(0, cuts_count):
            row['split_number'] = int(start_sec)
            cut_start = row['startdate'] + timedelta(seconds=int(start_sec) * sec)
            cut_end = cut_start + timedelta(seconds=sec)
            if get_labels:
                row['label'] = get_label(cut_start, cut_end, row['intervals'], overlap=overlap)
            df_cuts = pd.concat([df_cuts, row])
    return df_cuts


def get_label(start_datetime, end_datetime, result_time_points, overlap):
    acceptable_overlap = timedelta(seconds=overlap)
    for res_start_datetime, res_end_datetime in zip(result_time_points[0::2], result_time_points[1::2]):
        delta = min(res_end_datetime, end_datetime) - max(start_datetime, res_start_datetime)
        if delta >= acceptable_overlap:
            return True
    return False


def merge_columns(df, merge_map):
    """
    Logical OR for given one-hot columns

    :param df: input dataframe
    :param merge_map: dictionary: key - name after merge, value - list of columns to be merged
    :return: pandas DataFrame
    """
    for k, v in merge_map.items():
        tmp = df[v].apply(any, axis=1).astype(int)
        df.drop(columns=v, inplace=True)
        df[k] = tmp
    return df


def make_onehot(ecg_df, task_type, predicted_classes=None):
    mlb = MultiLabelBinarizer()
    one_hot = pd.DataFrame(mlb.fit_transform(ecg_df[task_type]), columns=mlb.classes_)
    if predicted_classes:
        drop_cols = set(one_hot.columns) - set(predicted_classes)
        one_hot.drop(columns=drop_cols, inplace=True)
    return one_hot


def train_val_test_split(df, test_ratio=0.2, val_ratio=0.2, SEED=1):
    train_df = df
    val_df, test_df = None, None

    if test_ratio is not None:
        train_df, test_df = train_test_split(train_df, test_size=test_ratio, random_state=SEED)
        test_df = test_df.reset_index(drop=True)

    if test_ratio is not None:
        # if test_ratio is not None:
        #     val_ratio = val_ratio / (1 - test_ratio)
        train_df, val_df = train_test_split(train_df, test_size=val_ratio, random_state=SEED)
        val_df = val_df.reset_index(drop=True)

    train_df = train_df.reset_index(drop=True)
    return train_df, val_df, test_df
