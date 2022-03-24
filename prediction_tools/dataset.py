from tensorflow.keras.utils import Sequence
import numpy as np
import json
import math
from scipy import signal
#from pyedflib import highlevel


def check_artefact(arr):
    return len(set(arr)) == 1


def resample(arr, arr_sampling_rate, sampling_rate):
    return signal.resample(arr, int(len(arr) * sampling_rate / arr_sampling_rate))


def znorm(arr):
    return (arr - arr.mean()) / arr.std()


class DataGenerator(Sequence):
    def __init__(self, ecg_info,
                 batch_size,
                 record_length,
                 frequency,
                 leads,
                 pad_by,
                 scale_by,
                 return_target=True):

        self.x = ecg_info.reset_index(drop=True)
        self.return_target = return_target
        if not self.return_target:
            self.x['target'] = None
        self.batch_size = batch_size
        self.rec_len = record_length
        self.freq = frequency
        self.leads = leads

        self.pad_by = pad_by
        self.scale_by = scale_by

        self.y = []
        self.first = True
        self.return_target = return_target

    def get_labels(self):
        return np.asarray(self.y)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def preprocess(self, record, recfreq=None):
        # this requires record shape of (number_of_points, number_of_channels)
        if recfreq is not None and recfreq != self.freq:
            record = np.apply_along_axis(resample, 0, record, recfreq, self.freq)
        if self.scale_by:
            record = np.apply_along_axis(znorm, 0, record)

        len_ = min(len(record), self.rec_len * self.freq)
        record = record[:len_]
        if len(record) < self.rec_len * self.freq:
            record = np.pad(record, ((0, self.rec_len * self.freq - record.shape[0]), (0, 0)), mode=self.pad_by)
        return record

    def _load_data(self, df_batch):
        print("parent")
        pass

    def _get_record(self, fpath):
        self.format = fpath[-4:]
        assert self.format is not None
        # assert self.format in ['.bdf', '.npz'], 'The input format must be either ".npz" or ".bdf"'
        assert self.format in ['.npz'], 'The input format must be either ".npz"'
        if self.format == '.npz':
            record = np.load(fpath)["arr_0"]
        # elif self.format == '.bdf':
        #     signals, signal_headers, header = highlevel.read_edf(fpath)
        #     # startdate = header['startdate']
        #     record = []
        #     for signal, signal_header in zip(signals, signal_headers):
        #         if 'EcgW' in signal_header['label']:
        #             record.append(signal)
        #     record = np.asarray(record)
        return record[self.leads, ]

    def __getitem__(self, idx):
        print("batch", idx)
        end = min(self.x.shape[0], (idx + 1) * self.batch_size)
        df_batch = self.x[idx * self.batch_size:end]
        X, Y = self._load_data(df_batch)

        if not self.first:
            self.y.extend(Y)
        self.first = False

        if not self.return_target:
            return np.stack(X)
        return np.stack(X), np.asarray(Y)


class ECGDataGenerator(DataGenerator):
    def _load_data(self, df_batch):
        X, Y = [], []
        for fpath, target, frequency in zip(df_batch.fpath, df_batch.target, df_batch.frequency):
            record = self._get_record(fpath)
            record = record.transpose()

            is_artefact = sum(np.apply_along_axis(check_artefact, 0, record))
            if not is_artefact:
                Y.append(target)
                X.append(self.preprocess(record, frequency))
        return X, Y


class HolterDataGenerator(DataGenerator):
    def _load_data(self, df_batch):
        X, Y = [], []
        for fpath, split_number, target, frequency in zip(df_batch.fpath, df_batch.split_number,
                                                          df_batch.target, df_batch.frequency):
            record = self._get_record(fpath)
            start_point = int(self.rec_len * frequency * split_number)
            end_point = int(self.rec_len * frequency * (split_number + 1))
            record = record[self.leads, start_point:end_point]
            record = record.transpose()
            is_artefact = sum(np.apply_along_axis(check_artefact, 0, record))
            if not is_artefact:
                Y.append(target)
                X.append(self.preprocess(record, frequency))
        return X, Y


class Record:
    def __init__(self, record_length, frequency, json_file, scale_by):
        self.rec_len = record_length
        self.freq = frequency
        self.scale_by = scale_by
        self.json_file = json_file
        with open(self.json_file, 'r') as fin:
            self.data_dict = json.load(fin)
        self.leads_order = ["I", 'II', 'III', 'IV']

    def get_leads_names(self):
        return [lead['name'] for lead in self.data_dict['datasets']]

    def get_freq(self):
        freq = [lead_data['frequency'] for lead_data in self.data_dict['datasets']]
        assert len(set(freq)) == 1, "The frequency of leads of the record differ"
        return freq[0]

    def get_record(self):
        leads_data = sorted(self.data_dict['datasets'], key=lambda i: i['name'])
        arr = np.array([lead_data['data'] for lead_data in leads_data])
        record_freq = self.get_freq()
        record = (arr[:, :int(record_freq * self.rec_len)]).transpose()
        record = self.preprocess(record, recfreq=record_freq)
        record = np.expand_dims(record, axis=0)
        return record

    def preprocess(self, record, recfreq=None):
        # this requires record shape of (number_of_points, number_of_channels)
        is_artefact = sum(np.apply_along_axis(check_artefact, 0, record))
        assert is_artefact == 0, "One of the record leads is damaged/artefact"
        if recfreq is not None and recfreq != self.freq:
            record = np.apply_along_axis(resample, 0, record, recfreq, self.freq)
        if self.scale_by:
            record = np.apply_along_axis(znorm, 0, record)

        len_ = min(len(record), self.rec_len * self.freq)
        record = record[:len_]
        if len(record) < self.rec_len * self.freq:
            record = np.pad(record, ((0, self.rec_len * self.freq - record.shape[0]), (0, 0)), mode=self.pad_by)
        return record




