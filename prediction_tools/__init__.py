from .data_utils import create_dataframe, create_dataframe_for_holters
from .dataset import ECGDataGenerator, HolterDataGenerator


data_generator = dict(tis=ECGDataGenerator, holters=HolterDataGenerator)
dataframe_creator = dict(tis=create_dataframe, holters=create_dataframe_for_holters)

