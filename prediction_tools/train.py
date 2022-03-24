from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy, Recall, Precision
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger, EarlyStopping)
from prediction_tools.generate_class_weights import get_class_weights
from prediction_tools.data_utils import train_val_test_split
from prediction_tools import data_generator, dataframe_creator
from model import get_model


def train_model(config):
    optimizer = Adam(config['training_config']["lr"])
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=7,
                                   min_lr=config['training_config']["lr"] / 100),
                 EarlyStopping(patience=9,  # Patience should be larger than the one in ReduceLROnPlateau
                               min_delta=0.00001)]

    # load data generator
    # print(config['data_config']['source'])
    ecg_info = dataframe_creator[config['data_config']['source']](config)
    if not isinstance(config['data_config']['train_test_split_ratio'], type(None)) \
            and 0 < config['data_config']['train_test_split_ratio'] < 1:
        train_ecg_info, val_ecg_info, _ = train_val_test_split(ecg_info,
                                                               test_ratio=config['data_config']['train_test_split_ratio'],
                                                               val_ratio=config['data_config']['train_test_split_ratio'],
                                                               SEED=config['SEED'])
    else:
        train_ecg_info, val_ecg_info, _ = train_val_test_split(ecg_info, val_ratio=0.2)
    print("train_ecg_info.shape", train_ecg_info.shape, "val_ecg_info.shape", val_ecg_info.shape)
    Generator = data_generator[config['data_config']['source']]
    train_gen = Generator(ecg_info=train_ecg_info,
                          batch_size=config['training_config']['batch_size'],
                          record_length=config['data_config']['ecg_length'],
                          frequency=config['data_config']['resampled_frequency'],
                          leads=config['data_config']['leads'],
                          pad_by=config['data_config']['pad_mode'],
                          scale_by=config['data_config']['scale_by'])

    val_gen = Generator(ecg_info=val_ecg_info,
                        batch_size=config['training_config']['batch_size'],
                        record_length=config['data_config']['ecg_length'],
                        frequency=config['data_config']['resampled_frequency'],
                        leads=config['data_config']['leads'],
                        pad_by=config['data_config']['pad_mode'],
                        scale_by=config['data_config']['scale_by'])

    class_weights = get_class_weights(train_ecg_info)
    # print("class_weights", class_weights)

    # use this to train from the beginning
    model = get_model(n_classes=len(config['merge_map'].keys()), leads_number=len(config['data_config']['leads']),
                      points=config['data_config']['resampled_frequency'] * config['data_config']['ecg_length'],
                      last_layer='sigmoid')
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[Accuracy(), Recall(), Precision()])

    # use this to continue training
    # model = load_model(config['model_path'], compile=True)

    # Create log

    callbacks += [TensorBoard(log_dir=f'{config["results_path"]}/logs', write_graph=False),
                  CSVLogger(f'{config["results_path"]}/training.log', append=False)]  # Change append to true if continuing training
    # Save the BEST and LAST model
    callbacks += [ModelCheckpoint(f'{config["results_path"]}/backup_model_last.hdf5'),
                  ModelCheckpoint(f'{config["results_path"]}/backup_model_best.hdf5', save_best_only=True)]
    # Train neural network
    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=config['training_config']["epochs"],
                        initial_epoch=0,  # If you are continuing an interrupted section change here
                        class_weight=class_weights,
                        callbacks=callbacks,
                        verbose=1)
    # Save final result
    model.save(f"{config['model_path']}", include_optimizer=True)


