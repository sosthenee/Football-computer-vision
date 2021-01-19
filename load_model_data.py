import os
import tarfile
import urllib.request


import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder

def load():
    DATA_DIR = os.path.join(os.getcwd(), 'data')
    MODELS_DIR = os.path.join(DATA_DIR, 'models')
    for dir in [DATA_DIR, MODELS_DIR]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    NUM_CLASSES = 90

    # Download and extract model
    MODEL_DATE = '20200711'
    MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8'
    MODEL_TAR_FILENAME = MODEL_NAME + '.tar.gz'
    MODELS_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/tf2/'
    MODEL_DOWNLOAD_LINK = MODELS_DOWNLOAD_BASE + MODEL_DATE + '/' + MODEL_TAR_FILENAME
    PATH_TO_MODEL_TAR = os.path.join(MODELS_DIR, MODEL_TAR_FILENAME)
    PATH_TO_CKPT = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'checkpoint/'))
    PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'pipeline.config'))
    if not os.path.exists(PATH_TO_CKPT):
        print('Downloading model. This may take a while... ', end='')
        urllib.request.urlretrieve(MODEL_DOWNLOAD_LINK, PATH_TO_MODEL_TAR)
        tar_file = tarfile.open(PATH_TO_MODEL_TAR)
        tar_file.extractall(MODELS_DIR)
        tar_file.close()
        os.remove(PATH_TO_MODEL_TAR)
        print('Done')

    # Download labels file
    LABEL_FILENAME = 'mscoco_label_map.pbtxt'
    LABELS_DOWNLOAD_BASE = \
        'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))
    if not os.path.exists(PATH_TO_LABELS):
        print('Downloading label file... ', end='')
        urllib.request.urlretrieve(LABELS_DOWNLOAD_BASE + LABEL_FILENAME, PATH_TO_LABELS)
        print('Done')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging


    tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
    #
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

    #PATH_TO_LABELS = 'data/soccer_label_map.pbtxt'
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    #print(categories)
    return detection_model, categories ,category_index;
