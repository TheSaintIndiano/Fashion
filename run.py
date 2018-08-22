__author__ = 'indiano'

import argparse
from multiprocessing import Process

from models.classical.model_run_pipeline import ModelRunPipeline
from models.classical.models import Models
from utils.data_utils import DataUtils

'''
python run.py --root_dir "./" --raw_data_dir "../../data/raw/sportswear/events" --hdf_file "../../data/hdf/sportswear" 
--checkpoint_dir "../../checkpoint/classical" --save_dir "../../save/classical" --model xgb --test_size 33
'''

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data and model checkpoints directories
parser.add_argument('--root_dir', type=str, default='./',
                    help='root directory of the project')
parser.add_argument('--raw_data_dir', type=str, default='../../data/raw/sportswear/events',
                    help="""name of raw events folder if the hdf file not generated or
                    data directory containing input with training examples.""")
parser.add_argument('--hdf_file', type=str, default='../../data/hdf/sportswear',
                    help='stored or new hdf filename without .hdf extension.')
parser.add_argument('--checkpoint_dir', type=str, default='../../checkpoint/classical',
                    help='directory to store checkpointed models.')
parser.add_argument('--save_dir', type=str, default='../../save/classical',
                    help='directory to store graphs & plots')

# Model params
parser.add_argument('--model', type=str, default='xgb',
                    help='xgb, naivebayes, knn or svc (Support Vector).')

# Optimization
parser.add_argument('--test_size', type=int, default=33,
                    help="""% of total data equals the test size for train test split.
                     Please enter a value between 0-100.""")

# Parsed/collected all the arguments
args = parser.parse_args()


def train(args):
    """
    Training using pipeline module
    :param args:
    :return:
    """

    # Preparing dataset
    data = DataUtils(args)
    # Generating & saving dataframe from raw event folder
    if args.hdf_file:
        print('Raw data files will not be processed.\n')
        print('{} file present.'.format(args.hdf_file))
        pass
    elif args.raw_data_dir:
        print('Processing Raw data files.\n')
        print('A hdf file will be generated for further exploration.')
        data.load_txt_files()
    data = data.prepare_data()

    # Creating pipelines
    model_pipe = Models(args)
    model_pipelined, model_name = model_pipe.model_pipeline()
    print('Pipeline created for model {}'.format(model_name))

    # Running a model pipeline
    model = ModelRunPipeline(args, model_pipelined, data)

    # Multiprocessing to spawn processes using an API similar to threading module
    proc = Process(target=model.run_pipeline, args=())

    proc.start()
    proc.join()

    print('\n\n****************** Classification done. Enjoy Life. :) *******************')


if __name__ == '__main__':
    train(args)
