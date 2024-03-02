import subprocess
import argparse
import os
from io import StringIO

from scipy.io.arff import loadarff
import arff
import pandas as pd

from data.filtered_dataset import FilteredDataset

def load_dataset(path):
    record_array, meta = loadarff(path)
    dataset = pd.DataFrame(record_array, columns=meta.names())
    
    for col, dtype in zip(dataset.columns, meta.types()):
        type_map = {'numeric': float, 'nominal':str}
        dataset[col] = dataset[col].astype(type_map[dtype])

    return dataset

def write_dataset(data, output_file):
    # retreived from https://stackoverflow.com/questions/48993918/exporting-dataframe-to-arff-file-python, written by Simon Provost
    attributes = [(j, 'REAL') if data[j].dtypes == 'float64' else (j, data[j].unique().astype(str).tolist()) for j in data]
    arff_dic = {
        'attributes': attributes,
        'data': data.values,
        'relation': 'gas',
        'description': ''
    }
    with open(output_file, "w", encoding="utf8") as f, StringIO() as tmp:
        arff.dump(arff_dic, tmp)
        f.write(tmp.getvalue().replace('"', "'"))  # switch the quote type because SciPy doesn't like double quotes for some reason


REPOSITORY_ROOT = './external_repos/ML-NIDS-for-SCADA'
PRE_PROCESS_CMD = lambda path, type, args: f'python {REPOSITORY_ROOT}/src/preprocess-data_{type}.py -d {path} --split 0.6 0.2 0.2 --encode-function '+args

STD_DIR = 'normal-datasets'
TS_DIR = 'time-series-datasets'        

if __name__ == '__main__':
    # arg parsing
    parser = argparse.ArgumentParser(description='Replicates experiments of Kus et al.')
    parser.add_argument('-r', '--ml-nids-root', type=str, help='Root directory of ML-NIDS-for-SCADA git repository')
    args = parser.parse_args()
    if args.ml_nids_root:
        REPOSITORY_ROOT = args.r

    raw_dataset = load_dataset(f'{REPOSITORY_ROOT}/data/IanArffDataset.arff')
    dataset = FilteredDataset(raw_dataset, category_col_name='categorized result', attack_col_name='specific result')
    variants = dataset.get_all_variants()

    if not os.path.exists('output'):
        os.mkdir('output')
    wd_root = 'output/kus_et_al_replication'
    for exp_class, exp_class_label in variants:
        for experiment, label in exp_class:
            if not (exp_class_label[0].value == 'attack' and exp_class_label[1].value == 'inc'):
                break
            cwd = '-'.join((wd_root, exp_class_label[0].value, exp_class_label[1].value, label))
            
            if not os.path.exists(cwd):
                os.mkdir(cwd)
            dataset_path = cwd+'/experiment_dataset.arff'
            write_dataset(experiment, dataset_path)

            setup_cmd = PRE_PROCESS_CMD(dataset_path, 'svm', f'--label binary -n minmax --payload-indicator-imputed -o {cwd}/{STD_DIR}/binary-std-minmax-kmeans')
            subprocess.run(f'{setup_cmd};python {REPOSITORY_ROOT}/src/svm_testset.py -d {cwd}/{STD_DIR}/binary-std-minmax-kmeans -i 1', shell=True)
            
            setup_cmd = PRE_PROCESS_CMD(dataset_path, 'rf', f'--label binary -n minmax --payload-indicator-imputed -o {cwd}/{STD_DIR}/binary-std-minmax-kmeans')
            subprocess.run(f'{setup_cmd};python {REPOSITORY_ROOT}/src/random_forest_testset.py -d {cwd}/{STD_DIR}/binary-std-minmax-kmeans -i 1', shell=True)

            setup_cmd = PRE_PROCESS_CMD(dataset_path, 'blstm', f'--label binary -n minmax --payload-indicator-imputed -o {cwd}/{TS_DIR}/binary-std-minmax-kmeans')
            subprocess.run(f'{setup_cmd};python {REPOSITORY_ROOT}/src/bidirectional-lstm.py -d {cwd}/{STD_DIR}/binary-std-minmax-kmeans -i 1', shell=True)
