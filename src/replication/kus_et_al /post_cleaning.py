import os
OUTPUT_DIR = 'output/'

def clean(dir: str, suffix: str):
    for file_name in os.listdir(dir):
        file = os.path.join(dir, file_name)
        if os.path.isdir(file):
            clean_interim_data(file)
        elif file_name[-4:] == suffix:
            os.remove(file)

def clean_interim_data(dir:str):
    clean(dir, suffix='.npy')

def clean_weights(dir:str):
    clean(dir, suffix='.hdf5')
    clean(dir, suffix='.v2')

if __name__ == '__main__':
    clean_interim_data(OUTPUT_DIR)
