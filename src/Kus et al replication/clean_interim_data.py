import os
OUTPUT_DIR = 'output/'

def clean(dir: str):
    for file_name in os.listdir(dir):
        file = os.path.join(dir, file_name)
        if os.path.isdir(file):
            clean(file)
        elif file_name[-4:] == '.npy':
            os.remove(file)

if __name__ == '__main__':
    clean(OUTPUT_DIR)