import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()
    with open(args.file) as f:
        for i, line in enumerate(f):
            print(i + 1)
            x = np.array(line.split()).astype(np.float32)
