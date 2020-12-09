#file to encode text to npz which is used to train a dataset

import argparse
import numpy as np
import encoder

from load_dataset import load_dataset

#parse the input text file to a tokenized training set
parser = argparse.ArgumentParser(
    description='pre encoded text files into tokenized training set..',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--model_name', metavar='MODEL', type=str, default='124M', help='model name before training')
parser.add_argument('--combine', metavar='CHARS', type=int, default=50000, help='concatenate files')
parser.add_argument('--encoding', type='str', default='utf-8', help='set the encoding for reading and writing files')
parser.add_argument('in_text', metavar='PATH', type=str, help='input file directory')
parser.add_argument('out_npz', metavar='OUT.npz', type=str , help='where to output file')


def main():
    #the main that runs
    args = parser.parse_args()
    enc = encoder.get_encoder(args.model_name)
    
    print('######### reading files ###########')
    chunks = load_dataset(enc,args.in_text,args.combine, encoding=args.encoding)
    print("output",args.out_npz)
    np.savez_compressed(input.out_npz,*chunks)


if __name__ == "__main__":
    main()
