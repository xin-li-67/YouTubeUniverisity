import os
import argparse
import pandas as pd

from pydub import AudioSegment
from pydub.utils import make_chunks

def main(args):
    df = pd.read_csv(args.file_name, sep='\t')
    print(df.head())
    print('Total data size:', len(df))

    def chunk_and_save(file):
        path = os.path.join(args.data_path, file)
        audio = AudioSegment.from_file(path)
        length = args.seconds * 1000
        chunks = make_chunks(audio, length)
        names = []
        for i, chunk in enumerate(chunks):
            _name = file.split(".")[0] + ".wav"
            name = "{}_{}".format(i, _name)
            wav_path = os.path.join(args.save_path, name)
            chunk.export(wav_path, format="wav")
        return names
    df.path.apply(lambda x: chunk_and_save(x))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="script to split common voice data into chunks")
    parser.add_argument('--seconds', type=int, default=None, help='set to None to record forever until keyboard interrupt')
    parser.add_argument('--data_path', type=str, default=None, required=True, help='full path to data')
    parser.add_argument('--file_name', type=str, default=None, required=True, help='common voice file')
    parser.add_argument('--save_path', type=str, default=None, required=True, help='full path to to save data')
    args = parser.parse_args()

    main(args)