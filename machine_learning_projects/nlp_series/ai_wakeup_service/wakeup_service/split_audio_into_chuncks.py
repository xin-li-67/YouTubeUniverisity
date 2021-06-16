import os
import argparse

from pydub import AudioSegment
from pydub.utils import make_chunks

def main(args):
    def chunck_and_save(file):
        audio = AudioSegment.from_file(file)
        len = args.seconds * 1000 # miliseconds
        chuncks = make_chunks(audio, len)
        names = []

        for i, chunk in enumerate(chuncks):
            _name = file.split("/")[-1]
            name = "{}_{}".format(i, _name)
            wav_path = os.path.join(args.save_path, name)
            chunk.export(wav_path, format="wav")
        return names
    
    chunck_and_save(args.audio_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="script to split audio files into chunks")
    parser.add_argument('--seconds', type=int, default=None, help='Set to None to record forever until keyboard interrupt')
    parser.add_argument('--audio_file_name', type=str, default=None, required=True, help='Enter the name of audio file')
    parser.add_argument('--save_path', type=str, default=None, required=True, help='Use full path to to save data')
    args = parser.parse_args()

    main(args)