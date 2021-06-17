import os
import json
import random
import argparse

# util script to create thr training and test json files for speech recognition
def main(args):
    data = []
    directory = args.file_folder_directory
    file_name = args.file_folder_directory.rpartition('/')[2]
    percent = args.percent

    with open(directory + "/" + file_name + "-metadata.txt") as f:
        for line in f:
            f_name = line.partition('|')[0]
            text = line.split('|')[1]
            data.append({
                "key": directory+"/"+f_name,
                "text": text
                })
    
    random.shuffle(data)

    