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

    f = open(args.save_json_path + "/" + "train.json", "w")
    with open(args.save_json_path + "/" + "train.json", "w") as f:
        i = 0
        while (i < int(len(data) - len(data)/percent)):
            line = json.dumps(data[i])
            f.write(line + "\n")
            i += 1
    
    f = open(args.save_json_path + "/" + "test.json", "w")
    with open(args.save_json_path + "/" + "test.json", "w") as f:
        i = 0
        while (i < int(len(data) - len(data)/percent)):
            line = json.dumps(data[i])
            f.write(line + "\n")
            i += 1
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Utility script to create the training and test json files for speechrecognition. """
    )
    parser.add_argument('--file_folder_directory', type=str, default=None, required=True,
                        help='directory of clips given by mimic-recording-studio')
    parser.add_argument('--save_json_path', type=str, default=None, required=True,
                        help='path to the dir where the json files are supposed to be saved')
    parser.add_argument('--percent', type=int, default=10, required=False,
                        help='percent of clips put into test.json instead of train.json')
    
    args = parser.parse_args()
    main(args)