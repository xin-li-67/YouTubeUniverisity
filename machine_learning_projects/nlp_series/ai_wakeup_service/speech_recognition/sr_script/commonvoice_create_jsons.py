import os
import csv
import json
import random
import argparse

from pydub import AudioSegment

def main(args):
    data = []
    directory = args.file_path.rpartition('/')[0]
    percent = args.percent

    with open(args.file_path) as f:
        total = sum(1 for line in f)
    
    with open(args.file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        index = 1
        
        if (args.convert):
            print(str(total) + "files found")
        
        for rows in reader:
            f_name = rows['path']
            filename = f_name.rpartition('.')[0] + ".wav"
            text = rows['sentence']
            if (args.convert):
                data.append({
                    "key": directory + "/clips/" + filename,
                    "text": text
                })
                print("converting file " + str(index) + "/" + str(total) + " to wav", end="\r")

                src = directory + "/clips/" + f_name
                dst = directory + "/clips/" + filename
                sound = AudioSegment.from_mp3(src)
                sound.export(dst, format="wav")
                index += 1
            else:
                data.append({
                    "key": directory + "/clips/" + f_name,
                    "text": text
                })
    
    random.shuffle(data)
    print("creating JSON")

    f = open(args.save_json_path + "/" + "train.json", "w")
    with open(args.save_json_path + "/" + "train.json", "w") as f:
        i = 0
        while(i < int(len(data) - len(data)/percent)):
            r = data[i]
            line = json.dumps(r)
            f.write(line + "\n")
            i += 1
    
    f = open(args.save_json_path + "/" + "test.json", "w")
    with open(args.save_json_path + "/" + "test.json", "w") as f:
        i = int(len(data) - len(data)/percent)
        while(i < len(data)):
            r = data[i]
            line = json.dumps(r)
            f.write(line + "\n")
            i += 1
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Utility script to convert commonvoice into wav and create the training and test json files for speechrecognition. """
    )
    parser.add_argument('--file_path', type=str, default=None, required=True,
                        help='path to one of the .tsv files found in cv-corpus')
    parser.add_argument('--save_json_path', type=str, default=None, required=True,
                        help='path to the dir where the json files are supposed to be saved')
    parser.add_argument('--percent', type=int, default=10, required=False,
                        help='percent of clips put into test.json instead of train.json')
    parser.add_argument('--convert', default=True, action='store_true',
                        help='says that the script should convert mp3 to wav')
    parser.add_argument('--not-convert', dest='convert', action='store_false',
                        help='says that the script should not convert mp3 to wav')

    args = parser.parse_args()
    main(args)