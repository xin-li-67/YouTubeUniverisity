# Convert XML to what YOLO needs
import xml.etree.ElementTree as ET
import os
import glob

folderName = os.path.basename(os.getcwd())

if folderName == "tools":
    os.chdir("..")

data_dir = '/dataset/'
dataset_names_path = "model_data/license_plate_names.txt"
dataset_train = "model_data/license_plate_train.txt"
dataset_test = "model_data/license_plate_test.txt"
is_subfolder = False
dataset_names = []

def parseXML(img_folder, file):
    for xml_file in glob.glob(img_folder + '/*.xml'):
        tree = ET.parse(open(xml_file))
        root = tree.getroot()
        img_name = root.find('filename').text
        img_path = img_folder + '/' + img_name

        for _, obj in enumerate(root.iter('object')):
            # difficult = obj.find('difficult').text
            cls = obj.find('name').text

            if cls not in dataset_names:
                dataset_names.append(cls)
            
            cls_id = dataset_names.index(cls)
            xmlbox = obj.find('bndbox')
            OBJECT = (str(int(float(xmlbox.find('xmin').text))) + ','
                    + str(int(float(xmlbox.find('ymin').text))) + ','
                    + str(int(float(xmlbox.find('xmax').text))) + ','
                    + str(int(float(xmlbox.find('ymax').text))) + ','
                    + str(cls_id))
            img_path += ' ' + OBJECT
        
        print(img_path)
        file.write(img_path + '\n')

def converter():
    for i, folder in enumerate(['train', 'valid']):
        with open([dataset_train, dataset_test][i], 'w') as f:
            print(os.getcwd() + data_dir + folder)

            img_path = os.path.join(os.getcwd() + data_dir + folder)

            if is_subfolder:
                for directory in os.listdir(img_path):
                    xml_path = os.path.join(img_path, directory)
                    parseXML(xml_path, f)
            else:
                parseXML(img_path, f)
    
    print("dataset_names:", dataset_names)

    with open(dataset_names_path, 'w') as f:
        for name in dataset_names:
            f.write(str(name) + '\n')

converter()