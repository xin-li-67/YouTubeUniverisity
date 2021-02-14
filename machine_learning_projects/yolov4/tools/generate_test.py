import os

image_files = []
os.chdir(os.path.join("data", "obj"))

for filename in os.listdir(os.getcwd()):
    if filename.endswith(".jpg"):
        image_files.append("data/obj/" + filename)
    
os.chdir("..")

with open("train.txt", "w") as f:
    for image in image_files:
        f.write(image)
        f.write("\n")
    
    f.close()

os.chdir("..")