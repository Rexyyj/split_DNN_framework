import os

image_dir = "/home/rex/dataset/coco/images/"
output_file = "./coco_val.txt"


def sort_func(name):
    num = name.split(".")[0].split("_")[1]
    return int(num)

images = os.listdir(image_dir)
# images.sort(key=sort_func)
abs_dir  = os.path.abspath(image_dir) +"/"
try:
    os.system("rm -rf "+ output_file)
except:
    pass


with open(output_file,'w') as f:
    for img in images:
        f.write(abs_dir+img + "\n")



