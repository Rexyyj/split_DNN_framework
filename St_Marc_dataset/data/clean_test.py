
import os


test_file  = "./test_30_fps_long.txt"
cleaned_file = "./test_30_fps_long_cleaned.txt"


with open(test_file,"r") as f:
    frames = f.readlines()


with open(cleaned_file,"w") as f:
    for frame in frames:
        if os.path.isfile(frame.replace("\n","").replace(".jpg",".txt").replace("/images/","/labels/")):
            f.write(frame)





