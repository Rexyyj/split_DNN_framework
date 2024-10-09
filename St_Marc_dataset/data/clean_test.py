
import os


test_file  = "./test_5_fps.txt"
cleaned_file = "./test_5_fps_cleaned.txt"


with open(test_file,"r") as f:
    frames = f.readlines()


with open(cleaned_file,"w") as f:
    for frame in frames:
        if os.path.isfile(frame.replace("\n","").replace(".jpg",".txt").replace("/images/","/labels/")):
            f.write(frame)





