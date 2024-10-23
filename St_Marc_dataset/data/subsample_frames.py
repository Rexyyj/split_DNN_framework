import os
import random

random.seed(42)

samples = []
with open("test_30_fps.txt") as file_in:
    for line in file_in:
        samples.append(line)

random.shuffle(samples) 

fps_options = [30, 15, 10, 5]

for fps in fps_options:
    sampled_samples = []

    for index, s in enumerate(samples):
        if index % (30 / fps) == 0:
            sampled_samples.append(s)

        if len(sampled_samples) >= 34:
            break

    
    with open(f'shuffle_sub_sample_test_{fps}fps.txt',"w") as f:
        f.writelines([f"{x}" for x in sampled_samples])

    print(f'written {len(sampled_samples)} / {len(samples)}')
