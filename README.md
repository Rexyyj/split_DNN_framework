# split_DNN_framework

## branch: ImageNet-VidVRD finetuning

-----------------

Hello Rex, there are some few additional files in this branch:
- download the dataset: https://xdshang.github.io/docs/imagenet-vidvrd.html. Both annotations and video are required.
- pytorchyolo/checkpoints/vidVRD.pth is the finetuned checkpoint
- pytorchyolo/config/custom.data describes the structure of the dataset
- pytorchyolo/config/yolov3-custom.cfg describes the achitecture of the model (same as pytorchyolo/config/yolov3-tiny.cfg)
- ImageNet-VidVRD/sampleAndLabel.ipynb: this has to be run in order to obtain images from the VidVRD dataset, you will need to change paths to according to your machine
- to test: poetry run yolo-test --weights checkpoints/yolov3_ckpt_300.pth --model config/yolov3-tiny.cfg --data config/custom.data

-------------------

### Testing results:

+-------+--------------+---------+
| 0     | dog          | 0.15703 |
| 1     | frisbee      | 0.00000 |
| 2     | bicycle      | 0.51512 |
| 3     | person       | 0.40603 |
| 4     | zebra        | 0.51533 |
| 5     | sofa         | 0.00000 |
| 6     | motorcycle   | 0.46186 |
| 7     | car          | 0.24427 |
| 8     | domestic_cat | 0.00001 |
| 9     | ball         | 0.00000 |
| 10    | red_panda    | 0.45264 |
| 11    | giant_panda  | 0.21266 |
| 12    | hamster      | 0.35790 |
| 13    | squirrel     | 0.00000 |
| 14    | monkey       | 0.07482 |
| 15    | snake        | 0.27165 |
| 16    | cattle       | 0.44699 |
| 17    | antelope     | 0.51726 |
| 18    | sheep        | 0.00069 |
| 19    | turtle       | 0.00000 |
| 20    | elephant     | 0.24961 |
| 21    | horse        | 0.44905 |
| 22    | bird         | 0.03521 |
| 23    | bear         | 0.19446 |
| 24    | fox          | 0.27784 |
| 25    | airplane     | 0.66711 |
| 26    | tiger        | 0.00000 |
| 27    | watercraft   | 0.51814 |
| 28    | rabbit       | 0.00000 |
| 29    | lizard       | 0.00000 |
| 30    | whale        | 0.13179 |
| 31    | lion         | 0.01535 |
| 32    | bus          | 0.00000 |
| 33    | skateboard   | 0.00093 |
| 34    | train        | 0.00000 |
+-------+--------------+---------+
---- mAP 0.20496 ----

## Running the split framework
Running the split framework require two machines. We should first deploy the DNN tail model on the edge server, and then run the DNN head model on the mobile device. 

### On the edge server (ux550 latop)
1. Activate Pytorch environment with: ```conda activate pytorch```
2. Enter the folder: ```/home/rex/gitHub/split_DNN_framework/experiments_football```
3. Run the DNN tail model with: ``` python3 experiment_tensor_jpeg_edge.py <DNN_split_layer>``` For the current tests, we alwasy split the DNN at the 8th layer.

### On the mobile device side (Jetson Orin Nano)
1. Activate Pytorch environment with: ```conda activate pytorch```
2. Enter the folder: ```/home/rex/gitRepo/split_DNN_framework/experiments_football```
3. Run the DNN head model with the test scenario. There are two different versions:
   * Printing version: ``` python3 experiment_tensor_jpeg_clientv2.py <DNN_split_layer>``` With this script, at the end of the test the AP of each class will be printed at the terminal.
   * Logging version: ``` python3 experiment_tensor_jpeg_client.py <DNN_split_layer>``` With this script, the mAP and the each step delay will be measured and logged to the folder: ```/home/rex/gitRepo/split_DNN_framework/measurements/yolo_tiny_splitpoint/layer_8/football_tensor_jpeg```