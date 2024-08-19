# split_DNN_framework

branch: football finetuning

-----------------

Hello Rex, there are some few additional files in this branch:
- pytorchyolo/checkpoints/yolov3_ckpt_300.pth is the finetuned checkpoint
- pytorchyolo/config/custom.data describes the structure of the dataset
- pytorchyolo/config/yolov3-custom.cfg describes the achitecture of the model (same as pytorchyolo/config/yolov3-tiny.cfg)
- pytorchyolo/data is the preprocess football dataset
- footballtest.py is the python script to run the test on the foolball dataset. Simply run 'python3 footballtest.py'. All params for testing are set at default.

-----------------

Training results:
---- Training Model ----
Training Epoch 300: 100%|████████████████████████████████████████████| 33/33 [00:08<00:00,  3.97it/s]
---- Saving checkpoint to: 'checkpoints/yolov3_ckpt_300.pth' ----

---- Evaluating Model ----
Validating: 100%|██████████████████████████████████████████████████████| 6/6 [00:02<00:00,  2.70it/s]
Computing AP: 100%|███████████████████████████████████████████████████| 4/4 [00:00<00:00, 700.25it/s]
---- mAP 0.31868 ----

-----------------

Testing results:
Environment information:
System: Linux 6.1.0-21-amd64
Not using the poetry package
Current Commit Hash: b139d49
Command line arguments: Namespace(model='config/yolov3-custom.cfg', weights='checkpoints/yolov3_ckpt_300.pth', data='config/custom.data', batch_size=8, verbose=False, img_size=416, n_cpu=8, iou_thres=0.5, conf_thres=0.01, nms_thres=0.4)
Validating: 100%|████████████████████████████████████████████████████████████████████████████████████| 22/22 [00:05<00:00,  4.40it/s]
Computing AP: 100%|███████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 366.29it/s]
+-------+---------------+---------+
| Index | Class         | AP      |
+-------+---------------+---------+
| 0     | ball          | 0.03723 |
| 1     | goalkeeper    | 0.11100 |
| 2     | referee       | 0.53188 |
| 3     | soccer-player | 0.78219 |
+-------+---------------+---------+
---- mAP 0.36557 ----
