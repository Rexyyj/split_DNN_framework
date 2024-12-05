import cv2

image_path = "/home/matteo/Documents/postDoc/RexTorino/split_DNN_framework/pytorchyolo/data/custom/images/ILSVRC2015_train_00005003_0.jpg"
label_path = "/home/matteo/Documents/postDoc/RexTorino/split_DNN_framework/pytorchyolo/data/custom/labels/ILSVRC2015_train_00005003_0.txt"

# image_path = "/home/matteo/Documents/postDoc/RexTorino/split_DNN_framework/pytorchyolo/data/custom/images/ILSVRC2015_val_00177001_94.jpg"
# label_path = "/home/matteo/Documents/postDoc/RexTorino/split_DNN_framework/pytorchyolo/data/custom/labels/ILSVRC2015_val_00177001_94.txt"

image = cv2.imread(image_path)
colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
with open(label_path, 'r') as f:
    for i, line in enumerate(f):
        label_idx, x_center, y_center, width, height = map(float, line.split())
        print(label_idx, x_center, y_center, width, height)
        x_center = int(x_center * image.shape[1])
        y_center = int(y_center * image.shape[0])
        width = int(width * image.shape[1])
        height = int(height * image.shape[0])
        print(x_center, y_center, width, height)
        cv2.rectangle(image, (x_center - width//2, y_center - height//2), (x_center + width//2, y_center + height//2), colors[i], 2)
        cv2.putText(image, str(int(label_idx)), (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[i], 2, cv2.LINE_AA)
        cv2.imwrite("simple_test.jpg", image)
cv2.imshow('image', image)
