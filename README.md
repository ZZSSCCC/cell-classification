# cell-classification

Training:
CUDA_VISIBLE_DEVICES=5 python train.py --name resnet34 --batch_size 100

Modify the code line 43 in model.py to running other metric learning losses. We have implemented them in the end of the file model.py

Rewrite the dataset.py for your own data. The dataset we used is private.

In order to use the dataset.py we provided, it is necessary to preprare you train.txt val.txt and test.txt as follows:
./img_path0 label0

./img_path1 label1

./img_path2 label2

./img_path3 label3

./img_path4 label4

......
