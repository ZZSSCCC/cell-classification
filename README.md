# cell-classification

Training:
CUDA_VISIBLE_DEVICES=5 python train.py --name resnet34 --batch_size 100

Modify the code line 43 in model.py to running other metric learning losses. We have implemented them in the end of the file model.py

