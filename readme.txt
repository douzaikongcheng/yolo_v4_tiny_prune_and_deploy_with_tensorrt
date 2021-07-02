该项参考：https://github.com/bubbliiiing/yolov4-tiny-pytorch
感谢bubbliiiing

1.进行剪枝和finetune剪枝后的模型
    python3 train.py

2.用剪枝后的模型进行预测（注意要加参数--prune）
    2.1：图片
        python3 predict.py --test_mode img --prune
    2.2：视频
        python3 predict.py --test_mode video --prune
    2.3：摄像头检测
        python3 predict.py --test_mode camera --prune

3.剪枝率：74.9%
    剪枝前的参数量：5.92M
    剪枝后的参数量：1.49M

4.训练和finetune数据集下载：
    参考https://github.com/bubbliiiing/yolov4-tiny-pytorch

5.代码在https://github.com/bubbliiiing/yolov4-tiny-pytorch基础上修改

6.首先对模型进行转换为onnx格式
    python3 pt_to_onnx.py

7.使用tensorrt加速：
    7.1：图片
        python3 predict_use_tensorrt.py --test_mode img --prune
    7.2：视频
        python3 predict_use_tensorrt.py --test_mode video --prune
    7.3：摄像头检测
        python3 predict_use_tensorrt.py --test_mode camera --prune

8.加速效果
    8.1
        ori：
        原始yolov4 tiny模型
        Inference time : 0.03680276870727539
    8.2
        prune：
        剪枝后的yolov4 tiny模型
        Inference time : 0.0277252197265625
    8.3
        FP16+temsorrt：
        使用tensorrt加速的剪枝后的yolov4 tiny模型
        Inference time : 0.009780645370483398
