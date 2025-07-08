## yolov6 4.0 onnx to bm1684 onnx
```shell
pip install -r requirements.txt  
python onnx_subgraph.py yolov6n_4.0_12class_640_n_finetune_v2_640_top100_dynamic_batch_2024_11_12_09_42_56.onnx
```
## bm1684 onnx to int8
基于[23.09 LTS SP4](https://developer.sophgo.com/site/index/material/93/all.html) 或[SOPHONSDK 开发指南](https://doc.sophgo.com/sdk-docs/v23.09.01-lts-sp4/docs_latest_release/docs/SophonSDK_doc/zh/html/index.html)

```shell
./onnx_to_int8 yolov6n_4.0_12class_640_n_finetune_v2_640_top100_dynamic_batch_2024_11_12_09_42_56_bm1684x.onnx 7 640 ./quant_img 128
```

