# HRNet-W18 Pretrained Weight

Download `model.safetensors` from the official timm model resource:

- Model page: https://huggingface.co/timm/hrnet_w18.ms_aug_in1k
- File: https://huggingface.co/timm/hrnet_w18.ms_aug_in1k/resolve/main/model.safetensors?download=true

Rename the downloaded file to `model_w18.safetensors` and place it in this
directory:

```text
weights/pretrained/model_w18.safetensors
```

Used to initialize the backbone during training. Evaluation uses the supplied
`weights/trained/hrnet_w18_lite.ckpt` checkpoint.