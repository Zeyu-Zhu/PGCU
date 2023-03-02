# Probability-based Global Cross-modal Upsampling for Pansharpening (CVPR'23)

[`Zeyu Zhu`](zeyuzhu2077@gmail.com) and [`Xiangyong Cao`](https://gr.xjtu.edu.cn/en/web/caoxiangyong/publication)


For more information, please see our 
- **Paper**: [`CVPR-2023-Open-Access`]() or [`arxiv`]().

# Environment
Setup a virtual environment using the provided``requirements.txt``.
```
pip  install -i requirements.txt
```

# Project Struct
    model/ 
        PGCU.py     The implementation of PGCU 
        PanNet.py   The implementation of PanNet and PanNet*(improved by PGCU)
    utils/
        funcation.py Some useful funcations
        dataset.py   Datapreprocess and dataloader
        visualize.py The recoder for training process
        metrices.py  The evaluation index, e.g. SAM, ESGAR, PSNR and etc
    train_pannet.py  Training PanNet and PanNet*

# How to use PGCU?
    The implementation of PGCU is in model/PGCU.py, and the upsampling factor is set to 4, which is the image scale of the PAN image. To use PGCU in a pan-sharpening network, you can simply replace the original upsampling method with PGCU. Specifically, 
    <1>. How to Declaration PGCU?
        There are two main hyperparameters needed to be set, which are the number of channels in the LRMS image and the length of the feature vector in F, G (Reference paper).
    <2>. How to use PGCU in forward function?
        Just simply replace the original upsampling method with PGCU.
            eg: upsampled_ms = self.PGCU(pan, lrms)
        A real example can be seen in model/FusionNet.py, in which FusionNet is the original pan-sharpening method and FusionNet* is the method whose upsampling component is replaced by PGCU
        
