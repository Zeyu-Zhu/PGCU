# Probability-based Global Cross-modal Upsampling for Pansharpening (CVPR'23)

[`Zeyu Zhu`](zeyuzhu2077@gmail.com) and [`Xiangyong Cao`](https://gr.xjtu.edu.cn/en/web/caoxiangyong/home)


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


The implementation of PGCU is in model/PGCU.py, and the upsampling factor is set to 4. To use PGCU in a pan-sharpening network, you can simply replace the original upsampling method with PGCU. A real example can be seen in model/PanNet.py, in which PanNet is the original pan-sharpening method and PanNet* is the method whose upsampling component is replaced by PGCU
## How to Declaration PGCU?
There are three main hyperparameters needed to be set
    the number of channels in the LRMS image
    the length of the feature vector in F, G.
    the number of DownSamplingBlock used in Information Extraction(for F)
## How to use PGCU in forward function?
    eg: upsampled_ms = self.PGCU(pan, lrms)
Just simply replace the original upsampling method with PGCU.
## How to change scale factor in PGCU?
PGCU is designed to upsample the LRMS to the scale of PAN. The only thing need to change is the difference between the number of DownSamplingBlock for F and G.
        
