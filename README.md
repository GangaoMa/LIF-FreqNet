The code of LIF-FreqNet
MSRS dataset is used to train. You can get it from [here.](https://github.com/Linfeng-Tang/MSRS)
After downloading the MSRS dataset, modify the path in the `cfg.yaml` configuration file under the `configs` folder to the path of the MSRS dataset.
Then run the `train.py` file. When executed for the first time, it will first segment the saliency masks to facilitate the use of the saliency object loss function.
Once the mask segmentation is completed, the model will start training.
In addition, after the model training is finished, if you test it on other datasets, you need to change the file path. The directory structure of the test dataset should include test images and sequence files.
