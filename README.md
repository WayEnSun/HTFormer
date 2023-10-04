# HTFormer
<<<<<<< HEAD

 

## Setup
* Prepare Anaconda, CUDA and the corresponding toolkits. CUDA version required: 10.0+

* Create a new conda environment and activate it.
```Shell
conda create -n HTFormer python=3.7 -y
conda activate HTFormer
```

* Install `pytorch` and `torchvision`.
```Shell
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch
# pytorch v1.5.0, v1.6.0, or higher should also be OK. 
```

* Install other required packages.
```Shell
pip install -r requirements.txt
```

## Test
* Prepare the datasets: OTB2015, VOT2018, UAV123, GOT-10k, TrackingNet, LaSOT, ILSVRC VID*, ILSVRC DET*, COCO*, and something else you want to test. Set the paths as the following: 
```Shell
├── HTFormer
|   ├── ...
|   ├── ...
|   ├── datasets
|   |   ├── COCO -> /opt/data/COCO
|   |   ├── GOT-10k -> /opt/data/GOT-10k
|   |   ├── ILSVRC2015 -> /opt/data/ILSVRC2015
|   |   ├── LaSOT -> /opt/data/LaSOT/LaSOTBenchmark
|   |   ├── OTB
|   |   |   └── OTB2015 -> /opt/data/OTB2015
|   |   ├── TrackingNet -> /opt/data/TrackingNet
|   |   ├── UAV123 -> /opt/data/UAV123/UAV123
|   |   ├── VOT
|   |   |   ├── vot2018
|   |   |   |   ├── VOT2018 -> /opt/data/VOT2018
|   |   |   |   └── VOT2018.json
```


### General command format
```Shell
python main/test.py --config testing_dataset_config_file_path
```

Take GOT-10k as an example:
```Shell
python main/test.py --config experiments/HTFormer/test/got10k/htformer-googlenet-got.yaml
```

## Training
* Prepare the datasets as described in the last subsection.
* Download the pretrained backbone model from [here](https://drive.google.com/file/d/1IaupGGr1Tn3L5e3IVUyB_7CJUNcYx3Vh/view?usp=sharing).
* Run the shell command.

### training based on the GOT-10k benchmark
```Shell
python main/train.py --config experiments/HTFormer/train/got10k/htformer-googlenet-trn.yaml
```

### training with full data
```Shell
python main/train.py --config experiments/HTFormer/train/fulldata/htformer-googlenet-trn-fulldata.yaml
```



## Acknowledgement
### Repository

* [video_analyst](https://github.com/MegviiDetection/video_analyst)
* [pytracking](https://github.com/visionml/pytracking)
* [PySOT](https://github.com/STVIR/pysot)
* [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)
* [mmdetection](https://github.com/open-mmlab/mmdetection)

This repository is developed based on the single object tracking framework [video_analyst](https://github.com/MegviiDetection/video_analyst). See it for more instructions and details.



=======
Video Object Tracking
>>>>>>> 5ee24dc865ba4296169e49bde9775336d12e193c
