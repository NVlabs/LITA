## Preparing Datasets for LITA

All datasets should be under a dataset root folder (`--data-path`). An example of dataset root folder is [here](https://huggingface.co/datasets/deahuang/LITA-Datasets/tree/main). Some directories already contain processed data. We will fill the empty/partial directories by the following instructions.


### Dense Video Captioning and Event Localization
```
├── activitynet-captions
│   └── activitynet_frames
│       └── v_00Dk03Jr70M
│           ...
└── youcook2
    └── youcook2_frames
        └── 01lB162koHA
            ...
```

1. Download ActivityNet frames from ActivityNet Challenge ([link](http://activity-net.org/challenges/2021/tasks/anet_captioning.html)) and put under `activitynet-captions`.

2. From dataset root
```Shell
cd activitynet-captions
tar -xf frames_activitynet_5fps_res_320x240.tar
wget https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip
unzip -qq captions.zip
```

3. Download videos from [YouCook2](http://youcook2.eecs.umich.edu/).

4. Extract video frames and put under `youcook2/youcook2_frames`. For `ffmpeg`, we use the following command
```Shell
ffmpeg -i <video_path> -y -an -qscale 0 -vf "fps=5,scale=320:240" <output_folder>/%06d.jpg 
```


### Video Question Answering
```
└── nextqa
```

1. Download videos for [NExT-QA](https://github.com/doc-doc/NExT-QA) and put under `nextqa`.

2. From dataset root
```Shell
cd nextqa
unzip -qq NExTVideo.zip
```


### Reasoning Temporal Localization
```
└── temporal_reasoning
```

The processed data for our Reasoning Temporal Localization (RTL) is already under the `temporal_reasoning` folder. We use videos from ActivityNet prepared above, so no further preparation is needed.


### Natural Language Visual Question Answering
```
├── LLaVA-Instruct-150K
└── coco
```

Follow LLaVA data preparation. From dataset root
```Shell
git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K

mkdir coco
cd coco
wget http://images.cocodataset.org/zips/train2017.zip
unzip -qq train2017.zip
```
