# LITA: Language Instructed Temporal-Localization Assistant

[De-An Huang](https://ai.stanford.edu/~dahuang/), [Shijia Liao](), [Subhashree Radhakrishnan](), [Hongxu Yin](https://hongxu-yin.github.io/), [Pavlo Molchanov](https://www.pmolchanov.com/), [Zhiding Yu](https://chrisding.github.io/), [Jan Kautz](https://jankautz.com/)

[[`arXiv`](https://arxiv.org/abs/2403.19046)] [[`Project`]()] [[`BibTeX`](#Citation)]


<img src="https://ai.stanford.edu/~dahuang/images/lita_beam.gif" height="400"/> <img src="https://ai.stanford.edu/~dahuang/images/lita_scuba.gif" height="400"/>


## Contents
- [Install](#install)
- [Dataset](#dataset)
- [Weights](#weights)
- [Demo](#demo)
- [Train](#train)
- [Evaluation](#evaluation)


## Install

1. The environment requirements are mostly the same as [LLaVA](https://github.com/haotian-liu/LLaVA). In addition, install `ffmpeg`.

2. Clone this repository and navigate to LITA folder
```bash
git clone https://github.com/NVlabs/LITA.git
cd LITA
```

3. Install Package
```Shell
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

4. Install additional packages for training cases
```
pip install ninja
pip install flash-attn --no-build-isolation
```


## Dataset

See [Preparing Datasets for LITA](docs/Video_Data.md).


## Weights

| Model Name | LLM version | Weights |
|------------|:---------------:|:---------:|
| LITA-13B-v1.3 | [Vicuna-13B-v1.3](https://huggingface.co/lmsys/vicuna-13b-v1.3) | [Link](https://drive.google.com/drive/u/2/folders/1-P7p-tq5aXZzSoefEJx4PSFKH8jt8KWy) |


## Gradio Demo

First, downlaod the LITA weights from above.
```Shell
python -m lita.serve.gradio_web_server \
    --model-path <weights-dir>/lita-vicuna-v1-3-13b-finetune 
```
To create a public link, append `--share` to the above command. You can also launch the demo with quantized bits (4-bit, 8-bit) by appending `--load-4bit` or `--load-8bit`. Note that inference with quantized bits may not be as accurate as the full-precision model.


## CLI Inference

We also provide inference using CLI without the need of Gradio interface. 
```Shell
python -m lita.serve.cli \
    --model-path <weights-dir>/lita-vicuna-v1-3-13b-finetune \
    --visual-path <video-path> --visual-data-type video
```
`<video-path>` is the path to the input video. Inference with quantized bits (`--load-4bit` or `--load-8bit`) also works here.


## Train

The LITA model only uses one stage supervised fine-tuning. The linear projection is initialized by the LLaVA pretrained weights. The training uses 8 A100 GPUs with 80GB memory.

### Prepare public checkpoints from Vicuna, LLaVA

```Shell
git clone https://huggingface.co/lmsys/vicuna-13b-v1.3
git clone https://huggingface.co/liuhaotian/llava-pretrain-vicuna-13b-v1.3
mv vicuna-13b-v1.3 vicuna-v1-3-13b
mv llava-pretrain-vicuna-13b-v1.3 llava-vicuna-v1-3-13b-pretrain
```
Similarly for 7B checkpoints. Replace `13b` with `7b` in the above commands.

### Supervised Fine-tuning

The LITA model can be trained using the supervised fine-tuning script [here](scripts/finetune_vid.sh). First update information in the script such as dataset directory (`--data_path`) and checkpoint directory (`./checkpoints`).
```Shell
cd LITA
sh scripts/finetune_vid.sh
```


## Evaluation

We provide the evaluation pipeline for the [ActivityNet-RTL](https://drive.google.com/drive/folders/1a9mM9h2vV-b9uH6gmYDDyGzrWDJee3Uc?usp=drive_link) dataset. Please first follow the [dataset instruction](docs/Video_Data.md#reasoning-temporal-localization) and refer to our paper for more details. 

1. Generate LITA responses and evaluate temporal localization metrics (mIOU and P@0.5)
```Shell
python lita/eval/eval_model_rtl.py \
    --model-path <weights-dir>/lita-vicuna-v1-3-13b-finetune  \
    --question-file \
    <datasets-dir>/temporal_reasoning/annot_val_1_q229.json \
    --image-folder \
    <datasets-dir>/activitynet-captions/activitynet_frames \
    --output-dir \
    <result-dir>/lita-vicuna-v1-3-13b-finetune
```

2. Evaluate the generated responses using GPT-4
```Shell
OPENAI_API_KEY="sk-***********************************" python lita/eval/eval_gpt_review_rtl.py \
    --context <datasets-dir>/activitynet-captions/val_1.json \
    --answer \
    <result-dir>/lita-vicuna-v1-3-13b-finetune/answers.json \
    --rule lita/eval/table/rule.txt \
    --output <result-dir>/reviews/lita-vicuna-v1-3-13b-finetune.jsonl
```

3. Summarize the evaluation results
```Shell
python lita/eval/summarize_gpt_review.py -f <result-dir>/reviews/lita-vicuna-v1-3-13b-finetune.jsonl
```


## License

Copyright © 2024, NVIDIA Corporation. All rights reserved.

This work is made available under the Nvidia Source Code License-NC. Click [here](LICENSE) to view a copy of this license.

The pre-trained models are shared under [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).


## <a name="Citation"></a> Citation

If you find LITA useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{huang2024lita,
  title={LITA: Language Instructed Temporal-Localization Assistant},
  author={De-An Huang and Shijia Liao and Subhashree Radhakrishnan and Hongxu Yin and Pavlo Molchanov and Zhiding Yu and Jan Kautz},
  journal={arXiv preprint arXiv:2403.19046},
  year={2024}
}
```


## Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon
