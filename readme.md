# HSIGene: A Foundation Model For Hyperspectral Image Generation

### [Paper (ArXiv)](https://arxiv.org/abs/2409.12470)

Official implementation of HSIGene: A Foundation Model For Hyperspectral Image Generation.


## Dependencies and Installation
```bash
conda create -n hsigene python=3.9
conda activate hsigene
pip install -r requirements.txt
```

## Usage
### For Unconditional Generation

1. Download models for hyperspectral image synthesis from [GoogleDrive](https://drive.google.com/file/d/1bBSRn5uyrGcsXWzu4CTzLzO3MH-XYGct/view?usp=drive_link) and put it to `checkpoints`.

2. Running the following script and the generated HSIs will be saved at `save_uncond`. 

```
python inference_uncond.py --num-samples 10 --ddim-steps 50 --save-dir save_uncond
```

### For Conditional Generation
1. Download models for hyperspectral image synthesis from [GoogleDrive](https://drive.google.com/file/d/1bBSRn5uyrGcsXWzu4CTzLzO3MH-XYGct/view?usp=drive_link) and put it to `checkpoints`. 
2. Download files from [huggingface](https://huggingface.co/openai/clip-vit-large-patch14) and put the files to `data_prepare/annotator/ckpts/clip/clip-vit-large-patch14`, or download  the `clip` folder from [BaiduNetdisk](https://pan.baidu.com/s/1_rPPuJei_aklAFT-a0KZ0w?pwd=n86f) (code:n86f) and put it to `data_prepare/annotator/ckpts`.
3. Running the following script and the generated HSIs will be saved at `save_cond`. Available conditions include *hed*, *mlsd*, *sketch*, *segmentation*, *content* and *text*. Example images and conditions are provided in `data_prepare/candidates` and `data_prepare/conditions` respectively.
```
# hed
python inference_single.py --conditions hed --fns f4 --condition-dir data_prepare/conditions --save-dir save_cond

# mlsd
python inference_single.py --conditions mlsd --fns c3 --condition-dir data_prepare/conditions --save-dir save_cond

# sketch
python inference_single.py --conditions sketch --fns a2 --condition-dir data_prepare/conditions --save-dir save_cond

# segmentation
python inference_single.py --conditions segmentation --fns w5 --condition-dir data_prepare/conditions --save-dir save_cond

# content
python inference_single.py --conditions content --fns a1 --condition-dir data_prepare/conditions --save-dir save_cond

# text
python inference_single.py --conditions text --prompt Wasteland --fns Wasteland --save-dir save_cond

# composable conditions
python inference_single.py --conditions 'mlsd segmentation' --fns c2 --condition-dir data_prepare/conditions --save-dir save_cond
```

### Prepare Your Own Conditions
To prepare the conditions, you have to put the original images into `data_prepare/candidates`. In addition, models for condition generation could be downloaded automatically or manually downloaded from [BaiduNetdisk](https://pan.baidu.com/s/1K1Y__blA6uJVV9l1QG7QvQ?pwd=98f1) (code:98f1) and need to be put to `data_prepare/annotator/ckpts`. 

Then, you can obtain you own conditions simply by:
```
cd data_prepare
python data_prepare.py
```

## Contact
If you have any question, please email `pp2373886592@gmail.com`

## Citation
```
@misc{pang2024hsigenefoundationmodelhyperspectral,
      title={HSIGene: A Foundation Model For Hyperspectral Image Generation}, 
      author={Li Pang and Datao Tang and Shuang Xu and Deyu Meng and Xiangyong Cao},
      year={2024},
      eprint={2409.12470},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.12470}, 
}
```
