# NIvsCI

## Prerequisites
- Python 2
- NVIDIA GPU + CUDA cuDNN
- PyTorch 0.3.1

## Run
- Normal training: python normal_training.py
- Construct negative samples in the offline manner: python constructNegativeSamples.py
- Enhanced training: python enhanced_training.py
- Test the results: python test.py

## Citation
If you find this useful for your research, please use the following.

```
@INPROCEEDINGS{quan2019enhance, 
  author={W. {Quan} and K. {Wang} and D. {Yan} and D. {Pellerin} and X. {Zhang}}, 
  booktitle={Proceedings of the International Symposium on Image and Signal Processing and Analysis}, 
  title={Improving the Generalization of Colorized Image Detection with Enhanced Training of CNN}, 
  year={2019}, 
  pages={246-252}
}
```

## Acknowledgments
This code borrows from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).  
