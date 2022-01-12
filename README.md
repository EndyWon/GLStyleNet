
# GLStyleNet
**[update 1/12/2022]**

paper: [GLStyleNet: Exquisite Style Transfer Combining Global and Local Pyramid Features](https://ietresearch.onlinelibrary.wiley.com/doi/pdf/10.1049/iet-cvi.2019.0844), published in [IET Computer Vision 2020](https://digital-library.theiet.org/content/journals/iet-cvi)

Arxiv paper: [GLStyleNet: Higher Quality Style Transfer Combining Global and Local Pyramid Features](https://arxiv.org/abs/1811.07260)
### Environment Required:
- Python 3.6
- TensorFlow 1.4.0
- CUDA 8.0

### Getting Started
Step 1: clone this repo


`git clone https://github.com/EndyWon/GLStyleNet`  
`cd GLStyleNet`


Step 2: download pre-trained vgg19 model


`bash download_vgg19.sh`


Step 3:  run style transfer
1. **Script Parameters**
  * `--content`  : content image path
  * `--content-mask`  : content image semantic mask
  * `--style`  : style image path
  * `--style-mask`  : style image semantic mask
  * `--content-weight`  : weight of content, default=10
  * `--local-weight`  : weight of local style loss
  * `--semantic-weight`  : weight of semantic map constraint
  * `--global-weight`  : weight of global style loss
  * `--output`  : output image path
  * `--smoothness`  : weight of image smoothing scheme
  * `--init`  : image type to initialize, value='noise' or 'content' or 'style', default='content'
  * `--iterations`   : number of iterations, default=500
  * `--device`  : devices, value='gpu'(all available GPUs) or 'gpui'(e.g. gpu0) or 'cpu', default='gpu'
  * `--class-num`   : count of semantic mask classes, default=5

2. **portrait style transfer** (an example)


`python GLStyleNet.py --content portrait/Seth.jpg --content-mask portrait/Seth_sem.png --style portrait/Gogh.jpg --style-mask portrait/Gogh_sem.png --content-weight 10 --local-weight 500 --semantic-weight 10 --global-weight 1 --init style --device gpu`


**！！！You can find all the iteration results in folder 'outputs'！！！**

![portraits](https://github.com/EndyWon/GLStyleNet/blob/master/examples/portraits.png)

3. **Chinese ancient painting style transfer** (an example)


`python GLStyleNet.py --content Chinese/content.jpg --content-mask Chinese/content_sem.png --style Chinese/style.jpg --style-mask Chinese/style_sem.png --content-weight 10 --local-weight 500 --semantic-weight 2.5 --global-weight 0.5 --init content --device gpu`

![Chinese](https://github.com/EndyWon/GLStyleNet/blob/master/examples/Chinese.png)

4. **artistic and photo-realistic style transfer**

#### artistic:

![artistic](https://github.com/EndyWon/GLStyleNet/blob/master/examples/artistic.png)

#### photo-realistic:

![photo-realistic](https://github.com/EndyWon/GLStyleNet/blob/master/examples/photo-realistic.png)


## Citation:

If you find this code useful for your research, please cite the paper:

```
@article{wang2020glstylenet,
  title={GLStyleNet: exquisite style transfer combining global and local pyramid features},
  author={Wang, Zhizhong and Zhao, Lei and Lin, Sihuan and Mo, Qihang and Zhang, Huiming and Xing, Wei and Lu, Dongming},
  journal={IET Computer Vision},
  volume={14},
  number={8},
  pages={575--586},
  year={2020},
  publisher={IET}
}
```

## Acknowledgement:
The code was written based on [Champandard's code](https://github.com/alexjc/neural-doodle).
