
# GLStyleNet
### Environment Required:
- Python 3.6
- TensorFlow 1.4.0

If you want to run on your GPU, make sure that the memory of your GPU is **large enough**, otherwise you may **not be able to**  process enough large pictures.
### Getting Started
Step 1: clone this repo


`git clone https://github.com/EndyWon/GLStyleNet`  
`cd GLStyleNet`


Step 2: download pre-trained vgg19 model


`bash ./download_vgg19.sh`


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
  * `-init`  : image type to initialize, value='noise' or 'content' or 'style', default='content'
  * `--iterations`   : number of iterations, default=500
  * `--device`  : devices, value='gpu'(all available GPUs) or 'gpui'(e.g. gpu0) or 'cpu', default='gpu'
  * `--class-num`   : count of semantic mask classes, default=5

2. **portrait style transfer** (an example)


`python GLStyleNet.py --content portrait/Seth.jpg --content-mask portrait/Seth_sem.png --style portrait/Gogh.jpg --style-mask portrait/Gogh_sem.png --content-weight 10 --local-weight 500 --semantic-weight 10 --global-weight 1 --init style --device gpu`


**！！！You can find all the iteration results in folder 'outputs'！！！**

![portraits](https://github.com/EndyWon/GLStyleNet/blob/master/examples/portraits.png)

3. **Chinese ancient painting style transfer** (an example)


`python GLStyleNet.py --content Chinese/content.jpg --content-mask Chinese/content_sem.png --style Chinese/style.jpg --style-mask Chinese/style_sem.png --content-weight 10 --local-weight 500 --semantic-weight 2.5 --global-weight 0.5 --init content --device gpu`

![portraits](https://github.com/EndyWon/GLStyleNet/blob/master/examples/Chinese.png)

