import tensorflow as tf
import numpy as np
import skimage.io
import itertools
import os
import bz2
import argparse
import scipy
import skimage.transform

CONTENT_LAYERS = ['4_1']
LOCAL_STYLE_LAYERS = ['1_1','2_1','3_1','4_1']
GLOBAL_STYLE_LAYERS=['1_1','2_1','3_1','4_1']


def conv2d(input_tensor, kernel, bias):
    kernel = np.transpose(kernel, [2, 3, 1, 0])
    x = tf.pad(input_tensor, [[0,0], [1,1], [1,1], [0,0]])
    x = tf.nn.conv2d(x, tf.constant(kernel), (1,1,1,1), 'VALID')
    x = tf.nn.bias_add(x, tf.constant(bias))
    return tf.nn.relu(x)

def avg_pooling(input_tensor, size=2):
    return tf.nn.pool(input_tensor, [size, size], 'AVG', 'VALID', strides=[size, size])

def norm(arr):
    n, *shape = arr.shape
    lst = []
    for i in range(n):
        v = arr[i, :].flatten()
        v /= np.sqrt(sum(v**2))
        lst.append(np.reshape(v, shape))
    return lst

def build_base_net(input_tensor,input_map=None):
    vgg19_file = os.path.join(os.path.dirname(__file__), 'vgg19.pkl.bz2')
    assert os.path.exists(vgg19_file), ("Model file with pre-trained convolution layers not found. Download here: "
        +"https://github.com/alexjc/neural-doodle/releases/download/v0.0/vgg19_conv.pkl.bz2")

    data = np.load(bz2.open(vgg19_file, 'rb'))
    k = 0
    net = {}
    # network divided into two parts，main and map，main downsamples the image，map dowsamples the semantic map
    net['img'] = input_tensor
    net['conv1_1'] = conv2d(net['img'], data[k], data[k+1])
    k += 2
    net['conv1_2'] = conv2d(net['conv1_1'], data[k], data[k+1])
    k += 2
    # average pooling without padding
    net['pool1']   = avg_pooling(net['conv1_2'])
    net['conv2_1'] = conv2d(net['pool1'], data[k], data[k+1])
    k += 2
    net['conv2_2'] = conv2d(net['conv2_1'], data[k], data[k+1])
    k += 2
    net['pool2']   = avg_pooling(net['conv2_2'])
    net['conv3_1'] = conv2d(net['pool2'], data[k], data[k+1])
    k += 2
    net['conv3_2'] = conv2d(net['conv3_1'], data[k], data[k+1])
    k += 2
    net['conv3_3'] = conv2d(net['conv3_2'], data[k], data[k+1])
    k += 2
    net['conv3_4'] = conv2d(net['conv3_3'], data[k], data[k+1])
    k += 2
    net['pool3']   = avg_pooling(net['conv3_4'])
    net['conv4_1'] = conv2d(net['pool3'], data[k], data[k+1])
    k += 2
    net['conv4_2'] = conv2d(net['conv4_1'], data[k], data[k+1])
    k += 2
    net['conv4_3'] = conv2d(net['conv4_2'], data[k], data[k+1])
    k += 2
    net['conv4_4'] = conv2d(net['conv4_3'], data[k], data[k+1])
    k += 2
    net['pool4']   = avg_pooling(net['conv4_4'])
    net['conv5_1'] = conv2d(net['pool4'], data[k], data[k+1])
    k += 2
    net['conv5_2'] = conv2d(net['conv5_1'], data[k], data[k+1])
    k += 2
    net['conv5_3'] = conv2d(net['conv5_2'], data[k], data[k+1])
    k += 2
    net['conv5_4'] = conv2d(net['conv5_3'], data[k], data[k+1])
    k += 2
    net['main'] = net['conv5_4']

    net['map'] = input_map
    for j, i in itertools.product(range(5), range(4)):
        if j < 2 and i > 1: continue 
        suffix = '%i_%i' % (j+1, i+1)

        if i == 0: 
            net['map%i'%(j+1)] = avg_pooling(net['map'], 2**j)
        net['sem'+suffix] = tf.concat([net['conv'+suffix], net['map%i'%(j+1)]], -1)
    return net


def extract_target_data(content, content_mask, style, style_mask):
    pixel_mean = np.array([103.939, 116.779, 123.680], dtype=np.float32).reshape((1,1,1,3))
    # local style patches extracting
    input_tensor = style-pixel_mean
    input_map= style_mask
    net = build_base_net(input_tensor, input_map)
    local_features = [net['sem'+layer] for layer in LOCAL_STYLE_LAYERS]
    # layer aggregation for local style
    LF=local_features[0]
    for i in range(1,len(LOCAL_STYLE_LAYERS)):
        lf=local_features[i]
        LF=tf.image.resize_images(LF,[lf.shape[1],lf.shape[2]],method=tf.image.ResizeMethod.BILINEAR)
        LF=tf.concat([LF,lf],3)
      
    dim = LF.shape[-1].value
    x = tf.extract_image_patches(LF, (1,3,3,1), (1,1,1,1), (1,1,1,1), 'VALID')
    patches=tf.reshape(x, (-1, 3, 3, dim))
       
    # content features
    input_tensor = content-pixel_mean
    input_map= content_mask
    net = build_base_net(input_tensor, input_map)    
    content_features = [net['conv'+layer] for layer in CONTENT_LAYERS]
    content_data=[]
    
    # global feature correlations based on fused features
    input_tensor = style-pixel_mean
    input_map= style_mask
    net = build_base_net(input_tensor, input_map)    
    global_features = [net['conv'+layer] for layer in GLOBAL_STYLE_LAYERS]
    GF=global_features[0]
    for i in range(1,len(GLOBAL_STYLE_LAYERS)):
        gf=global_features[i]
        GF=tf.image.resize_images(GF,[gf.shape[1],gf.shape[2]],method=tf.image.ResizeMethod.BILINEAR)
        GF=tf.concat([GF,gf],3)
        
    N=int(GF.shape[3])
    M=int(GF.shape[1]*GF.shape[2])
    GF=tf.reshape(GF,(M,N))   
    GF_corr=tf.matmul(tf.transpose(GF),GF)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        patches=patches.eval()
        for c in content_features:
            content_data.append(c.eval())
        global_data=GF_corr.eval()

    return content_data,patches,global_data

        
def format_and_norm(arr, depth, sem_weight):
    n, *shape = arr.shape
    norm = np.zeros(shape+[n], dtype=arr.dtype)
    un_norm = np.zeros(shape+[n], dtype=arr.dtype)
    for i in range(n):
        t = arr[i, ...]
        un_norm[..., i] = t
        t1 = t[..., :depth]
        t1 = t1/np.sqrt(3*np.sum(t1**2)+1e-6)
        t2 = t[..., depth:]
        t2 = t2/np.sqrt(sem_weight*np.sum(t2**2)+1e-6)
        
        norm[..., i] = np.concatenate([t1,t2], -1)
    return norm, un_norm


"""GLStyleNet"""
class Model(object):
    def __init__(self, args, content, style, style2, content_mask=None, style_mask=None):
        self.args = args
        if len(args.device)>3 and args.device[:3]=='gpu':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device[3:]
        elif args.device=='cpu':
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.pixel_mean = np.array([103.939, 116.779, 123.680], dtype=np.float32).reshape((1,1,1,3))

        self.content = np.expand_dims(content, 0).astype(np.float32)
        self.style = np.expand_dims(style, 0).astype(np.float32)
        self.style2= np.expand_dims(style2, 0).astype(np.float32)
        
        if content_mask is not None:
            self.content_mask = np.expand_dims(content_mask, 0).astype(np.float32)
        else:
            self.content_mask = np.ones(self.content.shape[:-1]+(1,), np.float32)
            self.args.semantic_weight= 0.0
        if style_mask is not None:
            self.style_mask = np.expand_dims(style_mask, 0).astype(np.float32)
        else:
            self.style_mask = np.ones(self.style.shape[:-1]+(1,), np.float32)
            self.args.semantic_weight = 0.0
        assert self.content_mask.shape[-1] == self.style_mask.shape[-1]
        
       
        self.args.semantic_weight=100/self.args.semantic_weight if self.args.semantic_weight else 1E+8
        
        self.mask_depth = self.content_mask.shape[-1]
        # get target content features, local patches, global feature correlations
        self.content_data, self.local_data, self.global_data= extract_target_data(self.content, self.content_mask, self.style, self.style_mask)
        tf.reset_default_graph()
        
        if args.init=='style':
            input_tensor = tf.Variable(self.style2)
        elif args.init=='content':
            input_tensor = tf.Variable(self.content)
        else:
            input_tensor = tf.Variable(np.random.uniform(16, 240, self.content.shape).astype(np.float32))
            
        input_map=tf.Variable(self.content_mask)
        self.net = build_base_net(input_tensor, input_map)

        self.content_features = [self.net['conv'+layer] for layer in CONTENT_LAYERS]
        self.local_features = [self.net['sem'+layer] for layer in LOCAL_STYLE_LAYERS]
        self.global_features = [self.net['conv'+layer] for layer in GLOBAL_STYLE_LAYERS]
        
        # local style layer aggregation
        LF=self.local_features[0]
        for i in range(1,len(LOCAL_STYLE_LAYERS)):
            lf=self.local_features[i]
            LF=tf.image.resize_images(LF,[lf.shape[1],lf.shape[2]],method=tf.image.ResizeMethod.BILINEAR)
            LF=tf.concat([LF,lf],3)
        
        # patch-matching,concatenate semantic maps
        self.local_loss = 0
        sem = LF
        patches = tf.extract_image_patches(sem, (1,3,3,1), (1,1,1,1), (1,1,1,1), 'VALID')
        patches = tf.reshape(patches, (-1, 3, 3, sem.shape[-1].value))
        
        pow2 = patches**2
        p1 = tf.reduce_sum(pow2[..., :-self.mask_depth], [1,2,3])
        p1 = tf.reshape(p1, [-1,1,1,1])
        p1 = pow2[..., :-self.mask_depth]/(3*p1+1e-6)
        p2 = tf.reduce_sum(pow2[..., -self.mask_depth:], [1,2,3])
        p2 = tf.reshape(p2, [-1,1,1,1])
        p2 = pow2[..., -self.mask_depth:]/(self.args.semantic_weight*p2+1e-6)
        norm_patch = tf.concat([p1, p2], -1)
        norm_patch = tf.reshape(norm_patch, [-1, 9*sem.shape[-1].value])
        
        norm, un_norm = format_and_norm(self.local_data, -self.mask_depth, self.args.semantic_weight)
        norm = np.reshape(norm, [9*sem.shape[-1].value, -1])
        sim = tf.matmul(norm_patch, norm)
        max_ind = tf.argmax(sim, axis=-1)
        target_patches = tf.gather(self.local_data, tf.reshape(max_ind, [-1]))
        
        # local style loss
        self.local_loss += tf.reduce_mean((patches[...,:-self.mask_depth]-target_patches[...,:-self.mask_depth])**2)
        self.local_loss *= args.local_weight
        
        # content loss
        self.content_loss = 0
        for c, t in zip(self.content_features, self.content_data) :
            self.content_loss += tf.reduce_mean((c-t)**2)
        self.content_loss *= args.content_weight
        
        # total variation regularization loss
        self.tv_loss = args.smoothness*(tf.reduce_mean(tf.abs(input_tensor[..., :-1,:]-input_tensor[..., 1:,:]))
                                +tf.reduce_mean(tf.abs(input_tensor[..., :, :-1]-input_tensor[..., :,1:])))
        
        # global style loss
        GF=self.global_features[0]
        for i in range(1,len(GLOBAL_STYLE_LAYERS)):
            gf=self.global_features[i]
            GF=tf.image.resize_images(GF,[gf.shape[1],gf.shape[2]],method=tf.image.ResizeMethod.BILINEAR)
            GF=tf.concat([GF,gf],3)

        N=int(GF.shape[3])
        M=int(GF.shape[1]*GF.shape[2])
        GF=tf.reshape(GF,(M,N))   
        GF_corr=tf.matmul(tf.transpose(GF),GF)
        
        self.global_loss = tf.reduce_sum(((GF_corr-self.global_data)**2)/((2*M*N)**2))
        self.global_loss *= args.global_weight
        
        # total loss
        self.loss = self.local_loss + self.content_loss + self.tv_loss + self.global_loss
        self.grad = tf.gradients(self.loss, self.net['img'])
        tf.summary.scalar('loss', self.loss)
        self.merged = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('./summary', tf.get_default_graph())
    def evaluate(self):
        sess = tf.Session()
        def func(img):
            self.iter += 1
            current_img = img.reshape(self.content.shape).astype(np.float32) - self.pixel_mean

            feed_dict = {self.net['img']:current_img, self.net['map']:self.content_mask}
            loss = 0
            grads = 0
            local_loss = 0
            content_loss = 0
            tv_loss=0
            global_loss=0
            sess.run(tf.global_variables_initializer())
            loss, grads, local_loss, content_loss, tv_loss, global_loss, summ= sess.run(
                [self.loss, self.grad, self.local_loss, self.content_loss, self.tv_loss, self.global_loss, self.merged],
                feed_dict=feed_dict)
            self.summary_writer.add_summary(summ, self.iter)
            if self.iter % 10 == 0:
                out = current_img + self.pixel_mean
                out = np.squeeze(out)
                out = np.clip(out, 0, 255).astype('uint8')
                skimage.io.imsave('outputs/%s-%d.jpg'%(self.args.output, self.iter), out)

            print('Epoch:%d,loss:%f,local loss:%f,global loss:%f,content loss:%f,tv loss: %f.'%
                (self.iter, loss, local_loss, global_loss, content_loss, tv_loss))
            if np.isnan(grads).any():
                raise OverflowError("Optimization diverged; try using a different device or parameters.")

            # Return the data in the right format for L-BFGS.
            return loss, np.array(grads).flatten().astype(np.float64)
        return func

    def run(self):
        args = self.args
        if args.init=='style':
            Xn = self.style2
        elif args.init=='content':
            Xn = self.content
        else:
            Xn = np.random.uniform(16, 240, self.content.shape).astype(np.float32)
            
        self.iter = 0
        # Optimization algorithm needs min and max bounds to prevent divergence.
        data_bounds = np.zeros((np.product(Xn.shape), 2), dtype=np.float64)
        data_bounds[:] = (0.0, 255.0)
        print ("GLStyleNet: Start")
        try:
            Xn, *_ = scipy.optimize.fmin_l_bfgs_b(
                            self.evaluate(),
                            Xn.flatten(),
                            bounds=data_bounds,
                            factr=0.0, pgtol=0.0,            # Disable automatic termination, set low threshold.
                            m=5,                             # Maximum correlations kept in memory by algorithm.
                            maxfun=args.iterations,        # Limit number of calls to evaluate().
                            iprint=-1)                       # Handle our own logging of information.
        except OverflowError:
            print("The optimization diverged and NaNs were encountered.",
                    "  - Try using a different `--device` or change the parameters.",
                    "  - Make sure libraries are updated to work around platform bugs.")
        except KeyboardInterrupt:
            print("User canceled.")
        except Exception as e:
            print(e)
            
        print ("GLStyleNet: Completed!")

        self.summary_writer.close()
    
    
def prepare_mask(content_mask, style_mask, n):
    from sklearn.cluster import KMeans
    x1 = content_mask.reshape((-1, content_mask.shape[-1]))
    x2 = style_mask.reshape((-1, style_mask.shape[-1]))
    kmeans = KMeans(n_clusters=n, random_state=0).fit(x1)
    y1 = kmeans.labels_
    y2 = kmeans.predict(x2)
    y1 = y1.reshape(content_mask.shape[:-1])
    y2 = y2.reshape(style_mask.shape[:-1])
    diag = np.diag([1 for _ in range(n)])
    return diag[y1].astype(np.float32), diag[y2].astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description='GLStyleNet: transfer style of a image onto a content image.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arg = parser.add_argument

    add_arg('--content',        default=None, type=str,    help='Content image path.')
    add_arg('--content-mask',     default=None, type=str,    help='Content image semantic mask.')
    add_arg('--content-weight',    default=10,  type=float,  help='Weight of content.')
    add_arg('--style',          default=None, type=str,    help='Style image path.')
    add_arg('--style-mask',      default=None, type=str,    help='Style image semantic map.')
    add_arg('--local-weight',    default=100,    type=float,   help='Weight of local style loss.')
    add_arg('--semantic-weight',  default=10,    type=float,   help='Weight of semantic map channel.')
    add_arg('--global-weight',     default=0.1,  type=float,   help='Weight of global style loss.')
    add_arg('--output',        default='output', type=str, help='Output image path.')
    add_arg('--smoothness',      default=1E+0, type=float,    help='Weight of image smoothing scheme.')
    
    add_arg('--init',         default='content', type=str,   help='Image path to initialize, "noise" or "content" or "style".')
    add_arg('--iterations',     default=500, type=int,       help='Number of iterations.')
    add_arg('--device',         default='gpu', type=str,    help='devices: "gpu"(default: all gpu) or "gpui"(e.g. gpu0) or "cpu" ')
    add_arg('--class-num',      default=5, type=int,      help='Count of semantic mask classes.')
    
    args = parser.parse_args()
    
    style = skimage.io.imread(args.style)
    if args.style_mask:
        style_mask = skimage.io.imread(args.style_mask)
    
    content = skimage.io.imread(args.content)
    if args.content_mask:
        content_mask = skimage.io.imread(args.content_mask)
    
    if style.shape[0]==content.shape[0] and style.shape[1]==content.shape[1]:
        style2=style
    else:
        style2=skimage.transform.resize(style,(content.shape[0],content.shape[1]))

    if args.content_mask and args.style_mask:
        content_mask, style_mask = prepare_mask(content_mask, style_mask, args.class_num)
        model = Model(args, content, style, style2, content_mask, style_mask)
    else:
        model = Model(args, content, style, style2)
    model.run()


if __name__ == '__main__':
    main()