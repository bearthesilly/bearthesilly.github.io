# Proj4 - Neural Radiance Field!

## Part 0: Calibrating Your Camera and Capturing a 3D Scan

### Part 0.1: Calibrating Your Camera



### Part 0.3: Estimating Camera Pose

***Deliverable:*** ***Here are the results of  2 screenshots of your camera frustums visualization in Viser***

<div style="display: flex; justify-content: space-around; align-items: center;">
        <figure>
            <img src="./material/android_frustum_1.png" style="zoom:58%; height: auto;">
        </figure>
             <figure>
            <img src="./material/android_frustum_2.png" style="zoom:60%; height: auto;">
        </figure>
</div>



## Part 1: Fit a Neural Field to a 2D Image

In this part, we will try to fit NeRF to 2D scenario. The pictures I chose are fox and ShanghaiTech:

> Fox Image Source: https://live.staticflickr.com/7492/15677707699_d9d67acf9d_b.jpg
>
> ShanghaiTech Image Source: https://zhuanlan.zhihu.com/p/616005193

<div style="display: flex; justify-content: space-around; align-items: center;">
        <figure>
            <img src="./material/fox.jpg" style="zoom:35%; height: auto;">
        </figure>
             <figure>
            <img src="./material/shanghaitech.png" style="zoom:60%; height: auto;">
        </figure>
</div>

***Deliverable:*** ***Model architecture report (number of layers, width, learning rate, and other important details)***

During training, the inference, aka, the predicted whole image is presented as below. From the progression visualization, we can infer that during training model learned and did better and better:
***Deliverable:*** ***Training progression visualization on both the provided test image and one of your own images:***

![](material/2d_nerf_results.png)

![](material/2d_nerf_results_shanghaitech.png)

In this model, some hyperparameter can be rather important. For instance, the choice of max positional encoding frequency and width (which can be interpreted as the dimension of the hidden layer). There hyperparameters can heavily affect the expressive power of the model. Intuitively, higher the max frequency is, larger the hidden dimension is, the more expressive the model is. Here is the naive comparison study visualization to directly demonstrate this conclusion:

***Deliverable: Final results for 2 choices of max positional encoding frequency and 2 choices of width (2x2 grid)***

<div style="display: flex; justify-content: space-around; align-items: center;">
        <figure>
            <img src="./material/2d_nerf_2x2_grid.png" style="zoom:58%; height: auto;">
        </figure>
             <figure>
            <img src="./material/2d_nerf_2x2_grid_shanghaitech.png" style="zoom:60%; height: auto;">
        </figure>
</div>

The PSNR curves of fox and ShanghaiTech during training are presented as below:

> Left v.s. Right: Fox v.s. ShanghaiTech

***Deliverable: PSNR curve for training on one image of your choice***

<div style="display: flex; justify-content: space-around; align-items: center;">
        <figure>
            <img src="./material/2d_nerf_psnr.png" style="zoom:58%; height: auto;">
        </figure>
             <figure>
            <img src="./material/2d_nerf_psnr_shanghaitech.png" style="zoom:60%; height: auto;">
        </figure>
</div>

## Part 2: Fit a Neural Radiance Field from Multi-view Images

***Deliverable: Brief description of how you implemented each part***

### Part 2.1: Lego

***Deliverable: Visualization of rays and samples with cameras (up to 100 rays)***

<div style="display: flex; justify-content: space-around; align-items: center;">
        <figure>
            <img src="./material/lego_frustum_1.png" style="zoom:100%; height: auto;">
        </figure>
             <figure>
            <img src="./material/lego_frustum_2.png" style="zoom:100%; height: auto;">
        </figure>
             <figure>
            <img src="./material/lego_ray.png" style="zoom:100%; height: auto;">
        </figure>
</div>

***Deliverable: Training progression visualization with predicted images across iterations***

![image](material/process.png)

***Deliverable: PSNR curve on the validation set***

<img src="material/3000_6144.png" alt="image" style="zoom:50%;" />

***Deliverable: Spherical rendering video of the Lego using provided test cameras***

<img src="material/lego_novel_10000_6144.gif" alt="gif" style="zoom:150%;" />

### Part 2.2: Training with Your Own Data



***Deliverable: GIF of camera circling your object showing novel views***

<img src="material/lego_novel.gif" alt="gif" style="zoom: 67%;" />

Unfortunately, the nerf training on the my Android Man dataset performs poorly. The PSNR and loss curve during training is provided below:

***Deliverable: Plot of training loss over iterations***

<div style="display: flex; justify-content: space-around; align-items: center;">
        <figure>
            <img src="./material/psnr_curve.png" style="zoom:58%; height: auto;">
        </figure>
             <figure>
            <img src="./material/training_loss_curve.png" style="zoom:60%; height: auto;">
        </figure>
</div>

***Deliverable: Discussion of code or hyperparameter changes you made***



For the hyperparameter, from the perspective of rendering and point sampling, `near and far` parameters are rather important. For my own datasets, I initially tried `0.02 and 0.5`, and jittered around the values to seek for the best performance. Unfortunately, all of them perform poorly. And then from the perspective of batch size and number of iterations can be important as well. For these two parameters, intuitively, the larger, the better. Here are the demonstration of the results under different parameters settings:

<div style="display: flex; justify-content: space-around; align-items: center;">
        <figure>
            <img src="./material/lego_novel_1000_1024.gif" style="zoom:150%; height: auto;"><figcaption>Num Iter: 1000; Batch Size: 1024</figcaption>
        </figure>
             <figure>
            <img src="./material/lego_novel_2000_4096.gif" style="zoom:150%; height: auto;"><figcaption>Num Iter: 2000; Batch Size: 4096</figcaption>
        </figure>
</div>

<div style="display: flex; justify-content: space-around; align-items: center;">
        <figure>
            <img src="./material/lego_novel_3000_6144.gif" style="zoom:150%; height: auto;"><figcaption>Num Iter: 3000; Batch Size: 6144</figcaption>
        </figure>
             <figure>
            <img src="./material/lego_novel_10000_6144.gif" style="zoom:150%; height: auto;"><figcaption>Num Iter: 10000; Batch Size: 6144</figcaption>
        </figure>
</div>