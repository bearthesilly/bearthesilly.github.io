# Project 5: Diffusion Models

## Project 5A: The Power of Diffusion Models!

### Overview

In part A you will play around with diffusion models, implement diffusion sampling loops, and use them for other tasks such as inpainting and creating optical illusions.

### Part 0: Setup

> We are going to use the [DeepFloyd IF](https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if) diffusion model. DeepFloyd is a two stage model trained by Stability AI. The first stage produces images of size and the second stage takes the outputs of the first stage and generates images of size.

Loading this model can be somehow difficult for some mysterious bug. I ran into `std::bad alloc()` issue for some undepictable reasons and the colab will simply crash down with few information for problem shooting. Hence, I followed the method introduced in the following link to finish part0.

> https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/deepfloyd_if_free_tier_google_colab.ipynb

Deliverables: 

- Come up with some interesting text prompts and generate their embeddings.

The three prompts I used are: 

```python
embeds_1 = embeddings_dict['an android dreaming of electric sheep']
embeds_2 = embeddings_dict['Dr.Jones holding the holy grail']
embeds_3 = embeddings_dict['a mobius strip']
```

- Choose 3 of your prompts to generate images and display the caption and the output of the model. Reflect on the quality of the outputs and their relationships to the text prompts. Make sure to try at least 2 different `num_inference_steps` values.

***Figure 1: an android dreaming of electric sheep***

<div style="display: flex; justify-content: space-around; align-items: center;">
        <figure>
            <img src="./material/part0_step_50/blade_runner.png" style="zoom:480%; height: auto;">
            <figcaption>
                stage 1; 50 steps
            </figcaption>
        </figure>
            <figure>
            <img src="./material/part0_step_50/upscaled_blade_runner.png" style="zoom:120%; height: auto;">
            <figcaption>
                stage 2; 50 steps
            </figcaption>
        </figure>
             <figure>
            <img src="./material/part0_step_50/ultimate_blade_runner.png" style="zoom:30%; height: auto;">
            <figcaption>
                stage 3; 50 steps
            </figcaption>
</div>

<div style="display: flex; justify-content: space-around; align-items: center;">
        <figure>
            <img src="./material/part0_step_100/blade_runner.png" style="zoom:480%; height: auto;">
            <figcaption>
                stage 1; 100 steps
            </figcaption>
        </figure>
            <figure>
            <img src="./material/part0_step_100/upscaled_blade_runner.png" style="zoom:120%; height: auto;">
            <figcaption>
                stage 2; 100 steps
            </figcaption>
        </figure>
             <figure>
            <img src="./material/part0_step_100/ultimate_blade_runner.png" style="zoom:30%; height: auto;">
            <figcaption>
                stage 3; 100 steps
            </figcaption>
</div>

Yes, I've watched the movie 'Blade Runner' recently. To be honest, I think this prompt is kind of difficult. The 100 inference steps figures basically can satisfy my requirement with the android and sheep, but the 50 inference steps figures are far from satisfaction. 

***Figure 2: Dr.Jones holding the holy grail***

<div style="display: flex; justify-content: space-around; align-items: center;">
        <figure>
            <img src="./material/part0_step_50/jones.png" style="zoom:480%; height: auto;">
            <figcaption>
                stage 1; 50 steps
            </figcaption>
        </figure>
            <figure>
            <img src="./material/part0_step_50/upscaled_jones.png" style="zoom:120%; height: auto;">
            <figcaption>
                stage 2; 50 steps
            </figcaption>
        </figure>
             <figure>
            <img src="./material/part0_step_50/ultimate_jones.png" style="zoom:30%; height: auto;">
            <figcaption>
                stage 3; 50 steps
            </figcaption>
</div>

<div style="display: flex; justify-content: space-around; align-items: center;">
        <figure>
            <img src="./material/part0_step_100/jones.png" style="zoom:480%; height: auto;">
            <figcaption>
                stage 1; 100 steps
            </figcaption>
        </figure>
            <figure>
            <img src="./material/part0_step_100/upscaled_jones.png" style="zoom:120%; height: auto;">
            <figcaption>
                stage 2; 100 steps
            </figcaption>
        </figure>
             <figure>
            <img src="./material/part0_step_100/ultimate_jones.png" style="zoom:30%; height: auto;">
            <figcaption>
                stage 3; 100 steps
            </figcaption>
</div>

Yes again, I've watched the movie 'Indiana Jones' series recently. All pictures above have the appearance of Dr. Jones (though maybe not the original one in the movie, but it is an old man professor as well). But the holy grail is not interpreted well enough.

***Figure 3: a mobius strip***

<div style="display: flex; justify-content: space-around; align-items: center;">
        <figure>
            <img src="./material/part0_step_50/mobius.png" style="zoom:480%; height: auto;">
            <figcaption>
                stage 1; 50 steps
            </figcaption>
        </figure>
            <figure>
            <img src="./material/part0_step_50/upscaled_mobius.png" style="zoom:120%; height: auto;">
            <figcaption>
                stage 2; 50 steps
            </figcaption>
        </figure>
             <figure>
            <img src="./material/part0_step_50/ultimate_mobius.png" style="zoom:30%; height: auto;">
            <figcaption>
                stage 3; 50 steps
            </figcaption>
</div>

<div style="display: flex; justify-content: space-around; align-items: center;">
        <figure>
            <img src="./material/part0_step_100/mobius.png" style="zoom:480%; height: auto;">
            <figcaption>
                stage 1; 50 steps
            </figcaption>
        </figure>
            <figure>
            <img src="./material/part0_step_100/upscaled_mobius.png" style="zoom:120%; height: auto;">
            <figcaption>
                stage 2; 50 steps
            </figcaption>
        </figure>
             <figure>
            <img src="./material/part0_step_100/ultimate_mobius.png" style="zoom:30%; height: auto;">
            <figcaption>
                stage 3; 50 steps
            </figcaption>
</div>

Both 50 inference steps figures and 100 inference steps figures are not satisfying. But at least the figures are about a strip. Maybe I believe that 'Mobius Strip' rarely appears in the training corpus of the DeepFloyd model so the model can't handle the concept of 'Mobius Strip'.

- Report the random seed that you're using here. You should use the same seed all subsequent parts.

The random seed I set here is 42. 

### Part 1: Sampling Loops

> In this part of the problem set, you will write your own "sampling loops" that use the pretrained DeepFloyd denoisers. These should produce high quality images such as the ones generated above.
>
> You will then modify these sampling loops to solve different tasks such as inpainting or producing optical illusions.

#### 1.1 Implementing the Forward Process

The forward process is defined by:
$$
q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
$$
And is equivalent to computing:
$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon} \quad \text{where} \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$
In the implementation, $\bar{\alpha}_t$ is retrieved from the loaded DeepFloyd Model. 

````python
alphas_cumprod = stage_1.scheduler.alphas_cumprod
print(f"We have in total {alphas_cumprod.shape[0]} noise coefficients")
# We have in total 1000 noise coefficients
````

**Deliverables**: Show the Campanile at noise level [250, 500, 750].

![image](material/part1/1_1.png)

> Original; t=250; t=500; t=750

#### 1.2 Classical Denoising

 Here try to use **Gaussian blur filtering** to try to remove the noise. Here, I set `kernel_size = 5` and `sigma = 2`

> If sigma None, then it is computed using kernel_size as $sigma = 0.3 * ((kernel\_size - 1) * 0.5 - 1) + 0.8$.

**Deliverables**: For each of the 3 noisy Campanile images from the previous part, show your best Gaussian-denoised version side by side.

![image](material/part1/1_2-1.png)

![image](material/part1/1_2-2.png)

#### 1.3 One-Step Denoising

Now, we'll use a pretrained diffusion model to denoise. The actual denoiser can be found at `stage_1.unet`. This is a UNet that has already been trained on a *very, very* large dataset of $(x_0, x_t)$ pairs of images. We can use it to recover Gaussian noise from the image. Then, we can remove this noise to recover (something close to) the original image. Note: this UNet is conditioned on the amount of Gaussian noise by taking timestep as additional input.

When removing the noise, it is not a simple subtraction. After deriving the math, the denoising formula should as below:
$$
x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\hat{\varepsilon}}{\sqrt{\bar{\alpha}_t}}
$$
**Deliverables**: the original image, the noisy image, and the estimate of the original image are as below:

![image](material/part1/1_3.png)

#### 1.4 Iterative Denoising

In part 1.3, you should see that the denoising UNet does a much better job of projecting the image onto the natural image manifold, but it does get worse as you add more noise. This makes sense, as the problem is much harder with more noise!

In theory, we could start with noise $x_{1000}$ at timestep $T=1000$, denoise for one step to get an estimate of $x_{999}$, and carry on until we get $x_0$. But this would require running the diffusion model 1000 times, which is quite slow. It turns out, we can actually speed things up by skipping steps. The rationale for why this is possible is due to a connection with differential equations. To actually do this, we have the following formula:
$$
\mathbf{x}_{t'} = \frac{\sqrt{\bar{\alpha}_{t'}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0 + \frac{\sqrt{\bar{\alpha}_t(1-\bar{\alpha}_{t'})}}{1-\bar{\alpha}_t}\mathbf{x}_t + \boldsymbol{\nu}\boldsymbol{\sigma}
$$

- $\mathbf{x}_t$ is your **image at timestep $t$**
- $\mathbf{x}_{t'}$ is your **noisy image at timestep $t'$** where $t' < t$ (**less noisy**)
- $\bar{\alpha}_t$ is defined by `alphas_cumprod`, as explained above.
- $\alpha_t = \frac{\bar{\alpha}_t}{\bar{\alpha}_{t'}}$
- $\beta_t = 1 - \alpha_t$
- $\mathbf{x}_0$ is our **current estimate of the clean image** using one-step denoising

**Deliverables**: The comprehensive required images gallery is as below:

![image](material/part1/1_4-1.png)

![image](material/part1/1_4-2.png)

#### 1.5 Diffusion Model Sampling

In part 1.4, we use the diffusion model to denoise an image. Another thing we can do with the `iterative_denoise` function is to generate images from scratch. We can do this by setting `i_start = 0` and passing `im_noisy` as random noise. This effectively denoises pure noise. Please do this, and show 5 results of the prompt`"a high quality photo"`.

**Deliverables**: The 5 sampled images are as below:

![image](material/part1/1_5_samples.png)

#### 1.6 Classifier-Free Guidance (CFG)

In order to greatly improve image quality (at the expense of image diversity), we can use a technicque called Classifier-Free Guidance.

In CFG, we compute both a conditional and an unconditional noise estimate. We denote these $\epsilon_c$ and $\epsilon_u$. Then, we let our new noise estimate be:
$$
\boldsymbol{\epsilon} = \boldsymbol{\epsilon}_u + \gamma(\boldsymbol{\epsilon}_c - \boldsymbol{\epsilon}_u)
$$
where $\mathbf{\gamma}=7$ controls the strength of CFG.

**Deliverables**: Show 5 images of `"a high quality photo"` with a CFG scale of  $\mathbf{\gamma}=7$

![image](material/part1/1_6_cfg_samples.png)

#### 1.7 Image-to-image Translation

In part 1.4, we take a real image, add noise to it, and then denoise. This effectively allows us to make edits to existing images. The more noise we add, the larger the edit will be. This works because in order to denoise an image, the diffusion model must to some extent "hallucinate" new things -- the model has to be "creative." Another way to think about it is that the denoising process "forces" a noisy image back onto the manifold of natural images.

Here, we're going to take the original Campanile image, noise it a little, and force it back onto the image manifold without any conditioning. Effectively, we're going to get an image that is similar to the Campanile (with a low-enough noise level). This follows the [SDEdit](https://sde-image-editing.github.io/) algorithm.

**Deliverables**: 

For 1.7.2 and 1.7.3, my own pictures are borrowed from project4: fox and shanghaitech:

<div style="display: flex; justify-content: space-around; align-items: center;">
        <figure>
            <img src="./material/fox.jpg" style="zoom:70%; height: auto;">
            <figcaption>
                Fox
            </figcaption>
        </figure>
            <figure>
            <img src="./material/shanghaitech.png" style="zoom:120%; height: auto;">
            <figcaption>
                ShanghaiTech
            </figcaption>

![image](material/part1/1_7-1.png)

![image](material/part1/1_7-2.png)

![image](material/part1/1_7-3.png)

#####  1.7.1 Editing Hand Drawn and Web Images

This procedure works particularly well if we start with a nonrealistic image (e.g. painting, a sketch, some scribbles) and project it onto the natural image manifold.

Please experiment by starting with hand-drawn or other non-realistic images and see how you can get them onto the natural image manifold in fun ways.

**Deliverables**:

![image](material/part1/1_7_1-1.png)

![image](material/part1/1_7_1-2.png)

<img src="material/part1/1_7_1-3.png" alt="image" style="zoom:150%;" />

##### 1.7.2 Inpainting

We can use the same procedure to implement inpainting (following the [RePaint](https://arxiv.org/abs/2201.09865) paper). The formula is as follows:
$$
\mathbf{x}_t \leftarrow \mathbf{m}\mathbf{x}_t + (1 - \mathbf{m})\text{forward}(\mathbf{x}_{\text{orig}}, t)
$$
Essentially, we leave everything inside the edit mask alone, but we replace everything outside the edit mask with our original image -- with the correct amount of noise added for timestep $\mathbf{t}$.

**Deliverables**:

![image](material/part1/1_7_2-1.png)

![image](material/part1/1_7_2-2.png)

![image](material/part1/1_7_2-4.png)![image](material/part1/1_7_2-3.png)

##### 1.7.3: Text Conditional Image to Image Translation

Now, we will do the same thing as SDEdit, but guide the projection with a text prompt. This is no longer pure "projection to the natural image manifold" but also adds control using language. This is simply a matter of changing the prompt from `"a high quality photo"` to any of your prompt!

**Deliverables**:

![image](material/part1/1_7_3-1.png)

> "a rocket ship" $\rightarrow$ Companile

![image](material/part1/1_7_3-2.png)

> "a pencil" $\rightarrow$ Companile

<img src="material/part1/1_7_3-3.png" alt="image" style="zoom:150%;" />

> "a rocket ship" $\rightarrow$ Fox

<img src="material/part1/1_7_3-4.png" alt="image" style="zoom:150%;" />

> "a pencil" $\rightarrow$ ShanghaiTech

#### 1.8 Visual Anagrams

The full algorithm will be:
$$
\begin{align}
\boldsymbol{\epsilon}_1 &= \text{CFG of UNet}(\mathbf{x}_t, t, \mathbf{p}_1) \\
\boldsymbol{\epsilon}_2 &= \text{flip}(\text{CFG of UNet}(\text{flip}(\mathbf{x}_t), t, \mathbf{p}_2)) \\
\boldsymbol{\epsilon} &= (\boldsymbol{\epsilon}_1 + \boldsymbol{\epsilon}_2)/2
\end{align}
$$
**Deliverables**:

````python
embeds_1 = prompt_embeds_dict["an oil painting of an old man"]
embeds_2 = prompt_embeds_dict["an oil painting of people around a campfire"]
````

![image](material/part1/1_8-1.png)

````python
embeds_1 = prompt_embeds_dict["a photo of a hipster barista"]
embeds_2 = prompt_embeds_dict["a photo of a dog"]
````

![image](material/part1/1_8-2.png)



#### 1.9 Hybrid Images

In order to create hybrid images with a diffusion model we can use a similar technique as above. We will create a composite noise estimate $\epsilon$, by estimating the noise with two different text prompts, and then combining low frequencies from one noise estimate with high frequencies of the other. The algorithm is:
$$
\begin{align}
\boldsymbol{\epsilon}_1 &= \text{CFG of UNet}(\mathbf{x}_t, t, \mathbf{p}_1) \\
\boldsymbol{\epsilon}_2 &= \text{CFG of UNet}(\mathbf{x}_t, t, \mathbf{p}_2) \\
\boldsymbol{\epsilon} &= f_{\text{lowpass}}(\boldsymbol{\epsilon}_1) + f_{\text{highpass}}(\boldsymbol{\epsilon}_2)
\end{align}
$$
It is recommend to use a gaussian blur of kernel size 33 and sigma 2.

**Deliverables**: 

<img src="material/part1/1_9-1.png" alt="image" style="zoom:100%;" />

<img src="material/part1/1_9-2.png" alt="image" style="zoom:100%;" />

> Sorry the second figure is mislabeled, the caption should be `skull + campfire`