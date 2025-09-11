# Project 1

## Part 1 - Colorize Small Picture

***Description / Hint***: 

The easiest way to align the parts is to exhaustively search over a window of possible displacements (say `[-15,15]` pixels), score each one using some image matching metric, and take the displacement with the best score. There is a number of possible metrics that one could use to score how well the images match. The simplest one is just the L2 norm also known as the **Euclidean Distance** which is simply `sqrt(sum(sum((image1-image2).^2)))` where the sum is taken over the pixel values. Another is **Normalized Cross-Correlation** (NCC), which is simply a dot product between two normalized vectors: (`image1./||image1||` and `image2./||image2||`).
Note that in the case like the Emir of Bukhara (show on right), the images to be matched do not actually have the same brightness values (they are different color channels), so you might have to use a cleverer metric, or different features than the raw pixels. 

***My Approach***: 

The core of the project is to find the optimal offset for Green channel and Red channel. Since mentioning 'optimal', a optimizing metric is important. This reveals a sole question: how to evaluate a offset? In another word, how do you know under one offset possibility, the pixels together perform well? Here I use the simplest L2 norm, that is to say, the brightness distribution gap between two color channels should be as 'similar' as possible. It is important to know though three channels are filmed under different color filter, there corresponding pixel brightness value should present identical pattern. 

With in a certain offset search range, using loop iteration, we try all the offsets and get the best one using L2 norm as metric. Then we generate the colorful graph via this offset. The result pictures are as below:

<div style="display: flex; justify-content: space-around; align-items: center;">
        <figure>
            <img src="./img/cathedral.jpg" style="zoom:60%; height: auto;">
            <figcaption>
                cathedral.jpg <br>
                Green offset: (2, 5) <br>
                Red offset: (3, 12)
            </figcaption>
        </figure>
             <figure>
            <img src="./img/monastery.jpg" style="zoom:60%; height: auto;">
            <figcaption>
                monastery.jpg <br>
                Green offset: (2, -3) <br>
                Red offset: (2, 3)
            </figcaption>
        </figure>
             <figure>
            <img src="./img/tobolsk.jpg" style="zoom:60%; height: auto;">
            <figcaption>
                tobolsk.jpg <br>
                Green offset: (3, 3) <br>
                Red offset: (3, 6)
            </figcaption>
        </figure>
</div>



## Part 2 - Colorize Large Picture

***Description / Hint***: 

Exhaustive search will become prohibitively expensive if the pixel displacement is too large (which will be the case for high-resolution glass plate scans). In this case, you will need to implement a faster search procedure such as an image pyramid. An image pyramid represents the image at multiple scales (usually scaled by a factor of 2) and the processing is done sequentially starting from the coarsest scale (smallest image) and going down the pyramid, updating your estimate as you go. It is very easy to implement by adding recursive calls to your original single-scale implementation. You should implement the pyramid functionality yourself using appropriate downsampling techniques.

***My Approach***: 

The essence of the pyramid algorithm is to have a few subgraphs which are downsampled by the factor of 2 and the depth of layer. For the i-th coarse level, this level's zoomed shift actually corresponds to the 2's (i-1) times the power real shift. And the shift of the current level base on the last level's shift. For i-th layer, the relative shift is calculated via metric optimization, and the absolute shift is exactly the 2*shift of the former layer. 

For the metric, I used **Normalized Cross-Correlation** (NCC) for relatively better performance.  

Another problem is: under the scenario of 'light brightness bias', if still using the raw pixel data for metric evaluation, for large pictures, there ***might be*** other shift that can have the optimal metric. This means that other wrong shift may be selected since still somehow causing the lowest metric. The examples are as follows:

<div style="display: flex; justify-content: space-around; align-items: center;">
        <figure>
            <img src="./img/emir.jpg" style="zoom:60%; height: auto;">
        </figure>
             <figure>
            <img src="./img/lugano.jpg" style="zoom:60%; height: auto;">
        </figure>
             <figure>
            <img src="./img/self_portrait.jpg" style="zoom:60%; height: auto;">
        </figure>
</div>



***BELL AND WHISTLE:*** The core way to deal with this is to use ***edge feature*** instead of  raw pixels. Since the brightness distribution of three channels sometimes can be biased, the trend of change of brightness, i.e., the 'derivative' of the brightness of one pixel, still holds. More over, the pattern we see in a picture, its edge are actually the pixels who has rather high derivative, since the style of color may suddenly change, causing the value of one channel suddenly change. And the best thing is: instead of forcing every pixel to align, now, we only have to align the most important feature, which is more robust. So using ***sobel*** function, I calculated the edge feature and then use this as the input for the pyramid and alignment algorithm.

The results are as follows:
<div style="display: flex; justify-content: space-around; align-items: center;">
        <figure>
            <img src="./img/church_color.jpg" style="zoom:60%; height: auto;">
            <figcaption>
                church.jpg 19.88s<br>
                Green offset: (dx=4, dy=25) <br>
                Red offset: (dx=-4, dy=58)
            </figcaption>
        </figure>
             <figure>
            <img src="./img/emir_color.jpg" style="zoom:60%; height: auto;">
            <figcaption>
                emir.jpg 30.26s<br>
                Green offset: (dx=24, dy=49) <br>
                Red offset: (dx=40, dy=107)
            </figcaption>
        </figure>
             <figure>
            <img src="./img/harvesters_color.jpg" style="zoom:60%; height: auto;">
            <figcaption>
                harvest.jpg 30.72s<br>
                Green offset: (dx=18, dy=60) <br>
                Red offset: (dx=14, dy=123)
            </figcaption>
        </figure>
</div>

<div style="display: flex; justify-content: space-around; align-items: center;">
        <figure>
            <img src="./img/icon_color.jpg" style="zoom:60%; height: auto;">
            <figcaption>
                icon.jpg 30.72s <br>
                Green offset: (dx=17, dy=41) <br>
                Red offset: (dx=23, dy=90)
            </figcaption>
        </figure>
             <figure>
            <img src="./img/italil_color.jpg" style="zoom:60%; height: auto;">
            <figcaption>
                italil.jpg 31.48s <br>
                Green offset: (dx=22, dy=38) <br>
                Red offset: (dx=36, dy=77)
            </figcaption>
        </figure>
             <figure>
            <img src="./img/lastochikino_color.jpg" style="zoom:60%; height: auto;">
            <figcaption>
                lastochikino.jpg 31.22s<br>
                Green offset: (dx=-1, dy=-3) <br>
                Red offset: (dx=-8, dy=76)
            </figcaption>
        </figure>
</div>

<div style="display: flex; justify-content: space-around; align-items: center;">
        <figure>
            <img src="./img/lugano_color.jpg" style="zoom:60%; height: auto;">
            <figcaption>
                lugano.jpg 33.08s<br>
                Green offset: (dx=-17, dy=41) <br>
                Red offset: (dx=-29, dy=92)
            </figcaption>
        </figure>
             <figure>
            <img src="./img/melons_color.jpg" style="zoom:60%; height: auto;">
            <figcaption>
                melons.jpg 32.10s<br>
                Green offset: (dx=10, dy=80) <br>
                Red offset: (dx=14, dy=177)
            </figcaption>
        </figure>
             <figure>
            <img src="./img/self_portrait_color.jpg" style="zoom:60%; height: auto;">
            <figcaption>
                self_portrait.jpg 32.71s<br>
                Green offset: (dx=29, dy=78) <br>
                Red offset: (dx=37, dy=176)
            </figcaption>
        </figure>
</div>

<div style="display: flex; justify-content: space-around; align-items: center;">
        <figure>
            <img src="./img/siren_color.jpg" style="zoom:60%; height: auto;">
            <figcaption>
                siren.jpg 31.49s<br>
                Green offset: (dx=-6, dy=49) <br>
                Red offset: (dx=-24, dy=96)
            </figcaption>
        </figure>
             <figure>
            <img src="./img/three_generations_color.jpg" style="zoom:60%; height: auto;">
            <figcaption>
                three_generations.jpg 28.14s<br>
                Green offset: (dx=13, dy=53) <br>
                Red offset: (dx=9, dy=111)
            </figcaption>
</div>


## Part 3 - Colorize Your Own Pictures

***Description / Hint***: 

The result of your algorithm on a few examples of your own choosing, downloaded from the [Prokudin-Gorskii collection](https://www.loc.gov/collections/prokudin-gorskii/?st=grid).

The results are as below:
<div style="display: flex; justify-content: space-around; align-items: center;">
        <figure>
            <img src="./img/Kapri_color.jpg" style="zoom:60%; height: auto;">
            <figcaption>
                Kapri.jpg 23.02s<br>
                Green offset: (dx=-14, dy=45) <br>
                Red offset: (dx=-11, dy=102)
            </figcaption>
        </figure>
             <figure>
            <img src="./img/Milanie_color.jpg" style="zoom:60%; height: auto;">
            <figcaption>
                Milanie.jpg 32.97s<br>
                Green offset: (dx=-18, dy=-11) <br>
                Red offset: (dx=-52, dy=2)
            </figcaption>
        </figure>
             <figure>
            <img src="./img/monastyr_color.jpg" style="zoom:60%; height: auto;">
            <figcaption>
                monastyr.jpg 31.61s<br>
                Green offset: (dx=-8, dy=49) <br>
                Red offset: (dx=-14, dy=113)
            </figcaption>
        </figure>
</div>

