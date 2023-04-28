<div id="top" align="center">
  
# Lite-Mono 
**A Lightweight CNN and Transformer Architecture for Self-Supervised Monocular Depth Estimation**
  [[paper link]](https://arxiv.org/abs/2211.13202)
  
  Ning Zhang*, Francesco Nex, George Vosselman, Norman Kerle
  
<a href="#license">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"/>
</a>  

<img src="./img/teaser_m.gif" width="100%" alt="teaser" align=center />

(Lite-Mono-8m 1024x320) 
  
</div>


## Table of Contents
- [Overview](#overview)
- [Results](#results)
  - [KITTI](#kitti) 
  - [Speed Evaluation](#speed-evaluation)
- [Data Preparation](#data-preparation)
- [Single Image Test](#single-image-test)
- [Evaluation](#evaluation)
- [Training](#training)
  - [Dependency Installation](#dependency-installation)
  - [Start Training](#start-training)
  - [Tensorboard Visualization](#tensorboard-visualization)
- [Citation](#citation)


## Overview
<img src="./img/overview.png" width="100%" alt="overview" align=center />


## Results
### KITTI
You can download the trained models using the links below.  

|     --model     | Params | ImageNet Pretrained | Input size |  Abs Rel  |   Sq Rel  |    RMSE   |  RMSE log | delta < 1.25 | delta < 1.25^2 | delta < 1.25^3 |
|:---------------:|:------:|:-------------------:|:----------:|:---------:|:---------:|:---------:|:---------:|:------------:|:--------------:|:--------------:|
|  [**lite-mono**](https://surfdrive.surf.nl/files/index.php/s/CUjiK221EFLyXDY)  |  3.1M  |         [yes](https://surfdrive.surf.nl/files/index.php/s/InMMGd5ZP2fXuia)         |   640x192  | 0.107 | 0.765 | 4.561 | 0.183 |   0.886  |    0.963   |    0.983   |
| [lite-mono-small](https://surfdrive.surf.nl/files/index.php/s/8cuZNH1CkNtQwxQ) |  2.5M  |         [yes](https://surfdrive.surf.nl/files/index.php/s/DYbWV4bsWImfJKu)         |   640x192  |   0.110   |   0.802   |   4.671   |   0.186   |     0.879    |      0.961     |      0.982     |
|  [lite-mono-tiny](https://surfdrive.surf.nl/files/index.php/s/TFDlF3wYQy0Nhmg) |  2.2M  |         yes         |   640x192  |   0.110   |   0.837   |   4.710   |   0.187   |     0.880    |      0.960     |      0.982     |
| [**lite-mono-8m**](https://surfdrive.surf.nl/files/index.php/s/UlkVBi1p99NFWWI) |  8.7M  |         [yes](https://surfdrive.surf.nl/files/index.php/s/oil2ME6ymoLGDlL)         |   640x192  |  0.101  |  0.729 | 4.454 |   0.178  |     0.897    |      0.965     |      0.983     |
|  [**lite-mono**](https://surfdrive.surf.nl/files/index.php/s/IK3VtPj6b5FkVnl)  |  3.1M  |         yes         |  1024x320  | 0.102 | 0.746 | 4.444 | 0.179 |   0.896  |    0.965   |    0.983   |
| [lite-mono-small](https://surfdrive.surf.nl/files/index.php/s/w8mvJMkB1dP15pu) |  2.5M  |         yes         |  1024x320  |   0.103   |   0.757   |   4.449   |   0.180   |     0.894    |      0.964     |      0.983     |
|  [lite-mono-tiny](https://surfdrive.surf.nl/files/index.php/s/myxcplTciOkgu5w) |  2.2M  |         yes         |  1024x320  |   0.104   |   0.764   |   4.487   |   0.180   |     0.892    |      0.964     |      0.983     |
| [**lite-mono-8m**](https://surfdrive.surf.nl/files/index.php/s/mgonNFAvoEJmMas) |  8.7M  |         yes         |  1024x320  |  0.097  |  0.710 | 4.309 |   0.174  |     0.905    |      0.967     |      0.984     |


### Speed Evaluation
<img src="./img/speed.png" width="100%" alt="speed evaluation" align=center />



## Data Preparation
Please refer to [Monodepth2](https://github.com/nianticlabs/monodepth2) to prepare your KITTI data. 


## Single Image Test
    python test_simple.py --load_weights_folder path/to/your/weights/folder --image_path path/to/your/test/image


## Evaluation
    python evaluate_depth.py --load_weights_folder path/to/your/weights/folder --data_path path/to/kitti_data/ --model lite-mono


## Training
#### dependency installation 
    pip install 'git+https://github.com/saadnaeem-dev/pytorch-linear-warmup-cosine-annealing-warm-restarts-weight-decay'
    
#### start training
    python train.py --data_path path/to/your/data --model_name mytrain --batch_size 12
    
#### tensorboard visualization
    tensorboard --log_dir ./tmp/mytrain

## Citation

    @article{zhang2022lite,
    title={Lite-Mono: A Lightweight CNN and Transformer Architecture for Self-Supervised Monocular Depth Estimation},
    author={Zhang, Ning and Nex, Francesco and Vosselman, George and Kerle, Norman},
    journal={arXiv preprint arXiv:2211.13202},
    year={2022}
    }
 
 
# Deep Learning Reproducibility Project
**Authors:** Fabian Gebben, Kieran de Klerk and Christiaan Theunisse

In this blog post we present the results from our study of the paper 'Lite-Mono: A Lightweight CNN and Transformer Architecture for Self-Supervised Monocular Depth Estimation' by Zhang et al[^lite-mono]. We did this for the course CS4240 Deep Learning of the Delft University of Technology. 

[^lite-mono]: Zhang, N., Nex, F., Vosselman, G., & Kerle, N. (2022). Lite-Mono: A Lightweight CNN and Transformer Architecture for Self-Supervised Monocular Depth Estimation.
<!-- How to add citations:
Some text where i want a reference [^fn15]

[^fn15]: APA style reference that will be included add the bottom of the blog
-->


## Table of contents
1. Introduction
2. Adapted training procedure
3. Using Dilated Neigbourhood Attention Convolutions instead of Dilated Convolutions
4. Alternative structures

## 1. Introduction 

<!-- Over here we introduce the paper and the stuff we are going to do -->

In this study, we will examine the Lite-Mono model proposed by Zhang et al [^lite-mono]. This paper introduces a new self-supervised model for estimating depth using images from a single camera. One of the most significant features of their model is its relatively lightweight architecture, which achieves comparable performance to other models. Our goal is to preserve this characteristic and maintain similar performance on the KITTI dataset. These are the two main areas of focus in our study.

In Chapter 2, we propose a method to adapt the training procedure to make it less computationally expensive while reducing dataset size in a structured manner, without sacrificing too much performance. This was essential for us since we had limited computational resources at our disposal.

In Chapter 3, we investigate the performance of a new convolutional layer in which we use a dilated neighbourhood attention transformer instead of a conventional dilated convolution. We aim to understand how this modification affects model performance.

Then in chapter 3, we will test a new convolutional layer where we use a dilated neighbourhood attention transformer instead of a normal dilated convolution to see how this affects the model performance.

Finally, in chapter 4, we compare the performances of two networks with slightly altered structures (by adding and removing a stage) to the baseline performance established in chapter 2.
## 2. Adapted training procedure

According to the paper, the Lite-Mono model was trained for 35 epochs on the KITTI dataset [^kitti_dataset]. This is dataset contains all types of raw car sensor data. Eigen et al. [^eigen_split] used the images taken by the two front stereo cameras to construct a dataset consisting only of images, where Zhou et al. created a subset suitable for monocular training, which is also used in the Lite-Mono paper. Since it is too time and resource consuming we can not train the model on the full dataset for all the changes to the model we want to test. Therefore, we will train on a subset of the data.

However, training on a subset will mostlikely result in overfitting the training data. To prevent this from happening, we will train for a smaller number of epochs. We will go over these topics one by one in this section. First, we will discuss and setup the resources we are going to use to train the model. Subsequently, a suitable subset of the data will be constructed and finally, we will determine the best number of epochs to train with on the smaller dataset.


#### Training resources

Since the original model took 15 hours to train on a Nvidia Titan XP, our own laptops will not be sufficient to run the training. Since Google Cloud platform gives free credits worth 300 USD to new users and the university provides 50 USD in credits per student, we decide to use a Google Cloud virtual machine (further refered to as Google VM). It took some time before we got everything working, but eventually we set up a Google VM with the following specifications:

```
vCPU: 16
MEMORY: 60 GB
GPU: NVIDIA Tesla P100
DISK: 250 GB
```
To run a training on the Google VM, we would connect via ssh. To make sure that the run was not aborted when the ssh connected was disconnected, we used the following command. 
```
nohup python -u train.py --data_path kitti_data/ \
    --model_name full_model_30_march_1605 --batch_size 12 --split eigen_zhou \ 
    --num_epochs 50 > full_model_30_march_1605_log.txt 2>&1 &
```
`nohup` to disconnect the process from the current shell session. `python -u` to run the python script and prevent it from buffering data that should be written to stdout. `train.py kitti_data/ --model_name full_model_30_march_1605 --batch_size 12 --split eigen_zhou --num-epochs 50` is the training file where we parse some additional parameters. `> full_model_30_march_1605_log.txt` to write the stdout stream to a text file. `2>&1` to write the stderr stream to the affore mentioned text file. `&` to run the process in the background.

One problem we could not solve, was that the ssh service would break down on the VM some time after we started the training. So it would become impossible to connect to the VM by ssh until the VM was reset. However, this ends all running processes, so we could only do this after the training ended.

#### How to construct a suitable subset

To create the smaller dataset we will use a fourth of the original dataset, which consists of 39810 images for training, 4424 for evaluation and 697 for testing. So the new training and evaluation set will respectively consist of 9950 and 1106 images. The images for testing will remain the same to have the best comparison against the original model performance.

The KITTI dataset contains 56 different scenes, where 28 scenes are used for training and 28 for testing in the split proposed by Eigen et al[^eigen_split]. Futhermore, images where the car is stationary are removed. After these steps about 20,000 unique poses remain for training, but since both the images from the left and the right camera are used, the resulting set is a shuffled list of 40,000 images.

That being the case, the subset we are going to construct should have the same distribution over the scenes and be shuffled. To create as much variance in the data and reduce overfitting, we also need to ensure that we do not include both images from a stereo image set (so both the left and right camera image from the same car position). And finally, we should have the same number of images from the left as from the right camera in the dataset to prevent overfitting on a certain camera, although the difference might be negligible.

We wrote a script to perform this task for the training and the validation set. Both the [script](https://github.com/christiaantheunisse/Lite-Mono/blob/107f1707cbaf7dda105e10c8903b06aaad426de2/create_smaller_dataset.ipynb) and the resulting [training](https://github.com/christiaantheunisse/Lite-Mono/blob/107f1707cbaf7dda105e10c8903b06aaad426de2/splits/eigen_reduced/training_files.txt) and [validation](https://github.com/christiaantheunisse/Lite-Mono/blob/107f1707cbaf7dda105e10c8903b06aaad426de2/splits/eigen_reduced/val_files.txt) set can be found on our Github. 

#### Right number of epochs

###### Training of the model with the full dataset
To ensure that our training setup was able to get the same performance as performance mentioned in the paper, we trained the model with the full dataset once for 50 epochs. The performance on the test set after each epoch is visualized in the figure below. Futhermore, the maximum values are displayed in a table. The model in the paper is trained for 35 epochs with and without pretraining for 30 epochs on ImageNet. It can be seen that the pretrained model performs only slightly better. We did no pretrain our model on ImageNet, but we trained it for 50 epochs on the KITTI dataset to see if we can get the same performance as the one pretrained on ImageNet. The resulting performance on the test set is given in the table below. The first four table entries: *Abs Rel, Sq Rel, RMSE* and *RMSE log* are different measures for the depth error, so these should be as low as possible. The last three entries give a measure for the depth accuracy, hence these should be as high as possible. These metrics are commonly used in depth estimation and more information can be found in [^performance_metrics].

![](https://i.imgur.com/aAPOVF6.png)

|                                      | Abs Rel | Sq Rel | RMSE   | RMSE log |  $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---                                 | :----:  | :----: | :----: | :----:   |  :----:          |    :----:       |       :----:        |
|Original Paper*                       |  0.107  | 0.765  | 4.561  | 0.183    |  0.886           | 0.963           | 0.983               |
|Original Paper                        |  0.121  | 0.876  | 4.918  | 0.199    |  0.859           | 0.953           | 0.980               |
|Our training full DS (after 35 epochs)|  0.131  | 0.963  | 5.081  | 0.208    |  0.845           | 0.948           | 0.978               |
|Our training full DS (after 50 epochs)|  0.125  | 0.930  | 4.966  | 0.203    |  0.854           | 0.952           | 0.979               |


*\*With pretraining on ImageNet for 30 epochs*

We see that our model performs slightly worse than the model trained in the paper. The author of the paper mentions on [Github](https://github.com/noahzn/Lite-Mono/issues/4#issuecomment-1477535597) that the performance varies among different training runs due to the stochasticity involved in the process. Some runs he performed later on performed better than the paper on a few metrics, but another time the training was not able to converge. However, this stochacitiy is, most likely, not responsible for all the differences, since we see that our training performs worse on all measures, especially after 35 epochs. The differences are most likely caused by a slight difference in, for example, the learning rate scheduling, which might slightly differ from the one uploaded to Github. The CUDA and Pytorch version can also influence the results. However, for the purpose of this project the performance is good enough, since we are going to train on a smaller dataset, which will make the performance drop anyway.

Besides, we can conclude that training for another 15 epochs hardly increases the performance, especially the accuracy measures. On the other hand, in the graphs we can see that the performance is still increasing which suggest that training the model for even more epochs might result in an increased performance.

###### Training of the model with the reduced dataset

Now we are going to train our model with our own (smaller) dataset for 35 epochs like the models in the paper. The performance on the test set is plotted in the figure below. The goal is to find the number of epochs where the model is on the verge of overfitting the training data, thus the point where the error on the test data increases and the test accuracy drops.

![](https://i.imgur.com/72RcKqS.png)

The best score and specific epoch after which the best score is obtained, are given for every metric in the table below. The scores are compared to the performance mentioned in the paper for the full dataset without pretraining and to the performance of our training of the full model after 35 epochs.

|                                      | Abs Rel | Sq Rel | RMSE   | RMSE log |  $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---                                 | :----:  | :----: | :----: | :----:   |  :----:          |    :----:       |       :----:        |
|Original Paper                        |  0.121  | 0.876  | 4.918  | 0.199    |  0.859           | 0.953           | 0.980               |
|Our training full DS (35 epochs)|  0.131  | 0.963  | 5.081  | 0.208    |  0.845           | 0.948           | 0.978               |
|Our training reduced DS (35 epochs)|  0.158  | 1.178  | 5.522  | 0.231    |  0.789           | 0.928           | 0.972               |
|*Epoch of best performance for reduced model*| *29*    | *20*   | *28*   | *29*     |  *29*            | *29*            | *29*                |

From the above table, it can be concluded that the performance is significantly lower for the reduced model, especially the depth error metrics. This might have to do with fact that the depth error follows a continuous scale, so a slight increase over all the predictions results in the same increase for the total error. On the other hand, the depth accuracy checks if a prediction lies within a certain relative distance from the ground truth. Consequently, a slight increase in the error hardly influences the accuracy, since the predictions might still lie within this relative distance. The table shows that this is particularly true for the accuracies with bigger margins.

To get a better idea of the impact of the performance drop, the performance is compared against the other models mentioned in the Lite-Mono paper. It can be seen that the performance of the model trained on the reduced dataset is similar to for example GeoNet[^GeoNet] and DDVO[^DDVO], both developed in 2018 but which have about 10 times as much parameters. Therefore, it can be concluded that training the model on the reduced dataset is a valid option to do ablation studies or make other changes to the model. The performance mentioned for the model on the reduced dataset in the table above will be the new benchmark. Given that all the metrics got their best performance within 29 epochs and the graphs show a decreasing performances in the last epochs, we will from now on train the model for 30 epochs on the reduced dataset to save computational power. Training the model with the reduced dataset will take about 6 ours on our Google VM.

### 3. Using Dilated Neigbourhood Attention Convolutions instead of Dilated Convolutions

Depth estimation from a 2D image is a challenging task in computer vision, and it requires the model to capture both local and global context information from the image. Dilated convolutions and neighbourhood attention mechanisms have both been widely used in deep learning architectures, with each having its own strengths and limitations. Dilated convolutions are effective at increasing the receptive field without increasing the number of parameters, while neighbourhood attention mechanisms allow the model to selectively attend to important local features in the input. However, both approaches have limitations in capturing long-range dependencies and capturing multi-modal distributions in the input.

To address these limitations, Dilated Neighbourhood Attention Transformers (DiNAT[^Dinat]) can be employed. DiNATs combine the benefits of dilated convolutions, neighbourhood attention mechanisms, and transformer architecture, to capture both local and global features and model long-range dependencies in the input.

By combining these different mechanisms, DiNATs can enhance the model's ability to capture complex patterns in the data and improve its performance in tasks that require both local and global information. Furthermore, DiNATs can handle multi-modal distributions in the input, making them suitable for tasks such as image segmentation, where multiple objects may be present in the same image.

The Lite-Mono approach used normal dilated convolutions. We thought it would be interesting to see what would happen to the performance if we instead use these DiNATs proposed by Hassani and Shi[^Dinat]. 

#### 3.1 Implementation

There are multiple convolutions happening inside of the Lite-Mono model. So we must first establish which of the convolutions we will change to a DiNat. In the following figure, the structure of model is given:

![](https://i.imgur.com/mauofVt.png)[^Dinat]

We will replace the dilated convolutions in all three marked CDC blocks in the figure above by the DiNATs. To realise this we will use the PyPi NATTEN package [^NATTEN]. The way we implemented this is shown in the figure below:
![](https://i.imgur.com/a3kpL9F.png)

A new class, NA2D, has been created to replace the functionality of the old CDilated class. It is structured in the same way to keep the adjustments needed to use it to a minimum. The function is then called as the convolutional layer in the DilatedConv class. However, due to the fact that the NATTEN function requires channels to be last instead of first (as is the case with the Conv2D function), the layers are permuted in order. After passing through the DiNAT layer, they are permuted back.

#### 3.2 Results
The model is trained on the reduced dataset that was presented in chapter 2 with a batch size of 12. We used a number of epochs of 30, since we saw the highest performance at this number of epochs in chapter 2 and want to prevent overfitting. Training the model gave the following results:

![](https://i.imgur.com/08Vtv09.png)

As we can see in the figure above, the performance and shapes seem very similair to the results obtained with the original model and the reduced dataset in chapter 2. One key difference to note is that there is no overfitting happening, which is probably due to the reduced number of epochs.


|                                      | Abs Rel | Sq Rel | RMSE   | RMSE log |  $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---                                 | :----:  | :----: | :----: | :----:   |  :----:          |    :----:       |       :----:        |
|Reduced DS Dilated Convolutions (30 epochs)|  0.158  | 1.178  | 5.522  | 0.231    |  0.789           | 0.928           | 0.972               |
|Reduced DS new DiNAT (30 epochs)|  0.161  | 1.223  | 5.594  | 0.233    |  0.783           | 0.926           | 0.972               |
|*Epoch of best performance for DiNAT model*| *29*    | *20*   | *29*   | *29*     |  *28*            | *29*            | *25*                |

<!-- When looking at the results in the table with the original model with the dilated convolutions and the new model with the DiNATs, we see that the performance is again very similair. However, the DiNAT model seems to perform slightly worse. This seems to indicate that the integration of a DiNAT instead of a dilated convolution does not improve performance. However, it does not necessarily mean that the performance of the Lite-Mono could not improve using a DiNAT. We used specific hyperparameters, which could be tuned to increase the performance. For example, we used the same kernel size and dilation as for the original model, which might not make sense for this new DiNAT layer. Due to resource limitations, this is something we could unfortunately not look into any further. In conclusion, the DiNAT might have some potential to increase performance of the Lite-Mono model, but this is not something we could directly observe. The performance was very similar, but slightly worse.  -->

When examining the results in the table comparing the original model with dilated convolutions to the new model with DiNATs, we can see that the performance is once again very similar. However, the DiNAT model appears to perform slightly worse, suggesting that integrating a DiNAT instead of a dilated convolution may not necessarily improve performance. That being said, it is important to note that the performance of the Lite-Mono DiNAT model could potentially be improved by tuning hyperparameters such as kernel size and dilation rate. We now used the same ones as for the dilated convolution model, but these might not work as well for the DiNAT model. Unfortunately, resource limitations prevented us from exploring this further. 

Another drawback compared to the original model was that it took slightly longer to train. The new model took approximately 6.5 hours to complete 30 epochs, whereas the original model took only 6 hours for the same number of epochs. We believe that this increase in training time may be due to the double permuting that is necessary to use the DiNAT layer in the new model.

In conclusion, while the DiNAT may have the potential to increase performance, we did not observe a direct improvement in this study. Overall, the performance of the two models was very similar, although the DiNAT model performed slightly worse.

### 4. Alternative structures
Zhang [^lite-mono] gives very little justification for the exact structure of the proposed network. Therefore, in this chapter present how we explored the impact slightly varying the network's structure has on the overall performance of the algorithm. Firstly, stage 4 (see picture below) was removed entirely in order to compare the performance of this new structure to the baseline. Next, we repeated this process but for a network with a 5^th^ stage added below stage 4. This stage takes inputs of size $\frac H {16} \times \frac W {16} \times C_4$ and has an output of size $\frac H {32} \times \frac W {32} \times C_5$. 
![](https://i.imgur.com/SWwTRS8.png)**Overview of Lite-Mono's structure [^lite-mono]**
Section 4.1 will discuss how we implemented the removal of stage 4. Section 4.2 will then explain how we added a 5^th^ stage. Finally section 4.3 will give an overview of the results we obtained after training and testing both of these slightly altered networks.
#### 4.1 Removing a stage
It turned out to be relatively straight forward to remove stage 4 as the main challenge was finding all the instances where the algorithm iterates over a set of blocks or stages or where it defines a list that is meant to be iterated over for each stage. Therefore, the main, relevant changes were made to [depth_decoder.py](https://github.com/christiaantheunisse/Lite-Mono/blob/15173ca9bb24a0209e0115f165978c5a91bc5fff/networks/depth_decoder.py), [depth_encoder.py](https://github.com/christiaantheunisse/Lite-Mono/blob/15173ca9bb24a0209e0115f165978c5a91bc5fff/networks/depth_encoder.py) and [options.py](https://github.com/christiaantheunisse/Lite-Mono/blob/15173ca9bb24a0209e0115f165978c5a91bc5fff/options.py). Whenever a for loop is executed in the code, the range over which it iterates was decreased by 1. This can be observed most clearly by looking at the differences between [depth_encoder.py on the 3_stages branch](https://github.com/christiaantheunisse/Lite-Mono/blob/15173ca9bb24a0209e0115f165978c5a91bc5fff/networks/depth_encoder.py) and the corresponding file on the [main branch](https://github.com/christiaantheunisse/Lite-Mono/blob/main/networks/depth_encoder.py). Aditionally, when you take a look at the `__init__` function for the `LiteMono` class in [depth_encoder.py](https://github.com/christiaantheunisse/Lite-Mono/blob/15173ca9bb24a0209e0115f165978c5a91bc5fff/networks/depth_encoder.py), you will see how we removed an entry in every list or array that has an entry for each full stage (a stage with a downsample block, a CDC block and a LGFI block) while keeping the same structure (i.e. `self.dilation = [[1, 2, 3], [1,2,3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]` being changed to `self.dilation = [[1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]`).

#### 4.2 Adding a stage
Adding a 5^th^ stage was relatively similar to the process described in section 4.1 but a little more complex. Making the necessary alterations to the loops follows the same process, however, this time, the range has to be increased by 1 instead of substracting one. The lists are where it gets more complicated: most of the changes to lists that have been altered are relatively intuitive. For example the lists that had the same value for each entry except the last one were kept like this by adding another entry with the same value in the middle of the lists. In certain lists there seems to be some pattern that wasn't clearly explained in the paper (i.e. `self.num_ch_enc` in the `__init__` function in [depth_encoder](https://github.com/christiaantheunisse/Lite-Mono/blob/5_stages/networks/depth_encoder.py)), these had to be estimated based on reasonable intuition. 

#### 4.3 Results
The models obtained after the processes described above were trained for 30 epochs and the performance statistics can be seen below in the plot and table where we compare them to the baseline structure (4 stages) proposed by Zhang [^lite-mono] which was trained on the same adapted program described in chapter 2. 

![](https://i.imgur.com/badfwu9.png)**Plots showing the performance statistics over the epochs. Dashed lines are for 3, full lines for 4 and dotted lines for 5 stages**

From the figure below, we can deduce that (at least in this instance), removing the 4^th^ stage has negative effects on performance while adding a 5^th^ stage seems to have some positive effects on performance. However, the performance gain by performing the latter alteration is not very significant and could be chalked up to the inherent stochastic behaviour of training large neural networks, while the network with 3 stages does show a relatively significant loss in performance. 
When we look at the table below however, we do see better performance from the largest network accross the board (apart from its minimal RMSE being equal to the baseline's) which would indicate that it is not entirely due to stochasticity.

**Table containing the maximal values obtained over the course of 30 epochs of training with the correspongin epoch in parentheses**
|    Stages      | Abs  Rel | Sq Rel | RMSE | RMSE log | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
|----------|----------|--------|------|----------|-----------------|-------------------|-------------------|
|3|0.162 *(29)*|1.223 *(29)*|5.552 *(26)*|0.233 *(29)*|0.785 *(26)*|0.925 *(30)*|0.972 *(30)*|
|4|0.158 *(29)*|1,178 *(20)*|5.522 *(28)*|0.231 *(29)*|0.789 *(29)*|0.928 *(29)*|0.972 *(29)*|
|5|0.155 *(28)*|1.149 *(13)*|5.522 *(30)*|0.227 *(28)*|0.797 *(28)*|0.930 *(28)*|0.974 *(28)*|

To conclude, removing a stage and adding a stage decreases and increases performance respectively. Whether the gain performance of the latter case is worth the added computational complexity and the higher memory requirements might depend on the specific usecase and provides for an interesting follow-up research question. 

<!-- [^Lite-mono]: Zhang, N., Nex, F., Vosselman, G., & Kerle, N. (2022). Lite-Mono: A Lightweight CNN and Transformer Architecture for Self-Supervised Monocular Depth Estimation. arXiv preprint arXiv:2211.13202. -->
[^kitti_dataset]: Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. Int. J. Robot. Res., 32(11):1231–1237, 2013. 
[^eigen_split]: David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using a multi-scale deep network. NeurIPS, 27, 2014. 
[^performance_metrics]: Cadena, C., Latif, Y., & Reid, I. D. (2016, October). Measuring the performance of single image depth estimation methods. In 2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (pp. 4150-4157). IEEE.
[^GeoNet]: Zhichao Yin and Jianping Shi. Geonet: Unsupervised learning of dense depth, optical flow and camera pose. In CVPR, 2018.
[^DDVO]: Chaoyang Wang, Jose Miguel Buenaposada, Rui Zhu, and Simon Lucey. Learning depth from monocular videos using direct methods. In CVPR, 2018.
[^Dinat]: Hassani, A., & Shi, H. (2022). Dilated neighborhood attention transformer. arXiv preprint arXiv:2209.15001.
[^NATTEN]: Natten. (2023, March 21). PyPI. https://pypi.org/project/natten/
#### Who was responsible for which part?
 - Christiaan Theunisse: setting up the Google enviroment and adapting the training procedure
 - Fabian Gebben: using dilated neighbourhood attention instead of normal dilated convolutions 
 - Kieran de Klerk: evaluating the performance of slightly altered versions of the algorithm



