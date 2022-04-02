# Noise-Sensitive Adversarial Learning for Weakly Supervised Salient Object Detection
----------------------------------------------
Accepted paper in IEEE Trans on Multimedia, 'Noise-Sensitive Adversarial Learning for Weakly Supervised Salient Object Detection', Yongri Piao, Wei Wu, Miao Zhang, Yongyao Jiang and Huchuan Lu.

![framework](https://github.com/wuweia123/IEEE-TMM-NSALWSS/blob/main/fig/framework.png)

## Prerequisites
### Environment
* Windows 10
* Torch 1.4.0
* CUDA 11.1
* Python 3.6.5

### Training data
link: https://pan.baidu.com/s/1n4YGVRhNabM5td4et9o5sw    code: wnvl

## Training
### 1st training stage
Case1 : Update soon

Case2 : We upload our 1st pseudo labels in Training data, you can directly use our offered <stage1_training_map> as pseudo labels for convenience. 

### 2nd training stage
#### setting the training data to the proper root as follows:

```
NSALWSS -- datasets -- DUTS_pseudo -- DUTS-TR-Image -- 10553 samples
                
                                   -- stage1_training_map -- 10553 pseudo labels
                
                                   -- stage2_training_map -- 10553 pseudo labels (not necessary but we also offered stage2's pseudo labels for convenience)
```

#### training

```Run train.py```

## testing
```Run test_code.py```
You need to configure your desired testset in ```--test_root```

The evaluation code can be found in [here](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox).

## Saliency maps & Checkpoint
We offer our saliency maps and checkpoints.
### Saliency maps
link: https://pan.baidu.com/s/1Dyhy107oQTow1UN1Wg9-KA    code: 1kfn
### Checkpoints
link: https://pan.baidu.com/s/1aMwHkQb-9C2YmM_P-j8f-A    code: 32fi
## Contact me
If you have any questions, please contact me: [1157008667@qq.com].

## Citation
We really hope this repo can contribute the conmunity, and if you find this work useful, please use the following citation:

```
@ARTICLE{9716868,
  author={Piao, Yongri and Wu, Wei and Zhang, Miao and Jiang, Yongyao and Lu, Huchuan},
  journal={IEEE Transactions on Multimedia}, 
  title={Noise-Sensitive Adversarial Learning for Weakly Supervised Salient Object Detection}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMM.2022.3152567}}
```
