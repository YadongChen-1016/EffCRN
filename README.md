# FCRN
The unofficial Pytorch implementation of Fully Convolutional Recurrent Networks for Speech Enhancement (in 2020 ICASSP). </br></br>
I am interested in lightweight audio noise reduction models. There are some uncertainties in the reproduction process of FCRN models, and the validity of the models has not been verified at present. You feel free to contact me by [email📫](mailto:yadongchen2022@163.com) if you've got any problems.

## Requirement
numpy </br>
torchinfo </br>
pytorch >= 1.8.0 </br>

## Instruction

| Model | Param.| FLOPs/frame |
|:-----:|:-----:|:-----------:|
| FCRN  | 5.2M  |   631.5M    |
| FCRN* | 5.3M  |   632.8M    |
</br>
"*" represents my reproduction. </br>
I reproduced the model structure of FCRN and found that the number of parameters is approximately 5.32M, and the FLOPs per frame is about 632.77M. The reference score of FCRN is get form <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9914702">here🔍</a>👈. 

## Citations
```shell
@INPROCEEDINGS{9054230,
  author={Strake, Maximilian and Defraene, Bruno and Fluyt, Kristoff and Tirry, Wouter and Fingscheidt, Tim},
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Fully Convolutional Recurrent Networks for Speech Enhancement}, 
  year={2020},
  volume={},
  number={},
  pages={6674-6678},
  doi={10.1109/ICASSP40776.2020.9054230}
}

@INPROCEEDINGS{9914702,
  author={Strake, Maximilian and Behlke, Adrian and Fingscheidt, Tim},
  booktitle={2022 International Workshop on Acoustic Signal Enhancement (IWAENC)}, 
  title={Self-Attention With Restricted Time Context And Resolution In Dnn Speech Enhancement}, 
  year={2022},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/IWAENC53105.2022.9914702}
}
```
