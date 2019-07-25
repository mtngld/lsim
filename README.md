# Learn Stereo, Infer Mono: Siamese Networks for Self-Supervised, Monocular, Depth Estimation

`lsim_model.py` is based on Monodepth, with modifications to allows siamese and mirroring:

```
@inproceedings{monodepth17,
  title     = {Unsupervised Monocular Depth Estimation with Left-Right Consistency},
  author    = {Cl{\'{e}}ment Godard and
               Oisin {Mac Aodha} and
               Gabriel J. Brostow},
  booktitle = {CVPR},
  year = {2017}
}
```

### Requirements
* Tensorflow 1.13
* Pandas
* Numpy


### Data
Use `monodepth` excellent downloader in order to get the data for kitti.
For cityscapes see https://www.cityscapes-dataset.com/


### Reference
```
@inproceedings{goldman2019lsim,
  title={Learn Stereo, Infer Mono: Siamese Networks for Self-Supervised, Monocular, Depth Estimation},
  author={Goldman, Matan and Hassner, Tal and Avidan, Shai},
  booktitle={Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year=2019
}
```
