# EPIC-KITCHENS-100 UDA Challenge Source Code
This repository contains the code used to produce the baseline results (TA3N) for the EPIC-KITCHENS-100 UDA Challenge. 

Some modifications have been made to the original TA3N code base to produce results on EPIC-KITCHENS-100 including:
1. Multiple classification heads to produce predictions for verb and nouns.
2. Modified dataloader to match structure 
3. Modified training scripts to run the EPIC-KITCHENS-100 UDA baselines based. These have been modified from`script_train_val.sh` in the original repository.

The original TA3N [code](https://github.com/cmhungsteve/TA3N) and [ICCV publication](http://openaccess.thecvf.com/content_ICCV_2019/html/Chen_Temporal_Attentive_Alignment_for_Large-Scale_Video_Domain_Adaptation_ICCV_2019_paper.html) can be found in the provided hyperlinks.

---
## Usage
We provide modified training scripts for TA3N to replicate EPIC-100 UDA Challenge results.

1. Install dependecies 
    * `conda env create -f environment.yml`
   
2. Download annotations and features.
    * Annotations should be downloaded to the folder `./annotations` and extracted features to `./data`.
         * Features: https://www.dropbox.com/sh/hsf8assfb9pzjos/AABqlWHx3YQATJZ_Gqnnhsj1a?dl=0
         * Annotations: https://github.com/epic-kitchens/epic-kitchens-100-annotations/tree/master/UDA_annotations
    * Alteratively variables in the bash scripts (`path_labels_root="annotations"`, `path_data_root="data`) can be modified to match the location of the labels/features on your machine.


3. Replicate results with the provided bash scripts:
    * `./script_test_ta3n.sh` to re-produce TA3N domain adaptation results
    * `./script_test_source_only.sh` to re-produce source-only results

3. Each script will generate a submission script`test.json` which can be uploaded to the UDA Codalab challenge to replicate the results. To re-train models, set `train=true` in the bash scripts.

For futher details please see original source code for Temporal Attentive Alignment: https://github.com/cmhungsteve/TA3N

---
## Acknowledgements
If you find this repository useful, please cite both the EPIC-KITCHENS dataset papers and the TA3N authors publications:

* EPIC-KITCHENS
```
@article{Damen2020RESCALING,
   title={Rescaling Egocentric Vision},
   author={Damen, Dima and Doughty, Hazel and Farinella, Giovanni Maria  and and Furnari, Antonino 
           and Ma, Jian and Kazakos, Evangelos and Moltisanti, Davide and Munro, Jonathan 
           and Perrett, Toby and Price, Will and Wray, Michael},
           journal   = {CoRR},
           volume    = {abs/2006.13256},
           year      = {2020},
           ee        = {http://arxiv.org/abs/2006.13256},
} 
@INPROCEEDINGS{Damen2018EPICKITCHENS,
   title={Scaling Egocentric Vision: The EPIC-KITCHENS Dataset},
   author={Damen, Dima and Doughty, Hazel and Farinella, Giovanni Maria  and Fidler, Sanja and
           Furnari, Antonino and Kazakos, Evangelos and Moltisanti, Davide and Munro, Jonathan
           and Perrett, Toby and Price, Will and Wray, Michael},
   booktitle={European Conference on Computer Vision (ECCV)},
   year={2018}
}
```

* Temporal Attentive Alignment
```
@article{chen2019taaan,
title={Temporal Attentive Alignment for Large-Scale Video Domain Adaptation},
author={Chen, Min-Hung and Kira, Zsolt and AlRegib, Ghassan and Yoo, Jaekwon and Chen, Ruxin and Zheng, Jian},
booktitle = {International Conference on Computer Vision (ICCV)},
year={2019},
url={https://arxiv.org/abs/1907.12743}
}

@article{chen2019temporal,
title={Temporal Attentive Alignment for Video Domain Adaptation},
author={Chen, Min-Hung and Kira, Zsolt and AlRegib, Ghassan},
booktitle = {CVPR Workshop on Learning from Unlabeled Videos},
year={2019},
url={https://arxiv.org/abs/1905.10861}
}
```
