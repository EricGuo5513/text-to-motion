# Generating Diverse and Natural 3D Human Motions from Text (CVPR 2022)
## [[Project Page]](https://ericguo5513.github.io/text-to-motion) [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.pdf)

![teaser_image](https://github.com/EricGuo5513/text-to-motion/blob/main/docs/teaser_image.png)
  
  Given a textual description for example, *"the figure rises from a lying position and walks in a counterclockwise circle, and then lays back down the ground"*, our approach generates a diverse set of 3d human motions that are faithful to the provided text.
  
## Python Virtual Environment

Anaconda is recommended to create this virtual environment.
  
  ```sh
  conda create -f environment.yaml
  conda activate text2motion_pub
  ```
  
If you cannot successfully create the environment, here is a list of required libraries:
  ```
  Python = 3.7.9   # Other version may also work but are not tested.
  PyTorch = 1.6.0 (conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch)  #Other version may also work but are not tested.
  scipy
  numpy
  tensorflow       # For use of tensorboard only
  spacy
  tqdm
  ffmpeg = 4.3.1   # Other version may also work but are not tested.
  matplotlib = 3.3.1
  ```
  
  After all, if you want to generate 3D motions from customized raw texts, you still need to install the language model for spacy. 
  ```sh
  python -m spacy download en_core_web_sm
  ```
  
  ## Download Data & Pre-trained Models
  
  **If you just want to play our pre-trained models, you don't need to download datasets.**
  ### Datasets
  We are using two 3D human motion-language dataset: HumanML3D and KIT-ML. For both datasets, you could find the details as well as download link [[here]](https://github.com/EricGuo5513/HumanML3D).   
  Please note you don't need to clone that git repository, since all related codes have already been included in current git project.
  
  Download and unzip the dataset files -> Create a dataset folder -> Place related data files in dataset folder:
  ```sh
  mkdir ./dataset/
  ```
  Take HumanML3D for an example, the file directory should look like this:  
  ```
  ./dataset/
  ./dataset/HumanML3D/
  ./dataset/HumanML3D/new_joint_vecs/
  ./dataset/HumanML3D/texts/
  ./dataset/HumanML3D/Mean.mpy
  ./dataset/HumanML3D/Std.npy
  ./dataset/HumanML3D/test.txt
  ./dataset/HumanML3D/train.txt
  ./dataset/HumanML3D/train_val.txt
  ./dataset/HumanML3D/val.txt  
  ./dataset/HumanML3D/all.txt 
  ```
 ### Pre-trained Models
  Create a checkpoint folder to place pre-traine models:
  ```sh
  mkdir ./checkpoints
  ```
    
 #### Download models for HumanML3D from [[here]](https://drive.google.com/file/d/1DSaKqWX2HlwBtVH5l7DdW96jeYUIXsOP/view?usp=sharing). Unzip and place them under checkpoint directory, which should be like
```
./checkpoints/t2m/
./checkpoints/t2m/Comp_v6_KLD01/           # Text-to-motion generation model
./checkpoints/t2m/Decomp_SP001_SM001_H512/ # Motion autoencoder
./checkpoints/t2m/length_est_bigru/        # Text-to-length sampling model
./checkpoints/t2m/text_mot_match/          # Motion & Text feature extractors for evaluation
 ```
 #### Download models for KIT-ML [[here]](https://drive.google.com/file/d/1tX79xk0fflp07EZ660Xz1RAFE33iEyJR/view?usp=sharing). Unzip and place them under checkpoint directory.
    
 ## Training Models
 
 All intermediate meta files/animations/models will be saved to checkpoint directory under the folder specified by argument "--name".
 ### Training motion autoencoder
 #### HumanML3D
```sh
python train_decomp_v3.py --name Decomp_SP001_SM001_H512 --gpu_id 0 --window_size 24 --dataset_name t2m
```
#### KIT-ML
```sh
python train_decomp_v3.py --name Decomp_SP001_SM001_H512 --gpu_id 0 --window_size 24 --dataset_name kit
```

### Train text2length model:
#### HumanML3D
```sh
python train_length_est.py --name length_est_bigru --gpu_id 0 --dataset_name t2m
```
#### KIT-ML
```sh
python train_length_est.py --name length_est_bigru --gpu_id 0 --dataset_name kit
```
### Training text2motion model:
#### HumanML3D
```sh
python train_comp_v6.py --name Comp_v6_KLD01 --gpu_id 0 --lambda_kld 0.01 --dataset_name t2m
```
#### KIT-ML
```sh
python train_comp_v6.py --name Comp_v6_KLD005 --gpu_id 0 --lambda_kld 0.005 --dataset_name kit
```
### Training motion & text feature extractors:
#### HumanML3D
```sh
python train_tex_mot_match.py --name text_mot_match --gpu_id 1 --batch_size 8 --dataset_name t2m
```
#### KIT-ML
```sh
python train_tex_mot_match.py --name text_mot_match --gpu_id 1 --batch_size 8 --dataset_name kit
```
    
## Generating and Animating 3D Motions (HumanML3D)
#### Sampling results from test sets
```sh
python eval_comp_v6.py --name Comp_v6_KLD01 --est_length --repeat_time 3 --num_results 10 --ext default --gpu_id 1
```
where *--est_length* asks the model to use sampled motion lengths for generation, *--repeat_time* gives how many sampling rounds are carried out for each description. This script will results in 3x10 animations under directory *./eval_results/t2m/Comp_v6_KLD01/default/*.

#### Sampling results from customized descriptions
```sh
python gen_motion_script.py --name Comp_v6_KLD01 --text_file input.txt --repeat_time 3 --ext customized --gpu_id 1
```
This will generate 3 animated motions for each description given in text_file *./input.txt*.

If you find problem with installing ffmpeg, you may not be able to animate 3d results in mp4. Try gif instead.

## Quantitative Evaluations
```sh
python final_evaluation.py 
```
This will evaluate the model performance on HumanML3D dataset by default. You could also run on KIT-ML dataset by uncommenting certain lines in *./final_evaluation.py*. The statistical results will saved to *./t2m_evaluation.log*.

### Misc
 Contact Chuan Guo at cguo2@ualberta.ca for any questions or comments.
