# <B> Generating Diverse and Natural 3D Human Motions from Text (CVPR 2022)<b>
## [[Project Page]](https://github.com/EricGuo5513/text-to-motion/blob/main/docs/teaser_image.png) [[Paper]](https://github.com/EricGuo5513/text-to-motion/blob/main/docs/teaser_image.png)

![teaser_image](https://github.com/EricGuo5513/text-to-motion/blob/main/docs/teaser_image.png)
  
  Given a textual description for example, *"the figure rises from a lying position and walks in a counterclockwise circle, and then lays back down the ground"*, our approach generates a diverse set of 3d human motions that are faithful to the provided text.
  
## Python Virtual Environment

Anaconda is recommended to create this virtual environment.
  
  ```sh
  conda create -f environment.yaml
  source activate text2motion_pub
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
  
  <b>If you just want to play our pre-trained models, you don't need to download datasets.<b>
  ### <b> Datasets <b>
  We are using two 3D human motion-language dataset: HumanML3D and KIT-ML. For both datasets, you could find the details as well as download link [here](https://github.com/EricGuoICT/HumanML3D).   
  Please note you don't need to clone that git repository, since all related codes have already been included in this git project.
  
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
  ```
 ### <b> Pre-trained Models <b>
  Create a checkpoint folder to place pre-traine models:
  ```sh
  mkdir ./checkpoints
  ```
    
 #### Download models for HumanML3D from [here](https://drive.google.com/file/d/1DSaKqWX2HlwBtVH5l7DdW96jeYUIXsOP/view?usp=sharing). Unzip and place them under checkpoint directory, which should be like
```
./checkpoints/t2m/
./checkpoints/t2m/Comp_v6_KLD01/           # Text-to-motion generation model
./checkpoints/t2m/Decomp_SP001_SM001_H512/ # Motion autoencoder
./checkpoints/t2m/length_est_bigru/        # Text-to-length sampling model
./checkpoints/t2m/text_mot_match/          # Motion & Text feature extractors for evaluation
 ```
 #### Download models for KIT-ML [here](https://drive.google.com/file/d/1tX79xk0fflp07EZ660Xz1RAFE33iEyJR/view?usp=sharing)
    
 ## Training
    
    
 ## Testing
    
 
