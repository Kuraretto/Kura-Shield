Kura Shield DeepFake Forensic Lab

An Ai based solution for detecting Image/Video/Audio Deefakes

This Project is a webased forensic tool designed to detect AI-generated content. It uses ensemble of CNNs for visual analysis and a specialized architecture for auido deepfakes, the current stage of the project provides users with a 
user-friendly web interfasce designed using streamlit, offering proper visualization of the output and as we know AI/ML models are not accurate so we have also added the grad cam feature for you to futher analyse image/ videos and 
make your own judgement. 

Note: These models were only trained using 4 popular dataset so the training size is not very huge, I am actively working on improving the accuracy by gathering more dataset and retrain the models but this will take time as I am limited  
with the resources I have 

Datasets or Image and Video models:- Faceforensic++, ClebDF, Wild-Deepfake
Dataset used for Audio :- the-fake-or-real-dataset 

all these datasets are opensorce and can be used by anyone 

links: 

https://www.kaggle.com/datasets/amanrawat001/celeb-df-preprocessed

https://www.kaggle.com/datasets/maysuni/wild-deepfake

https://www.kaggle.com/datasets/adham7elmy/faceforencispp-extracted-frames

https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset

I am working on getting a much more robust dataset to train the models, will keep this repo updated on the progress 

Installation

1. Make sure you have conda installed in your pc, as its a mandatory to run this proejct, if you don't have it you can download it form here https://www.anaconda.com/download
2. Have Git installed 
3. Create a new folder, or go to the directory where you want to use the proejct, open command prompt and type "git clone https://github.com/Kuraretto/Kura-Shield.git"
4. This should automatically install all the files including all the models.
5. Now cd into Kura_Shield using the command "cd Kura-Shield"
6. Once you are into the project directory you have to create the Virtual environment which is required to use the project use this command to install all the requirements "conda env create -f DF_environment.yml" This will take time as it
   has to download all the libraries and requirements so please be patient.
7. Now to run the web app simply type in "streamlit run app.py"

   while doing this you might run into an error caused my numpy library ( i was not able to find a proper fix in the environment) so to avert this error, stop the app in the terminal and in the same terminal type in 

   pip install "numpy<2" 

   then run the app.py again using streamlit run app.py and it should open the webapp in your localhost

   The code will automatically load all the models so you don't have to worry about anything, the output from all the models will be stored in your main project directory under the folder "output" and then
   seperate folder for each type of output ie. video, audio and image

   
   
