# Hindi-Character-Recognition

### The app is live on huggingface spaces! Try it -> [Gradio-HCR](https://huggingface.co/spaces/abhiswain/Gradio-Hindi-Character-Recognition)

> I also created this project as to understand the difference [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/) makes v/s writing the code in pure Pytorch. You can see it for yourself: 
> 1. Code in Pytorch Lightning -> [Lightning.py](https://github.com/Abhiswain97/Hindi-Character-Recognition/blob/master/Lightning.py)
> 2. Code in pure Pytorch -> all the files under `src` folder


## Running the app locally:

You can run the streamlit and gradio app locally. 

1. Install the requirements: `pip install -r requirements.txt`

2. Now, just do `streamlit run app.py` or `gradio gradio_app.py`

### Streamlit demo

![App](https://user-images.githubusercontent.com/54038552/210263132-2e95ad65-5049-4a8c-b453-2616cbc4df20.gif)

### Gradio demo

![GradioApp](https://user-images.githubusercontent.com/54038552/220750032-5f7ba8dd-2f70-418c-9752-53bf25aaf8d1.gif)

## Training the model (Optional)

1. Install the requirements: `pip install -r requirements.txt`

2. Hindi Character Recognition

    Getting the data:
    - Download the data from [here](https://www.kaggle.com/datasets/suvooo/hindi-character-recognition)
    - Unzip it. You need to split the data into 4 different directories, since we are training for Hindi digits & letters separately.
    ![image](https://user-images.githubusercontent.com/54038552/209815855-cd629bdd-5a9a-474e-8ad6-1d4df1954fdc.png)
    
    How to run ?
    - You can create your custom model in the `model.py` file or can go with the `HNet` already present. For custom models created, you need to import them to `train.py`, for them to to use. Remember we are training different models for Hindi Digit & Characters.
    - Now to train the model with default params do, `python train.py`. You can also specify epochs and lr. Most important, is the `model_type`
    - To train do, `python train.py --epochs <num-epochs> --lr <learning-rate> --model_type <type-of-model>`
    
