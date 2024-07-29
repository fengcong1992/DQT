This folder contains all the code and instructions about load trained models and utilzie the model to forecast.<br>
**Put model_load_example.py** in the same **directory** as **main.py**.<br>
Since everything related to the model is **pre-defined**, the best way to reload the models is to utilize existing functions.<br>
Saving the entire model requires to exactly recreate the same file structure (including the source code of the model) on the target system, which is quite brittle.<br>
So during the training process only the parameters (state_dict) are stored. <br>
![Example Image](./model_load.png)
