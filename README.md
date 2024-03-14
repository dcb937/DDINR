# DDINR
 Implicit Neural Representation with Domain Decomposition for Unstructured Grid Scientific Data



# Installation

```bash
conda create -n DDINR python=3.9
conda activate DDINR
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```



```bash
pip install -r requirements.txt
```



# Usage

Before using this code, please follow these steps to manually create the necessary folders and prepare your data:

1. Create a `data` folder and an `outputs` folder.
2. Place your unstructured grid data that you need to compress into the `data` folder. Note that the data should be stored in VTK or VTU file format.
3. Modify the corresponding training configuration files in the opt directory to match your data and training requirements.


### SingleTask

```bash
python main.py -p opt/SingleTask/default.yaml -g 0 
```



### MultiTask

```bash
python MultiTask.py -p opt/MultiTask/default.yaml -g 0,1,2,3 -stp main.py -debug
```

The option `-g` depands on the number of GPUs on your device.



### Training Result

```bash
tensorboard --logdir=outputs/default_{time}
```



### Visualization

```bash
python UI.py
```

