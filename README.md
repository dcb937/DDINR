# DDINR
 Implicit Neural Representation with Domain Decomposition for Unstructed Grid Data



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

