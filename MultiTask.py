import os
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
from typing import List
from omegaconf import OmegaConf
import time
import argparse
import shutil
import subprocess
from utils.misc import omegaconf2dotlist, CONCAT, omegaconf2dict
from utils.Multiprocess import Task, Queue
from pathlib import Path

timestamp = time.strftime("_%Y_%m%d_%H%M%S")

def gen_task_list(yaml_path:str, main_script_path:str):
    task_list = []
    opt = OmegaConf.load(yaml_path)
    # create a temp dir to save the opt in .yaml file
    yaml_dir = os.path.dirname(yaml_path)
    temp_dir = os.path.join(yaml_dir, 'temp_opt' + timestamp).replace('\\','/')
    os.makedirs(temp_dir, exist_ok=True)
    temp_stdout = os.path.join(yaml_dir, 'temp_stdout' + timestamp).replace('\\','/')
    os.makedirs(temp_stdout, exist_ok=True)
    static = omegaconf2dotlist(opt.Static)
    dynamic_list = CONCAT(opt.Dynamic)
    dotlist_list = [static+dynamic for dynamic in dynamic_list]
    # task
    for task_idx,dotlist in enumerate(dotlist_list):
        task_opt_yaml = OmegaConf.from_dotlist(dotlist)
        task_opt_path = opj(temp_dir, str(task_idx) + '.yaml').replace('\\','/')
        OmegaConf.save(task_opt_yaml, task_opt_path)
        # devide = False if task_opt_yaml.devide.type == 'None' else True
        devide = False
        command = "python {} -p {}".format(main_script_path, task_opt_path)
        if args.debug:
            stdout = ''
        else:
            stdout = opj(temp_stdout, str(task_idx) + '.log')
        task_list.append(Task(command=command, stdout=stdout, devide=devide))
    return task_list, temp_dir, temp_stdout

def main():
    task_list, temp_dir, temp_stdout = gen_task_list(args.p, args.stp)
    try:
        if os.name == "nt":   
            for task in task_list:
                command = f'{task.command} -g 0'
                print(f'Command: {command}')
                process = subprocess.run(command, shell=True, check=True)
        else:
            queue = Queue(task_list, args.g)
            queue.start(args.t, remind=True, batch_compress=True)
        shutil.rmtree(temp_dir)
        shutil.rmtree(temp_stdout)
    except:
        shutil.rmtree(temp_dir)
        shutil.rmtree(temp_stdout)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Batch Compression')
    parser.add_argument('-p', type=str, default='opt/MultiTask/default.yaml', help='config file path')
    parser.add_argument('-stp',type=str,default=opj(opd(__file__),'main.py'),help='the singletask script path')
    parser.add_argument('-g', help='availabel gpu list',default='0,1,2,3', type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('-t',type=float,default=2,help='the time interval between each task-assigning loop. For batch compression')
    parser.add_argument('-debug',action='store_true',help='debug')
    args = parser.parse_args()
    main()
