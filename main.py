import os
import shutil
import sys
from tqdm import tqdm
import argparse
import time
import math
import json
from omegaconf import OmegaConf
from utils.logger import MyLogger, reproduc
from utils.OctTree import OctTreeMLP
from utils.tool import save_img, get_folder_size
from utils.metrics import eval_performance
from utils.ModelSave import save_tree_models
from utils.VTK import save_vtk, parse_and_read_vtk, get_new_attribute_name_list, get_vtk_size_bytes

class CompressFramework:
    def __init__(self, opt, Log) -> None:
        self.opt = opt
        self.Log = Log
        self.compress_opt = opt.CompressFramwork
        self.data_path = self.compress_opt.Path
        self.points_array, self.points_value_array = parse_and_read_vtk(self.compress_opt)
        # self.points_array, self.points_value_array = parse_vtk(self.compress_opt)

    def compress(self):
        time_start = time.time()
        time_eval = 0
        tree_mlp = OctTreeMLP(self.compress_opt, self.points_array, self.points_value_array)
        f_structure = open(os.path.join(self.Log.info_dir, 'structure.txt'), 'w+')
        for key in tree_mlp.net_structure:
            f_structure.write('*'*12+key+'*'*12+'\n')
            f_structure.write(str(tree_mlp.net_structure[key])+'\n')
        f_structure.close()
        self.Log.log_metrics({'ratio_set': self.compress_opt.Ratio}, 0)
        self.Log.log_metrics({'ratio_theory': tree_mlp.ratio}, 0)
        sampler = tree_mlp.sampler
        optimizer = tree_mlp.optimizer
        lr_scheduler = tree_mlp.lr_scheduler

        new_attribute_name_list = get_new_attribute_name_list()
        metrics = {}
        for name in new_attribute_name_list:
            metrics[f'{name}_psnr_best'] = 0
            metrics[f'{name}_psnr_epoch'] = 0

        pbar = tqdm(sampler, desc='Training', leave=True, file=sys.stdout)

        for step, batch_size in enumerate(pbar):
            optimizer.zero_grad()
            loss = tree_mlp.cal_loss(batch_size)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            pbar.set_postfix_str("loss={:.8f}".format(loss.item()))
            pbar.update(1)
            if sampler.judge_eval(self.compress_opt.Eval.epochs):
                time_eval_start = time.time()
                # predict 相当于解压缩，遍历输入数据的每一个点得出预测的值 eg: 512*512*512
                predict_points, predict_points_value = tree_mlp.predict(device=self.compress_opt.Eval.device, batch_size=self.compress_opt.Eval.batch_size)
                metrics['decode_time'] = time.time()-time_eval_start

                # eval performance的时候的origin_data, predict_data都是没有经过归一化的原始数据，predict_data求的时候是用归一化的去求，但最后转回来了
                psnr = eval_performance(self.points_array, self.points_value_array, predict_points, predict_points_value)
                for i in range(len(psnr)):
                    if psnr[i] > metrics[f'{new_attribute_name_list[i]}_psnr_best']:
                        metrics[f'{new_attribute_name_list[i]}_psnr_best'] = psnr[i]
                        metrics[f'{new_attribute_name_list[i]}_psnr_epoch'] = sampler.epochs_count
                        if self.compress_opt.Network.is_residual == True:
                            pass   # TODO: 未处理保存ResSine网络参数的代码
                        else:
                            save_tree_models(tree_mlp=tree_mlp, model_dir=os.path.join(self.Log.compressed_dir, f'models_{new_attribute_name_list[i]}_psnr_best'))
                        if step > sampler.length * 0.9:
                            save_vtk(self.compress_opt.Path, os.path.join(self.Log.decompressed_dir, f'decompressed_{new_attribute_name_list[i]}_psnr_best.vtk'), predict_points_value)

                    self.Log.log_metrics({f'{new_attribute_name_list[i]}_psnr': psnr[i]}, sampler.epochs_count)  # TODO
                    time_eval += (time.time() - time_eval_start)
        model_dir = os.path.join(self.Log.compressed_dir, 'models')
        if self.compress_opt.Network.is_residual == True:
            pass  # TODO: 未处理保存ResSine网络参数的代码
        else:
            save_tree_models(tree_mlp=tree_mlp, model_dir=model_dir)
        save_vtk(self.compress_opt.Path, os.path.join(self.Log.decompressed_dir, 'decompressed_psnr_final.vtk'),
                 predict_points_value)

        if self.compress_opt.Network.is_residual == True:
            pass  # TODO: 未处理保存ResSine网络参数的代码
        else:
            ratio_actual = get_vtk_size_bytes()/get_folder_size(model_dir)
            self.Log.log_metrics({'ratio_actual': ratio_actual}, 0)
            metrics['ratio_set'], metrics['ratio_theory'], metrics['ratio_actual'] = self.compress_opt.Ratio, tree_mlp.ratio, ratio_actual

        compress_time = int(time.time()-time_start-time_eval)
        print('Compression time: {}s={:.2f}min={:.2f}h'.format(compress_time, compress_time/60, compress_time/3600))
        metrics['time'] = compress_time
        with open(os.path.join(self.Log.info_dir, 'metrics.json'), 'w+') as f_metrics:
            json.dump(metrics, f_metrics)
        f_metrics.close()
        self.Log.log_metrics({'time': compress_time}, 0)
        self.Log.close()


def main():
    opt = OmegaConf.load(args.p)
    Log = MyLogger(**opt['Log'])
    shutil.copy(args.p, Log.script_dir)
    shutil.copy(__file__, Log.script_dir)
    reproduc(opt["Reproduc"])

    compressor = CompressFramework(opt, Log)
    compressor.compress()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='single task for compression')
    parser.add_argument('-p', type=str, default='opt/SingleTask/default.yaml', help='config file path')
    parser.add_argument('-g', help='availabel gpu list', default='0,1,2,3',
                        type=lambda s: [int(item) for item in s.split(',')])
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.g])
    main()
