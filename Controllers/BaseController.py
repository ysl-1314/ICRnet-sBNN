import copy
import pprint
import time

import numpy as np
import optuna
import torch
from accelerate.test_utils.scripts.test_distributed_data_loop import test_data_loader
from tqdm import tqdm

from Metrics import MetricTest
from Modules.Interpolation import SpatialTransformer
from Modules.Loss import DiceCoefficient, DiceCoefficientAll
from Networks import BaseRegistraionNetwork
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Utils import EarlyStopping, ParamsAll

from rich.progress import Progress, BarColumn, TimeRemainingColumn

progress = Progress(
    "[progress.description]{task.description}",
    BarColumn(),
    "[progress.percentage]{task.percentage:3.2f}%",
    "{task.completed:5.0f}",
    "best: {task.fields[best]:.5f}",
    "best_epoch: {task.fields[best_epoch]:5.0f}",
    TimeRemainingColumn(),
)


class BaseController:
    def __init__(self, net: BaseRegistraionNetwork):
        self.net = net
        self.swa_start = 1
        self.update_freq = 1
        self.K = 10
        self.n_samples=10

        self.toprate = 20

    def cuda(self):
        self.net.cuda()

    def swag_low_rank(self, K,e=1,w_swa=None,w_swa_2=None,D=None):

        n_models = e //self.update_freq

        w = self.net.get_weight_vector()
        w_2 = torch.pow(w, 2)
        w_swa = (w_swa * n_models + w) / (n_models + 1)
        w_swa_2 = (w_swa_2 * n_models + w_2) / (n_models + 1)

        col_idx = n_models % K
        if n_models % K == 0 and n_models > 0:
            D[:, :-1] = D[:, 1:]
            D[:, -1] = 0
        D[:, col_idx] = w - w_swa

        var = w_swa_2 - torch.pow(w_swa, 2)

        return w_swa, var, D

    def sample_swag(self,w_swa, sqrt_var_vec, D):
        z_1 = torch.randn(sqrt_var_vec.size(0), dtype=torch.double, device=sqrt_var_vec.device)
        noise_diag = (1 / np.sqrt(2)) * sqrt_var_vec * z_1

        K = D.size(dim=1)
        z_2 = torch.randn(K, dtype=torch.double, device=D.device)
        noise_low_rank = (1 / np.sqrt(K - 1)) * D @ z_2

        posterior_noise = noise_diag + noise_low_rank

        return w_swa + posterior_noise

    def bma(self, w_swa, var_vec, D, n_samples=100):
        model = copy.deepcopy(self.net)
        sample_params = []
        sqrt_var_vec = var_vec.sqrt()
        sample_params_mean = torch.zeros_like(w_swa, dtype=torch.double, device=w_swa.device, requires_grad=False)

        for i in tqdm(range(n_samples)):
            sampled_params = self.sample_swag(w_swa, sqrt_var_vec, D)
            sample_params.append(sampled_params)

            sample_params_mean += sampled_params

        sample_params_mean /= n_samples
        return model

    def train(self,
              train_dataloader: DataLoader,
              validation_dataloader: DataLoader,
              save_checkpoint,
              earlystop: EarlyStopping,
              logger: SummaryWriter,
              start_epoch=0,
              max_epoch=1000,
              lr=1e-4,
              v_step=50,
              verbose=1):

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        optimizer_unet = torch.optim.Adam(self.net.unet.parameters(), lr=lr)
        earlystop.on_train_begin()

        end = time.perf_counter()
        if verbose == 0:
            task = progress.add_task('Training...',
                                     total=max_epoch * 2,
                                     best=0,
                                     best_epoch=0)
            progress.start()
        # setting initial moments
        w = self.net.get_weight_vector()
        w_2 = torch.pow(w, 2)
        w_swa = w.clone().to(torch.double)
        w_swa_2 = w_2.clone().to(torch.double)
        D = torch.zeros((w_swa.size(dim=0),self.K), dtype=torch.double, device=torch.device('cuda'))

        flag=0
        for e in range(start_epoch, max_epoch * 2):
            start = end
            #swa 开始
            train_loss_dict = self.trainIter(train_dataloader, optimizer, optimizer_unet)
            if e>self.swa_start and e % self.update_freq == 0:
                flag=1
                w_swa, var, D = self.swag_low_rank(self.K,e,w_swa,w_swa_2,D)

            # validation
            if flag:
                model_save = self.bma(w_swa,var, D, n_samples=self.n_samples)
                validation_dice = self.validationIter(validation_dataloader,model_save)

            # save checkpoint
            if save_checkpoint and flag:
                save_checkpoint(model_save, e + 1)

            # console
            train_loss_mean_str = ''
            for key in train_loss_dict:
                train_loss_mean_str += '%s : %f, ' % (key,
                                                      train_loss_dict[key])
            end = time.perf_counter()

            if verbose and flag:
                print(e + 1, '%.2f' % (end - start), train_loss_mean_str,
                      validation_dice)

            # TODO:Logger for Tensorboard
            # TODO:Visualization for Tensorboard

            # early stop
            if flag:
                if earlystop.on_epoch_end(e + 1, validation_dice,
                                          model_save) and e >= max_epoch:
                    if verbose == 0:
                        progress.update(task,
                                        advance=1,
                                        best=earlystop.best,
                                        best_epoch=earlystop.best_epoch,
                                        refresh=True)
                        time.sleep(0.0001)
                        progress.stop_task(task)
                        progress.remove_task(task)
                    break
                if verbose == 0:
                    progress.update(task,
                                    advance=1,
                                    best=earlystop.best,
                                    best_epoch=earlystop.best_epoch,
                                    refresh=True)
                    time.sleep(0.0001)


        return earlystop.best

    # 训练
    def trainIter(self, dataloader: DataLoader,
                  optimizer: torch.optim.Optimizer,
                  optimizer_unet: torch.optim.Optimizer):
        train_loss_dict = {}

        self.net.train()
        for name, param in self.net.named_parameters():
            if "unet" in name:
                param.requires_grad = False
        for data in dataloader:
            src = data['src']['img'].cuda()
            tgt = data['tgt']['img'].cuda()
            optimizer.zero_grad()
            # forward to loss
            loss_dict = self.net.objective(src, tgt)
            loss = loss_dict['loss'].mean()
            # backward
            loss.backward()
            # update
            optimizer.step()
            for key in loss_dict:
                loss_mean = loss_dict[key].mean().item()
                if key not in train_loss_dict:
                    train_loss_dict[key] = [loss_mean]
                else:
                    train_loss_dict[key].append(loss_mean)


        self.net.eval()
        for name, param in self.net.named_parameters():
            if "unet" in name:
                param.requires_grad = True
        for data in dataloader:
            src = data['src']['img'].cuda()
            tgt = data['tgt']['img'].cuda()
            optimizer_unet.zero_grad()
            # forward to loss
            loss_dict = self.net.objective_unet(src, tgt)
            Id_loss = loss_dict['Id_loss'].mean()
            # backward
            Id_loss.backward()
            # update
            optimizer_unet.step()
            for key in loss_dict:
                loss_mean = loss_dict[key].mean().item()
                if key not in train_loss_dict:
                    train_loss_dict[key] = [loss_mean]
                else:
                    train_loss_dict[key].append(loss_mean)

        self.net.train()
        for name, param in self.net.named_parameters():
            if "unet" in name:
                param.requires_grad = False
        for data in dataloader:
            src = data['src']['img'].cuda()
            tgt = data['tgt']['img'].cuda()
            optimizer.zero_grad()
            # forward to loss
            loss_dict = self.net.objective_bidir(src, tgt)
            loss = loss_dict['bidir_loss'].mean()
            # backward
            loss.backward()
            # update
            optimizer.step()
            for key in loss_dict:
                loss_mean = loss_dict[key].mean().item()
                if key not in train_loss_dict:
                    train_loss_dict[key] = [loss_mean]
                else:
                    train_loss_dict[key].append(loss_mean)

        for key in train_loss_dict:
            train_loss_dict[key] = np.mean(train_loss_dict[key])

        return train_loss_dict

    # 验证
    def validationIter(self, dataloader: DataLoader,model):
        dice_list = []
        with torch.no_grad():
            # dice_estimator = DiceCoefficient()
            dice_estimator = DiceCoefficientAll()
            for data in dataloader:
                src = data['src'][0].cuda().float()
                tgt = data['tgt'][0].cuda().float()
                # regard all types of segmeant as one
                src_seg = data['src_seg'][0].cuda().float()
                tgt_seg = data['tgt_seg'][0].cuda().float()

                result = model.test(src, tgt)
                phi = result[0][-1]
                phi_reverse = result[2][-1]
                warped_src_seg = model.transformer(src_seg,
                                                      phi,
                                                      mode='nearest')
                warped_tgt_seg = model.transformer(tgt_seg,
                                                      phi_reverse,
                                                      mode='nearest')
                dice = dice_estimator(tgt_seg,
                                      warped_src_seg.int()).unsqueeze(0)
                dice_reverse = dice_estimator(src_seg,
                                              warped_tgt_seg.int()).unsqueeze(0)
                dice_list.append((dice + dice_reverse) / 2)
            # statistics
            dice_tensor = torch.cat(dice_list, 0)
            return dice_tensor.mean().item()

    # 测试
    def test(self,
             dataloader: DataLoader,
             logger: SummaryWriter,
             name: str = None,
             network: str = None,
             excel_save_path: str = None,
             verbose=2):
        metric_test = MetricTest()
        with torch.no_grad():
            num = 0
            for data in dataloader:
                src = data['src'][0].cuda().float()
                tgt = data['tgt'][0].cuda().float()
                case_no = data['case_no'].item()
                src_seg = data['src_seg'][0].cuda().float()
                tgt_seg = data['tgt_seg'][0].cuda().float()
                slc_idx = data['slice']
                resolution = data['resolution'].item()

                results = self.net.test(src, tgt)

                phi = results[0][-1]
                phi_reverse = results[2][-1]
                warped_src_seg = self.net.transformer(src_seg,
                                                      phi,
                                                      mode='nearest')
                warped_tgt_seg = self.net.transformer(tgt_seg,
                                                      phi_reverse,
                                                      mode='nearest')

                # logger.add_image('test/src', src[0], num+1)
                # logger.add_image('test/tgt', tgt[0], num+1)
                # logger.add_image('test/s_t', results_t[1][0], num+1)
                # logger.add_image('test/t_s', resultt_s[1][0], num+1)
                # logger.add_image('test/src_seg', src_seg[0] * 50, num+1)
                # logger.add_image('test/tgt_seg', tgt_seg[0] * 50, num+1)
                # logger.add_image('test/s_t_seg', warped_src_seg[0] * 50, num+1)
                # logger.add_image('test/t_s_seg', warped_tgt_seg[0] * 50, num+1)

                metric_test.testMetrics(src_seg.int(), warped_src_seg.int(),
                                        tgt_seg.int(), warped_tgt_seg.int(),
                                        resolution, case_no, slc_idx)
                metric_test.testFlow(phi, phi_reverse, case_no)
                num += 1

        mean = metric_test.mean()
        if verbose >= 2:
            metric_test.saveAsExcel(network, name, excel_save_path)
        if verbose >= 1:
            metric_test.output()
        return mean, metric_test.details

    # 计算逆一致性误差ICE
    def ice_diff(self,
                 dataloader: DataLoader):
        ice_list = []
        stn = SpatialTransformer([128, 128]).cuda()
        with torch.no_grad():
            for data in dataloader:
                src = data['src'][0].cuda().float()
                tgt = data['tgt'][0].cuda().float()
                # regard all types of segmeant as one

                results = self.net.test(src, tgt)
                phi = results[0]
                phi_reverse = results[1]
                ice = stn(phi, phi_reverse) + phi_reverse
                ice_list.append(torch.abs(ice))
            # statistics
            ice_tensor = torch.cat(ice_list, 0)
            print(torch.mean(ice_tensor))
            print(torch.std(ice_tensor))
            return ice_tensor.mean().item()

    def hyperOpt(self,
                 hyperparams,
                 load_checkpoint,
                 n_trials,
                 train_dataloader: DataLoader,
                 validation_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 earlystop: EarlyStopping,
                 logger: SummaryWriter,
                 max_epoch=500,
                 lr=1e-4):
        def objective(trial: optuna.Trial):
            hyperparams = trial.study.user_attrs['hyperparams']
            params_instance = ParamsAll(trial, hyperparams)
            print(params_instance)
            load_checkpoint(self.net, 0)
            self.net.setHyperparam(**params_instance)

            self.train(train_dataloader,
                       validation_dataloader,
                       None,
                       earlystop,
                       None,
                       0,
                       max_epoch,
                       lr,
                       v_step=0,
                       verbose=0)

            res, _ = self.test(test_dataloader, verbose=0)
            print(res)
            return 1 - res['mean']

        study = optuna.create_study()
        study.set_user_attr('hyperparams', hyperparams)
        study.optimize(objective, n_trials, n_jobs=1)
        print(study.best_params)
        return study.best_params

    def speedTest(self, dataloader: DataLoader, device_type='gpu'):
        case_time = []
        slice_time = []
        if device_type is 'cpu':
            self.net.cpu()
        with torch.no_grad():
            for data in dataloader:
                if device_type is 'gpu':
                    src = data['src'][0].cuda().float()
                    tgt = data['tgt'][0].cuda().float()
                else:
                    src = data['src'][0].cpu().float()
                    tgt = data['tgt'][0].cpu().float()
                torch.cuda.synchronize()
                start = time.time()
                result = self.net.test(src, tgt)
                torch.cuda.synchronize()
                end = time.time()
                case_time.append(end - start)

                torch.cuda.synchronize()
                for i in range(src.size()[0]):
                    start = time.time()
                    result = self.net.test(src[i:i + 1], tgt[i:i + 1])
                    torch.cuda.synchronize()
                    end = time.time()
                    slice_time.append(end - start)
        case_res = {'mean': np.mean(case_time), 'std': np.std(case_time)}
        slice_res = {'mean': np.mean(slice_time), 'std': np.std(slice_time)}
        print(device_type)
        print('case', '%.3f(%.3f)' % (case_res['mean'], case_res['std']))
        print('slice', '%.3f(%.3f)' % (slice_res['mean'], slice_res['std']))

    def estimate(self, case_data: torch.Tensor):
        with torch.no_grad():
            src = case_data['src'].cuda().float()
            tgt = case_data['tgt'].cuda().float()
            src_seg = case_data['src_seg'].cuda().float()
            tgt_seg = case_data['tgt_seg'].cuda().float()
            slc_idx = case_data['slice']
            results = self.net.test(src, tgt)
            phi = results[0]
            phi_reverse = results[1]
            warped_src = self.net.transformer(src, phi)
            warped_src_seg = self.net.transformer(src_seg, phi, mode='nearest')
            warped_tgt = self.net.transformer(tgt, phi_reverse)
            warped_tgt_seg = self.net.transformer(tgt_seg, phi_reverse, mode='nearest')

            res = {
                'src': src.cpu().numpy()[:, 0, :, :],
                'tgt': tgt.cpu().numpy()[:, 0, :, :],
                'src_seg': src_seg.cpu().numpy()[:, 0, :, :],
                'tgt_seg': tgt_seg.cpu().numpy()[:, 0, :, :],
                'phi': phi.cpu().numpy(),
                'phi_reverse': phi_reverse.cpu().numpy(),
                'warped_src': warped_src.cpu().numpy()[:, 0, :, :],
                'warped_src_seg': warped_src_seg.cpu().numpy()[:, 0, :, :],
                'warped_tgt': warped_tgt.cpu().numpy()[:, 0, :, :],
                'warped_tgt_seg': warped_tgt_seg.cpu().numpy()[:, 0, :, :],
                'slc_idx': slc_idx
            }
            return res
