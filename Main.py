import numpy as np
import math
from datetime import datetime
import os
import argparse
import scipy.io as scio
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from Net_Functions import *

parser = argparse.ArgumentParser(description='parameters to Network')
parser.add_argument('--N', type=int, default=100) # The number of RIS reflective elements
parser.add_argument('--M', type=int, default=8)  # The number of transmit antennas
parser.add_argument('--K', type=int, default=4)  # The number of users
parser.add_argument('--eb', type=int, default=10)  # Rician factor of channels
parser.add_argument('--num_iter', type=int, default=10)  # The number of iterations
parser.add_argument('--SNR', type=float, default=10)  # SNR
parser.add_argument('--lr', type=float, default=0.001)  # Learning rate
parser.add_argument('--Train_flag', type=int, default=1) # 1 for traning, 0 for testing
parser.add_argument('--Active_Init', type=str, default="None") # GNN or None
parser.add_argument('--Passive_Init', type=str, default="None") # PI or None
parser.add_argument('--Update_Type', type=str, default="PI") # PI
parser.add_argument('--Dis_Bits', type=int, default=0)  # The number of phase shifters bits, 0 for continuous phase shift
parser.add_argument('--Dis_eta', type=int, default=20)  # For approximate quantization function
parser.add_argument('--ImCSI_Flag', type=int, default=0)  # 1 for perfect CSI, 0 for imperfect CSI
parser.add_argument('--AllSNR_Flag', type=int, default=0)  # 0 for training at specific SNR, 1 for training at all SNR
args = parser.parse_args()

class para_list:
    def __init__(self):
        self.N = args.N
        self.N = args.N
        self.M = args.M
        self.K = args.K
        self.eb = args.eb
        
        self.num_iter = args.num_iter
        self.Active_Init = args.Active_Init
        self.Passive_Init = args.Passive_Init
        self.Update_Type = args.Update_Type
        self.Dis_Bits = args.Dis_Bits
        self.Dis_eta = args.Dis_eta
        self.ImCSI_Flag = args.ImCSI_Flag
        self.AllSNR_Flag = args.AllSNR_Flag
        
        self.SNR = args.SNR
        self.SNR_tag = str(int(self.SNR*10)).zfill(4)
        self.Pt = np.power(10., args.SNR / 10.)            

        self.nr_of_samples_per_batch = 50
        self.nr_of_epoch_training = 20
        self.fac_of_imperfectCSI = 20
        self.nr_of_batch_per_epoch = 20
        self.nr_of_batch_test = 20
        self.nr_of_epoch_start = 0       
        self.lr = args.lr
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
        self.Train_flag = args.Train_flag


para = para_list() 
ImCSI_List = {False:"PerCSI", True:"ImCSI"}
AllSNR_List = {False:para.SNR_tag, True:"All"}

type_tag = "_Iter" + str(para.num_iter) + "_M" + str(para.M) + "_N" + str(para.N) + "_K" + str(para.K) + "_Rician" + str(para.eb) + "_SNR" + AllSNR_List[para.AllSNR_Flag] + "_" + para.Active_Init + "_" + para.Passive_Init + "_" + str(para.Dis_Bits) + "Bits_" + ImCSI_List[para.ImCSI_Flag]
appendix_tag = "_Final"
model_dir = para.Update_Type + 'Net' + type_tag + appendix_tag
Init_model_dir = 'Init_Net' + type_tag + appendix_tag

output_file_name = "Log_%s.txt" % model_dir

# Network Structure        
class WMMSE_Net(torch.nn.Module):
    def __init__(self, para):
        super(WMMSE_Net, self).__init__()
        if para.ImCSI_Flag == True:
            self.B_step = torch.nn.Parameter(torch.Tensor([2]*para.num_iter).double())
            self.theta_step = torch.nn.Parameter(torch.Tensor([2]*para.num_iter).double())
        if para.Update_Type == "PI":
            self.gamma = torch.nn.Parameter(torch.Tensor([0.001]*para.num_iter).double())            
        if para.Passive_Init == "PI":
            self.gamma_Init = torch.nn.Parameter(torch.Tensor([0.001]).double())            
        if para.Active_Init == "GNN":
            self.w_in = nn.Parameter(init.xavier_normal_(torch.Tensor(4*para.M+6, 6*para.M)).double())
            self.w_aggregate = nn.Parameter(init.xavier_normal_(torch.Tensor(6*para.M, 3*para.M)).double())
            self.w_combine = nn.Parameter(init.xavier_normal_(torch.Tensor(9*para.M, 3*para.M)).double())
            self.w_out = nn.Parameter(init.xavier_normal_(torch.Tensor(3*para.M, 2)).double())
            self.b_in = nn.Parameter(torch.Tensor([0.01]*6*para.M).double())
            self.b_aggregate = nn.Parameter(torch.Tensor([0.01]*3*para.M).double())
            self.b_combine = nn.Parameter(torch.Tensor([0.01]*3*para.M).double())
            self.b_out = nn.Parameter(torch.Tensor([0.01]*2).double())    

    def forward(self, G_Los_torch, Hr_Los_torch, weight_torch, pd_torch, ps_torch, Train_flag=True):
        if para.AllSNR_Flag == True:
            Pt_torch = torch.rand(1, device=para.device) * 10
        else:
            Pt_torch = torch.Tensor([para.Pt]).double().to(para.device)
        if para.ImCSI_Flag == True:
            ImCSI_rho = torch.rand(1, device=para.device) * 0.5
            fac_ture_torch = torch.Tensor([torch.sqrt(1/(ImCSI_rho+1))]).to(para.device)
            fac_error_torch = torch.Tensor([torch.sqrt(0.5*ImCSI_rho/(ImCSI_rho+1))]).to(para.device)        
            
        # Generate input channel with LOS channel
        Hd_w = half_torch * torch.complex(torch.randn(para.nr_of_samples_per_batch, para.K, para.M, device = para.device), torch.randn(para.nr_of_samples_per_batch, para.K, para.M, device = para.device))
        G_sig = half_torch * torch.complex(torch.randn(para.nr_of_samples_per_batch, para.N, para.M, device = para.device), torch.randn(para.nr_of_samples_per_batch, para.N, para.M, device = para.device))
        Hr_sig = half_torch * torch.complex(torch.randn(para.nr_of_samples_per_batch, para.K, para.N, device = para.device), torch.randn(para.nr_of_samples_per_batch, para.K, para.N, device = para.device))
        Hd = pd_torch * Hd_w
        G = eb1_torch*G_Los_torch + eb2_torch*G_sig
        Hr = ps_torch*(eb1_torch*Hr_Los_torch + eb2_torch*Hr_sig)        
        Hr_shape = torch.reshape(bdiag(torch.reshape(Hr, [-1, para.N, 1])), [-1, para.K, para.N, para.N])
        Hr_G = torch.matmul(Hr_shape, torch.unsqueeze(G, axis=1))
        
        phi = torch.rand(para.nr_of_samples_per_batch, para.N, 1, device = para.device, dtype=torch.complex128)*2*math.pi  
        theta = torch.exp(1j*phi)
        
        theta_t = torch.tile(torch.unsqueeze(theta, axis=1).transpose(-2,-1).conj(), [1, para.K, 1, 1])
        H = Hd + torch.squeeze(torch.matmul(theta_t, Hr_G))
        loss_list = []
        
        # Traditional initialize, WMMSE-PINet
        W = torch.matmul(H.transpose(1,2).conj(), torch.linalg.inv(torch.matmul(H, H.transpose(1,2).conj())))
        norm_W_sum = torch.linalg.norm(W, ord='fro', axis=[-2, -1])
        norm_W_sum = torch.tile(torch.reshape(norm_W_sum, [-1, 1, 1]), [1, para.M, para.K])
        W = torch.div(W, norm_W_sum) * torch.sqrt(Pt_torch)
        mu, lambda_torch = K_update_mu_lambda(para, H, W, Pt_torch)
        W = K_update_W(para, H, W, weight_torch, mu, lambda_torch, Pt_torch)
        
        # Advanced initialize with GNN, WMMSE-PINet+
        if para.Passive_Init == "PI":
            theta = K_init_theta_PI(para, W, theta, Hd, Hr, G, mu, lambda_torch, weight_torch, self.gamma_Init)        
        if para.Dis_Bits > 0:
            theta = Discrete_theta(para, theta, Train_flag)
        theta_t = torch.tile(torch.unsqueeze(theta, axis=1).transpose(-2,-1).conj(), [1, para.K, 1, 1])
        H_RIS = torch.squeeze(torch.matmul(theta_t, Hr_G))
        H = Hd + H_RIS
        if para.Active_Init == "GNN":
            W = K_init_W_GNN(para, Hd, H_RIS, mu, lambda_torch, weight_torch, Pt_torch, self)
        
        # Initialization for WMMSE-PINet-ImCSI
        if para.ImCSI_Flag == True:
            loss_list.append(Get_SINR_imperfectCSI(para, Hd_w, G_Los_torch, G_sig, Hr_Los_torch, Hr_sig, eb1_torch, eb2_torch, pd_torch, ps_torch, fac_ture_torch, fac_error_torch, W, theta, weight_torch, Pt_torch))
            B_old = torch.zeros(para.nr_of_samples_per_batch, para.N+1, para.N+1, device = para.device)
        else:
            loss_list.append(Get_SINR(para, H, W, weight_torch))
            
        for ii in range(para.num_iter):     
            # Sample a new group of channels
            if para.ImCSI_Flag == True:
                B_step_iter = torch.nn.functional.sigmoid(self.B_step[ii])
                theta_step_iter = torch.nn.functional.sigmoid(self.theta_step[ii])
                Hd_w_t = fac_ture_torch * Hd_w + fac_error_torch * torch.complex(torch.randn(para.nr_of_samples_per_batch, para.K, para.M, device = para.device), torch.randn(para.nr_of_samples_per_batch, para.K, para.M, device = para.device))
                G_sig_t = fac_ture_torch * G_sig + fac_error_torch * torch.complex(torch.randn(para.nr_of_samples_per_batch, para.N, para.M, device = para.device), torch.randn(para.nr_of_samples_per_batch, para.N, para.M, device = para.device))
                Hr_sig_t = fac_ture_torch * Hr_sig + fac_error_torch * torch.complex(torch.randn(para.nr_of_samples_per_batch, para.K, para.N, device = para.device), torch.randn(para.nr_of_samples_per_batch, para.K, para.N, device = para.device))
                Hd_t = pd_torch * Hd_w_t
                G_t = eb1_torch*G_Los_torch + eb2_torch*G_sig_t
                Hr_t = ps_torch*(eb1_torch*Hr_Los_torch + eb2_torch*Hr_sig_t)

                Hr_shape = torch.reshape(bdiag(torch.reshape(Hr_t, [-1, para.N, 1])), [-1, para.K, para.N, para.N])
                Hr_G = torch.matmul(Hr_shape, torch.unsqueeze(G_t, axis=1))
                theta_t = torch.tile(torch.unsqueeze(theta, axis=1).transpose(-2,-1).conj(), [1, para.K, 1, 1])
                H = Hd_t + torch.squeeze(torch.matmul(theta_t, Hr_G))
            
            mu, lambda_torch = K_update_mu_lambda(para, H, W, Pt_torch)
            if para.ImCSI_Flag == True:
                W = K_update_W(para, H, W, weight_torch, mu, lambda_torch, Pt_torch)
                
            if para.Update_Type == "PI":                               
                if para.ImCSI_Flag == True:
                    theta_new, B_old = K_update_theta_PI_ImperfectCSI(para, W, theta, Hd_t, Hr_t, G_t, mu, lambda_torch, weight_torch, self.gamma[ii], B_old, B_step_iter)
                    theta = (1-theta_step_iter)*theta + theta_step_iter * theta_new
                else:
                    theta = K_update_theta_PI(para, W, theta, Hd, Hr, G, mu, lambda_torch, weight_torch, self.gamma[ii])
            elif para.Update_Type == "Init":
                theta = K_update_theta_PI_NoAI(para, W, theta, Hd, Hr, G, mu, lambda_torch, weight_torch)
                
            if para.ImCSI_Flag == True: 
                if para.Dis_Bits > 0:
                    theta_Dis = Discrete_theta(para, theta, Train_flag)
                    loss_list.append(Get_SINR_imperfectCSI(para, Hd_w, G_Los_torch, G_sig, Hr_Los_torch, Hr_sig, eb1_torch, eb2_torch, pd_torch, ps_torch, fac_ture_torch, fac_error_torch, W, theta_Dis, weight_torch, Pt_torch))
            else: 
                if para.Dis_Bits > 0:
                    theta_Dis = Discrete_theta(para, theta, Train_flag)
                    theta_t = torch.tile(torch.unsqueeze(theta_Dis, axis=1).transpose(-2,-1).conj(), [1, para.K, 1, 1])                
                    H_Dis = Hd + torch.squeeze(torch.matmul(theta_t, Hr_G))
                    W_Dis = K_update_W(para, H_Dis, W, weight_torch, mu, lambda_torch, Pt_torch)
                    loss_list.append(Get_SINR(para, H_Dis, W_Dis, weight_torch))                    
                theta_t = torch.tile(torch.unsqueeze(theta, axis=1).transpose(-2,-1).conj(), [1, para.K, 1, 1])
                H = Hd + torch.squeeze(torch.matmul(theta_t, Hr_G))
                W = K_update_W(para, H, W, weight_torch, mu, lambda_torch, Pt_torch)
                
            if para.Dis_Bits == 0:
                if para.ImCSI_Flag == True:
                    loss_list.append(Get_SINR_imperfectCSI(para, Hd_w, G_Los_torch, G_sig, Hr_Los_torch, Hr_sig, eb1_torch, eb2_torch, pd_torch, ps_torch, fac_ture_torch, fac_error_torch, W, theta, weight_torch, Pt_torch))
                else:                    
                    loss_list.append(Get_SINR(para, H, W, weight_torch))
                    
        if Train_flag == True:
            loss = -torch.mean(torch.stack(loss_list))  
        else:
            loss = torch.mean(torch.stack(loss_list), axis = 1)
        return loss

model = WMMSE_Net(para)
model = model.to(para.device)

if para.ImCSI_Flag == True:
    B_step_params = id(model.B_step)
    theta_step_params = id(model.theta_step)
    other_params = filter(lambda p: (id(p) != B_step_params) and (id(p) != theta_step_params), model.parameters())
    params = [
        {"params": other_params, "lr": para.lr},
        {"params": model.B_step, "lr": para.lr * 100},
        {"params": model.theta_step, "lr": para.lr * 100},
    ]
    optimizer = torch.optim.Adam(params)
else:    
    optimizer = torch.optim.Adam(model.parameters(), lr=para.lr)
    
StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

if __name__ == "__main__":
    G_Los, Hr_Los, weight, pd, ps, eb1, eb2 = load_dataset(para)
    train_data = Dataset(G_Los, Hr_Los, weight, pd, ps)
    
    eb1_torch = torch.from_numpy(eb1).to(para.device)
    eb2_torch = torch.from_numpy(eb2).to(para.device)
    half_torch = torch.sqrt(torch.Tensor([0.5])).to(para.device)

    startTime = datetime.now()
    print("Finish Loading Dataset at " + str(startTime))
    if para.Train_flag == 0:
        print("Testing:")
        test_result = Test_WSR(para, model, train_data)
        print("Test WSR Before Loading Model = ", test_result)
        
        load_dir = "./Model/" + model_dir + "_Final.pkl"        
        model.load_state_dict(torch.load(load_dir), strict=False)  
        print("Restore model from ", load_dir)
        
        test_result = Test_WSR(para, model, train_data)
        print("Test WSR After Loading Model = ", test_result)
    else:
        train_iter = DataLoader(dataset=train_data, batch_size=para.nr_of_samples_per_batch, shuffle=True)
        ##########################
        #### Stage Init Begin ####
        ##########################
        if para.Active_Init == "GNN" and para.Passive_Init == "PI":
            if para.ImCSI_Flag == 1:
                para.num_iter = 0
            else:
                para.num_iter = 2
            para.Update_Type = "Init"

            print("Test Before Init Training:")
            test_result = Test_WSR(para, model, train_data)
            print("Test WSR Before Init Training = ", test_result)

            training_loss = []    
            Train_startTime = datetime.now()
            print("Start Init Training at " + str(Train_startTime))
            for epoch_i in range(para.nr_of_epoch_start + 1, para.nr_of_epoch_training + 1):
                loss_epoch = []
                epoch_startTime = datetime.now()
                epoch_nan = 0            
                for data in train_iter:
                    G_Los, Hr_Los, weight, pd, ps = map(lambda x: x.to(para.device), data)
                    loss = model(G_Los, Hr_Los, weight, pd, ps)
                    optimizer.zero_grad()            
                    loss.backward()        
                    optimizer.step()
                    loss_epoch.append(loss)  
                StepLR.step()
                training_loss.append(torch.mean(torch.Tensor(loss_epoch)).numpy())

                nowTime = datetime.now()
                Train_diffTime = nowTime - Train_startTime
                epoch_diffTime = nowTime - epoch_startTime
                restTime = Train_diffTime / (epoch_i - para.nr_of_epoch_start) * (para.nr_of_epoch_training - epoch_i)
                endTime = nowTime + restTime
                epoch_time = str(epoch_diffTime.seconds) + '.' + str(epoch_diffTime.microseconds)
                output_data = "Stage Init, Epoch:[%d/%d], WMMSE_train: %.2f, cost_time: %.2f, may end at: " % (epoch_i, para.nr_of_epoch_training, float(training_loss[-1]), float(epoch_time)) + str(endTime)
                print(output_data)

                test_result = Test_WSR(para, model, train_data)

                if epoch_i <= 1:
                    save_dir = "./Model/" + Init_model_dir + "_epoch" + str(epoch_i) + ".pkl"
                    torch.save(model.state_dict(), save_dir)
                if epoch_i % 5 == 0:
                    save_dir = "./Model/" + Init_model_dir + "_epoch" + str(epoch_i) + ".pkl"
                    torch.save(model.state_dict(), save_dir)
                    print("Test WSR = ", test_result)
                    for var in model.state_dict().keys():# 输出读入的参数并进行冻结
                        if "gamma" in var or "rho" in var:
                            data = eval("model." + var).data
                            print(var + " = : " + str(data))       

            save_dir = "./Model/" + Init_model_dir + "_Final.pkl"
            torch.save(model.state_dict(), save_dir)
            print("Save Init Final Model at: " + save_dir)
            print("Training took:" + str(datetime.now() - Train_startTime))
        
        ##########################
        #### Stage Init Ends #####
        ##########################

        
        ###########################
        #### Stage Main Begins ####
        ###########################
        
        para.num_iter = args.num_iter
        para.Update_Type = args.Update_Type
        para.Dis_eta = args.Dis_eta

        for var in model.state_dict().keys():
            if "step" in var:
                data = eval("model." + var).data
                print(var + "_eff = : " + str(torch.nn.functional.sigmoid(data)))  
            if ("gamma" in var or "rho" in var or "step" in var) and ("Init" not in var):                
                data = eval("model." + var).data
                print(var + " = : " + str(data))
                eval("model." + var).requires_grad = True
            else:                
                data = eval("model." + var).data
                print(var + " = : " + str(data))
                eval("model." + var).requires_grad = False
        
        print("Test Before Main Training:")
        test_result = Test_WSR(para, model, train_data)
        print("Test WSR Before Main Training = ", test_result)

        training_loss = []    
        Train_startTime = datetime.now()
        print("Start Main Training at " + str(Train_startTime))
        for epoch_i in range(para.nr_of_epoch_start + 1, para.nr_of_epoch_training + 1):
            loss_epoch = []
            epoch_startTime = datetime.now()
            epoch_nan = 0
            for data in train_iter:
                G_Los, Hr_Los, weight, pd, ps = map(lambda x: x.to(para.device), data)
                loss = model(G_Los, Hr_Los, weight, pd, ps)
                optimizer.zero_grad()            
                loss.backward()        
                optimizer.step()
                loss_epoch.append(loss)  
            StepLR.step()
            training_loss.append(torch.mean(torch.Tensor(loss_epoch)).numpy())

            nowTime = datetime.now()
            Train_diffTime = nowTime - Train_startTime
            epoch_diffTime = nowTime - epoch_startTime
            restTime = Train_diffTime / (epoch_i - para.nr_of_epoch_start) * (para.nr_of_epoch_training - epoch_i)
            endTime = nowTime + restTime
            epoch_time = str(epoch_diffTime.seconds) + '.' + str(epoch_diffTime.microseconds)
            output_data = "Stage Main, Epoch:[%d/%d], WMMSE_train: %.2f, cost_time: %.2f, may end at: " % (epoch_i, para.nr_of_epoch_training, float(training_loss[-1]), float(epoch_time)) + str(endTime)
            print(output_data)
            test_result = Test_WSR(para, model, train_data)

            if epoch_i <= 1:
                save_dir = "./Model/" + model_dir + "_epoch" + str(epoch_i) + ".pkl"
                torch.save(model.state_dict(), save_dir)
            if epoch_i % 5 == 0:
                save_dir = "./Model/" + model_dir + "_epoch" + str(epoch_i) + ".pkl"
                torch.save(model.state_dict(), save_dir)
                print("Test WSR = ", test_result)
                for var in model.state_dict().keys():
                    if "gamma" in var or "rho" in var or "step" in var:
                        data = eval("model." + var).data
                        print(var + " = : " + str(data))  
                    if "step" in var:
                        data = eval("model." + var).data
                        print(var + "_eff = : " + str(torch.nn.functional.sigmoid(data)))      
        
        save_dir = "./Model/" + model_dir + "_Final.pkl"
        torch.save(model.state_dict(), save_dir)
        print("Save Final Model at: " + save_dir)
        print("Training took:" + str(datetime.now() - Train_startTime))

endTime = datetime.now()
print("End running at " + str(endTime))
