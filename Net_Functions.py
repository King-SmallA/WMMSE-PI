from torch.utils.data import Dataset, DataLoader

import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy.io as scio

    
def load_dataset(para):
    dataset_dir = "Channel" + "_M" + str(para.M) + "_N" + str(para.N) + "_K" + str(para.K) + "_Rician" + str(para.eb) + "_0725.mat"
    print("Load dataset from \'./Dataset/", dataset_dir)
    mat = scio.loadmat("./Dataset/" + dataset_dir)
    G_Los = mat['G_Los']  # array
    G_Los = np.array(G_Los)
    Hr_Los = mat['Hr_Los']  # array
    Hr_Los = np.array(Hr_Los)
    weight = mat['weight']  # array
    weight = np.array(weight)
    pd = mat['pd']  # array
    pd = np.array(pd)
    ps = mat['ps']  # array
    ps = np.array(ps)
    eb1 = mat['eb1']  # array
    eb1 = np.array(eb1)
    eb2 = mat['eb2']  # array
    eb2 = np.array(eb2)
    print("G_Los.shape = " + str(G_Los.shape))
    print("Hr_Los.shape = " + str(Hr_Los.shape))
    print("weight.shape = " + str(weight.shape))
    print("pd.shape = " + str(pd.shape))
    print("ps.shape = " + str(ps.shape))
    print("eb1.shape = " + str(eb1.shape))
    print("eb2.shape = " + str(eb2.shape))
    return G_Los, Hr_Los, weight, pd, ps, eb1, eb2


class Dataset(Dataset):
    def __init__(self, G_Los, Hr_Los, weight, pd, ps):
        self.G_Los = G_Los
        self.Hr_Los = Hr_Los
        self.weight = weight
        self.pd = pd
        self.ps = ps
    
    def __getitem__(self, item):
        return self.G_Los[item], self.Hr_Los[item], self.weight[item], self.pd[item], self.ps[item]
        
    def __len__(self):
        return np.shape(self.G_Los)[0]
    
    
def bdiag(X):
    # batch-trace: [B, N, 1] -> [B, N, N]
    B = X.size(0)
    n = X.size(1)
    X = X.view(-1, n, 1)
    result = torch.zeros(B,n,n, dtype=torch.complex128, device=X.device)
    for i in range(B):
        result[i,:,:] = torch.diag(X[i, :, 0])
    return result 

    
def bDiag(X):
    # batch-trace: [B, N, N] -> [B, N, 1]
    B = X.size(0)
    n = X.size(1)
    result = torch.zeros(B,n, dtype=torch.complex128, device=X.device)
    for i in range(B):
        result[i,:] = torch.diag(X[i, :, :])
    return result
    
    
def Discrete_theta(para, theta, Train_flag):
    phase_num = 2**para.Dis_Bits
    q_list = torch.linspace(-math.pi, math.pi, phase_num + 1, device=para.device)[:-1]
    theta = torch.angle(theta)
    nr_of_samples_in = theta.size()[0]
    dif_theta = torch.tile(theta, [1, 1, phase_num]) - torch.tile(torch.reshape(q_list, [1, 1, -1]), [nr_of_samples_in, para.N, 1])
    if Train_flag == True:
        dif_tanh = F.tanh(dif_theta * para.Dis_eta)
    else: 
        dif_tanh = torch.sign(dif_theta * para.Dis_eta)
    dis_phase = math.pi / phase_num * torch.sum(dif_tanh, axis=-1) - math.pi / phase_num
    dis_theta = torch.unsqueeze(torch.exp(1j*dis_phase), -1)
    return dis_theta


def Test_WSR(para, model, train_data):
    test_iter = DataLoader(dataset=train_data, batch_size=para.nr_of_samples_per_batch, shuffle=False)
    loss_test_total = 0
    num_test_total = 0
    for data in test_iter:
        G_Los, Hr_Los, weight, pd, ps = map(lambda x: x.to(para.device), data)
        loss = model(G_Los, Hr_Los, weight, pd, ps, False)
        loss_test_total = loss_test_total + loss.cpu().detach().numpy()
        num_test_total = num_test_total + 1
        if num_test_total >= para.nr_of_batch_test:
            break
    return loss_test_total/num_test_total

    
def Get_SINR(para, H, W, weight):
    He = torch.square(torch.abs(torch.matmul(H, W)))    
    Diag_He = bDiag(He)
    sum_temp = torch.div(Diag_He, torch.sum(He, axis=-1) - Diag_He + torch.ones_like(Diag_He))
    f0 = torch.sum(torch.multiply(weight, torch.log(torch.abs(torch.ones_like(Diag_He) + sum_temp))), axis=1)
    return f0  

    
def Get_SINR_imperfectCSI(para, Hd_w, G_Los, G_sig, Hr_Los, Hr_sig, eb1_torch, eb2_torch, pd_torch, ps_torch, fac_ture_torch, fac_error_torch, W, theta, weight_torch, Pt_torch):
    M = para.M
    N = para.N
    K = para.K
    
    Hd_w_expand = torch.tile(Hd_w, [para.fac_of_imperfectCSI, 1, 1])
    G_Los_expand_torch = torch.tile(G_Los, [para.fac_of_imperfectCSI, 1, 1])
    G_sig_expand = torch.tile(G_sig, [para.fac_of_imperfectCSI, 1, 1])
    Hr_Los_expand_torch = torch.tile(Hr_Los, [para.fac_of_imperfectCSI, 1, 1])
    Hr_sig_expand = torch.tile(Hr_sig, [para.fac_of_imperfectCSI, 1, 1])
    pd_expand_torch = torch.tile(pd_torch, [para.fac_of_imperfectCSI, 1, 1])
    ps_expand_torch = torch.tile(ps_torch, [para.fac_of_imperfectCSI, 1, 1])
    W_expand = torch.tile(W, [para.fac_of_imperfectCSI, 1, 1])
    theta_expand = torch.tile(theta, [para.fac_of_imperfectCSI, 1, 1])
    weight_expand_torch = torch.tile(weight_torch, [para.fac_of_imperfectCSI, 1])
    
    Hd_w_expand = fac_ture_torch * Hd_w_expand + fac_error_torch * torch.complex(torch.randn(para.nr_of_samples_per_batch * para.fac_of_imperfectCSI, para.K, para.M, device = para.device), torch.randn(para.nr_of_samples_per_batch * para.fac_of_imperfectCSI, para.K, para.M, device = para.device))
    G_sig_expand = fac_ture_torch * G_sig_expand + fac_error_torch * torch.complex(torch.randn(para.nr_of_samples_per_batch * para.fac_of_imperfectCSI, para.N, para.M, device = para.device), torch.randn(para.nr_of_samples_per_batch * para.fac_of_imperfectCSI, para.N, para.M, device = para.device))
    Hr_sig_expand = fac_ture_torch * Hr_sig_expand + fac_error_torch * torch.complex(torch.randn(para.nr_of_samples_per_batch * para.fac_of_imperfectCSI, para.K, para.N, device = para.device), torch.randn(para.nr_of_samples_per_batch * para.fac_of_imperfectCSI, para.K, para.N, device = para.device))
    
    Hd = pd_expand_torch * Hd_w_expand
    
    G = eb1_torch*G_Los_expand_torch + eb2_torch*G_sig_expand
    Hr = ps_expand_torch*(eb1_torch*Hr_Los_expand_torch + eb2_torch*Hr_sig_expand)

    Hr_shape = torch.reshape(bdiag(torch.reshape(Hr, [-1, para.N, 1])), [-1, para.K, para.N, para.N])
    Hr_G = torch.matmul(Hr_shape, torch.unsqueeze(G, axis=1))
    theta_t = torch.tile(torch.unsqueeze(theta_expand, axis=1).transpose(-2,-1).conj(), [1, para.K, 1, 1])
    H = Hd + torch.squeeze(torch.matmul(theta_t, Hr_G))
    
    mu, lambda_torch = K_update_mu_lambda(para, H, W_expand, Pt_torch)
    W = K_update_W(para, H, W_expand, weight_expand_torch, mu, lambda_torch, Pt_torch)
    return Get_SINR(para, H, W, weight_expand_torch)
    
    
def K_update_mu_lambda(para, H, W, Pt_torch):
    M = para.M
    N = para.N
    K = para.K
    H_W = torch.matmul(H, W)    
    W_WH_sum = torch.matmul(W, W.transpose(1,2).conj())
    norm_W_sum = torch.square(torch.linalg.norm(W, ord='fro', axis=[-2, -1]))
    norm_W_sum = torch.tile(torch.reshape(norm_W_sum, [-1, 1]), [1, para.K])
    mu_temp = bDiag(torch.matmul(torch.matmul(H, W_WH_sum), H.transpose(1,2).conj()))
    mu = torch.multiply(torch.div(torch.ones_like(mu_temp), mu_temp + torch.div(norm_W_sum, Pt_torch)), bDiag(H_W))
    lamda_temp = torch.multiply(mu.conj(), bDiag(H_W))
    lambda_torch = torch.div(torch.ones_like(lamda_temp), torch.ones_like(lamda_temp) - lamda_temp)
    return mu, lambda_torch


def K_update_W(para, H, W, weight, mu, lambda_torch, Pt_torch):
    M = para.M
    N = para.N
    K = para.K
    temp = torch.multiply(torch.multiply(torch.multiply(weight, mu), mu.conj()), lambda_torch)
    W_mat_part1 = bdiag(torch.tile(torch.unsqueeze(torch.div(torch.sum(temp, axis=1), Pt_torch), axis=-1), [1, M]))
    W_mat_part2 = torch.matmul(H.transpose(1,2).conj(), torch.multiply(torch.tile(torch.reshape(temp, [-1, K, 1]), [1, 1, M]), H))
    W_mat = W_mat_part1 + W_mat_part2  
    temp = torch.multiply(torch.multiply(weight, mu), lambda_torch)
    W = torch.multiply(torch.tile(torch.reshape(temp, [-1, 1, K]), [1, M, 1]), torch.matmul(torch.linalg.inv(W_mat), H.transpose(1,2).conj()))
    norm_W_sum = torch.linalg.norm(W, ord='fro', axis=[-2, -1])
    norm_W_sum = torch.tile(torch.reshape(norm_W_sum, [-1, 1, 1]), [1, M, K])
    W = torch.div(W, norm_W_sum) * torch.sqrt(Pt_torch)
    return W


def K_update_theta_PI(para, W, theta, Hd, Hr, G, mu, lambda_torch, weight, gamma):
    M = para.M
    N = para.N
    K = para.K
    W_WH_sum = torch.tile(torch.reshape(torch.matmul(W, W.transpose(1,2).conj()), [-1, 1, M, M]), [1, K, 1, 1])
    Hr_shape = torch.reshape(bdiag(torch.reshape(Hr, [-1, N, 1])), [-1, K, N, N])
    Hr_temp = torch.matmul(Hr_shape, torch.unsqueeze(G, axis=1))
    temp = torch.multiply(torch.multiply(torch.multiply(weight, mu), mu.conj()), lambda_torch)
    A_mat = torch.matmul(torch.matmul(Hr_temp, W_WH_sum), Hr_temp.transpose(-2, -1).conj())
    A = torch.sum(torch.multiply(torch.tile(torch.reshape(temp, [-1, K, 1, 1]), [1, 1, N, N]), A_mat), axis=1)
    temp_Hr_temp = torch.multiply(torch.tile(torch.reshape(temp, [-1, K, 1, 1]), [1, 1, N, M]), Hr_temp)
    
    beta_part1_mat = torch.matmul(torch.matmul(temp_Hr_temp, W_WH_sum).transpose(1,2), torch.tile(torch.reshape(Hd.transpose(-2, -1).conj(), [-1, 1, M, K]), [1, N, 1, 1]))
    beta_part1_mat = torch.reshape(torch.sum(bDiag(torch.reshape(beta_part1_mat, [-1, K, K])), axis=1), [-1, N, 1])
    temp_2 = torch.multiply(torch.multiply(weight, mu.conj()), lambda_torch)
    temp_2_Hr_temp = torch.multiply(torch.tile(torch.reshape(temp_2, [-1, K, 1, 1]), [1, 1, N, M]), Hr_temp)
    beta_part2_mat = torch.matmul(temp_2_Hr_temp.transpose(1,2), torch.tile(torch.reshape(W, [-1, 1, M, K]), [1, N, 1, 1]))
    beta_part2_mat = torch.reshape(torch.sum(bDiag(torch.reshape(beta_part2_mat, [-1, K, K])), axis=1), [-1, N, 1])
    beta = beta_part1_mat - beta_part2_mat
    nr_of_samples_in = beta.size()[0]
    zero_vec = torch.zeros(nr_of_samples_in, 1, 1, dtype=torch.complex128, device=para.device)
    B = torch.cat((torch.cat((-A, -beta), -1), torch.cat((-beta.transpose(-2, -1).conj(), -zero_vec), -1)), -2)
    R = B + gamma*torch.tile(torch.reshape(torch.eye(N+1, device=para.device), [1, N+1, N+1]), [nr_of_samples_in, 1, 1])    
    one_vec = torch.ones(nr_of_samples_in, 1, 1, dtype=torch.complex128, device=para.device)
    x = torch.cat((theta, one_vec), axis=-2)
    x = torch.exp(1j*torch.angle(torch.matmul(R, x)))
    theta = torch.multiply(x[:, :-1, :], torch.tile(torch.reshape(x[:, -1, :].conj(), [-1, 1, 1]), [1, N, 1]))
    return theta


def K_update_theta_PI_NoAI(para, W, theta, Hd, Hr, G, mu, lambda_torch, weight):
    M = para.M
    N = para.N
    K = para.K
    W_WH_sum = torch.tile(torch.reshape(torch.matmul(W, W.transpose(1,2).conj()), [-1, 1, M, M]), [1, K, 1, 1])
    Hr_shape = torch.reshape(bdiag(torch.reshape(Hr, [-1, N, 1])), [-1, K, N, N])
    Hr_temp = torch.matmul(Hr_shape, torch.unsqueeze(G, axis=1))
    temp = torch.multiply(torch.multiply(torch.multiply(weight, mu), mu.conj()), lambda_torch)
    A_mat = torch.matmul(torch.matmul(Hr_temp, W_WH_sum), Hr_temp.transpose(-2, -1).conj())
    A = torch.sum(torch.multiply(torch.tile(torch.reshape(temp, [-1, K, 1, 1]), [1, 1, N, N]), A_mat), axis=1)
    temp_Hr_temp = torch.multiply(torch.tile(torch.reshape(temp, [-1, K, 1, 1]), [1, 1, N, M]), Hr_temp)
    
    beta_part1_mat = torch.matmul(torch.matmul(temp_Hr_temp, W_WH_sum).transpose(1,2), torch.tile(torch.reshape(Hd.transpose(-2, -1).conj(), [-1, 1, M, K]), [1, N, 1, 1]))
    beta_part1_mat = torch.reshape(torch.sum(bDiag(torch.reshape(beta_part1_mat, [-1, K, K])), axis=1), [-1, N, 1])
    temp_2 = torch.multiply(torch.multiply(weight, mu.conj()), lambda_torch)
    temp_2_Hr_temp = torch.multiply(torch.tile(torch.reshape(temp_2, [-1, K, 1, 1]), [1, 1, N, M]), Hr_temp)
    beta_part2_mat = torch.matmul(temp_2_Hr_temp.transpose(1,2), torch.tile(torch.reshape(W, [-1, 1, M, K]), [1, N, 1, 1]))
    beta_part2_mat = torch.reshape(torch.sum(bDiag(torch.reshape(beta_part2_mat, [-1, K, K])), axis=1), [-1, N, 1])
    beta = beta_part1_mat - beta_part2_mat
    nr_of_samples_in = beta.size()[0]
    zero_vec = torch.zeros(nr_of_samples_in, 1, 1, dtype=torch.complex128, device=para.device)
    B = torch.cat((torch.cat((-A, -beta), -1), torch.cat((-beta.transpose(-2, -1).conj(), -zero_vec), -1)), -2)
    B_norm = torch.linalg.norm(B, ord='fro', axis=[-2, -1])
    R = B + bdiag(torch.tile(torch.reshape(B_norm, [-1, 1, 1]), [1, N+1, 1])) 
    one_vec = torch.ones(nr_of_samples_in, 1, 1, dtype=torch.complex128, device=para.device)
    x = torch.cat((theta, one_vec), axis=-2)
    x = torch.exp(1j*torch.angle(torch.matmul(R, x)))
    theta = torch.multiply(x[:, :-1, :], torch.tile(torch.reshape(x[:, -1, :].conj(), [-1, 1, 1]), [1, N, 1]))
    return theta  

    
def K_update_theta_PI_ImperfectCSI(para, W, theta, Hd, Hr, G, mu, lambda_torch, weight, gamma, B_old, gstep):
    M = para.M
    N = para.N
    K = para.K
    W_WH_sum = torch.tile(torch.reshape(torch.matmul(W, W.transpose(1,2).conj()), [-1, 1, M, M]), [1, K, 1, 1])
    Hr_shape = torch.reshape(bdiag(torch.reshape(Hr, [-1, N, 1])), [-1, K, N, N])
    Hr_temp = torch.matmul(Hr_shape, torch.unsqueeze(G, axis=1))
    temp = torch.multiply(torch.multiply(torch.multiply(weight, mu), mu.conj()), lambda_torch)
    A_mat = torch.matmul(torch.matmul(Hr_temp, W_WH_sum), Hr_temp.transpose(-2, -1).conj())
    A = torch.sum(torch.multiply(torch.tile(torch.reshape(temp, [-1, K, 1, 1]), [1, 1, N, N]), A_mat), axis=1)
    temp_Hr_temp = torch.multiply(torch.tile(torch.reshape(temp, [-1, K, 1, 1]), [1, 1, N, M]), Hr_temp)
    
    beta_part1_mat = torch.matmul(torch.matmul(temp_Hr_temp, W_WH_sum).transpose(1,2), torch.tile(torch.reshape(Hd.transpose(-2, -1).conj(), [-1, 1, M, K]), [1, N, 1, 1]))
    beta_part1_mat = torch.reshape(torch.sum(bDiag(torch.reshape(beta_part1_mat, [-1, K, K])), axis=1), [-1, N, 1])
    temp_2 = torch.multiply(torch.multiply(weight, mu.conj()), lambda_torch)
    temp_2_Hr_temp = torch.multiply(torch.tile(torch.reshape(temp_2, [-1, K, 1, 1]), [1, 1, N, M]), Hr_temp)
    beta_part2_mat = torch.matmul(temp_2_Hr_temp.transpose(1,2), torch.tile(torch.reshape(W, [-1, 1, M, K]), [1, N, 1, 1]))
    beta_part2_mat = torch.reshape(torch.sum(bDiag(torch.reshape(beta_part2_mat, [-1, K, K])), axis=1), [-1, N, 1])
    beta = beta_part1_mat - beta_part2_mat
    nr_of_samples_in = beta.size()[0]
    zero_vec = torch.zeros(nr_of_samples_in, 1, 1, dtype=torch.complex128, device=para.device)
    B = (1-gstep) * B_old + gstep * torch.cat((torch.cat((-A, -beta), -1), torch.cat((-beta.transpose(-2, -1).conj(), -zero_vec), -1)), -2)
    R = B + gamma*torch.tile(torch.reshape(torch.eye(N+1, device=para.device), [1, N+1, N+1]), [nr_of_samples_in, 1, 1])    
    one_vec = torch.ones(nr_of_samples_in, 1, 1, dtype=torch.complex128, device=para.device)
    x = torch.cat((theta, one_vec), axis=-2)
    x = torch.exp(1j*torch.angle(torch.matmul(R, x)))
    theta = torch.multiply(x[:, :-1, :], torch.tile(torch.reshape(x[:, -1, :].conj(), [-1, 1, 1]), [1, N, 1]))
    return theta, B


def K_init_W_GNN(para, Hd, H_RIS, mu, lambda_torch, weight, Pt_torch, self):
    M = para.M
    N = para.N
    K = para.K
    nr_of_samples_in = mu.size()[0]
    H = Hd + H_RIS
    mu = torch.unsqueeze(mu, axis=-1)
    lambda_torch = torch.unsqueeze(lambda_torch, axis=-1)
    weight = torch.unsqueeze(weight, axis=-1)

    net_input = torch.cat((Hd, H_RIS, mu, lambda_torch, weight), axis=-1)
    net_input = torch.cat((torch.real(net_input), torch.imag(net_input)), axis=-1)
    
    z1 = F.relu(net_input@self.w_in + self.b_in)
    z2_combine = []
    for ii in range(K):
        temp = []
        for jj in range(K):
            if ii != jj:
                temp.append(z1[:, jj, :]) # 这边还可以加一个全连接处理一下
        z2_combine.append(torch.stack(temp))
    z2_combine = torch.max(torch.stack(z2_combine), axis=1).values.transpose(0,1)
    z2_combine = F.relu(z2_combine@self.w_aggregate + self.b_aggregate)
    z2_input = torch.cat((z1, z2_combine), axis=-1)
    z3 = F.relu(z2_input@self.w_combine + self.b_combine)
    net_output = z3@self.w_out + self.b_out

    power_eff = torch.nn.functional.softmax(net_output[:, :, 0], dim=-1)*Pt_torch
    dir_eff = torch.nn.functional.softmax(net_output[:, :, 1], dim=-1)*Pt_torch

    W_mat_part1 = torch.tile(torch.unsqueeze(torch.eye(M, device=para.device), axis=0), [nr_of_samples_in, 1, 1])
    W_mat_part2 = torch.matmul(H.transpose(1,2).conj(), torch.multiply(torch.tile(torch.reshape(dir_eff, [-1, K, 1]), [1, 1, M]), H))
    W_mat = W_mat_part1 + W_mat_part2
    power_eff = torch.sqrt(power_eff)
    W_temp = torch.matmul(torch.linalg.inv(W_mat), H.transpose(1,2).conj())
    norm_W_temp =torch.tile(torch.unsqueeze(torch.linalg.norm(W_temp, ord=2, axis=[-2]), axis=1), [1, M, 1]) 
    W_temp = torch.div(W_temp, norm_W_temp)
    W = torch.multiply(torch.tile(torch.unsqueeze(power_eff, axis=-2), [1, M, 1]), W_temp)    
    return W


def K_init_theta_PI(para, W, theta, Hd, Hr, G, mu, lambda_torch, weight, gamma):
    M = para.M
    N = para.N
    K = para.K
    W_WH_sum = torch.tile(torch.reshape(torch.matmul(W, W.transpose(1,2).conj()), [-1, 1, M, M]), [1, K, 1, 1])
    Hr_shape = torch.reshape(bdiag(torch.reshape(Hr, [-1, N, 1])), [-1, K, N, N])
    Hr_temp = torch.matmul(Hr_shape, torch.unsqueeze(G, axis=1))
    temp = torch.multiply(torch.multiply(torch.multiply(weight, mu), mu.conj()), lambda_torch)
    A_mat = torch.matmul(torch.matmul(Hr_temp, W_WH_sum), Hr_temp.transpose(-2, -1).conj())
    A = torch.sum(torch.multiply(torch.tile(torch.reshape(temp, [-1, K, 1, 1]), [1, 1, N, N]), A_mat), axis=1)
    temp_Hr_temp = torch.multiply(torch.tile(torch.reshape(temp, [-1, K, 1, 1]), [1, 1, N, M]), Hr_temp)
    
    beta_part1_mat = torch.matmul(torch.matmul(temp_Hr_temp, W_WH_sum).transpose(1,2), torch.tile(torch.reshape(Hd.transpose(-2, -1).conj(), [-1, 1, M, K]), [1, N, 1, 1]))
    beta_part1_mat = torch.reshape(torch.sum(bDiag(torch.reshape(beta_part1_mat, [-1, K, K])), axis=1), [-1, N, 1])
    temp_2 = torch.multiply(torch.multiply(weight, mu.conj()), lambda_torch)
    temp_2_Hr_temp = torch.multiply(torch.tile(torch.reshape(temp_2, [-1, K, 1, 1]), [1, 1, N, M]), Hr_temp)
    beta_part2_mat = torch.matmul(temp_2_Hr_temp.transpose(1,2), torch.tile(torch.reshape(W, [-1, 1, M, K]), [1, N, 1, 1]))
    beta_part2_mat = torch.reshape(torch.sum(bDiag(torch.reshape(beta_part2_mat, [-1, K, K])), axis=1), [-1, N, 1])
    beta = beta_part1_mat - beta_part2_mat
    nr_of_samples_in = beta.size()[0]
    zero_vec = torch.zeros(nr_of_samples_in, 1, 1, dtype=torch.complex128, device=para.device)
    one_vec = torch.ones(nr_of_samples_in, 1, 1, dtype=torch.complex128, device=para.device)
    x = torch.cat((theta, one_vec), axis=-2)    
    B = torch.cat((torch.cat((-A, -beta), -1), torch.cat((-beta.transpose(-2, -1).conj(), -zero_vec), -1)), -2)
    
    R = B + gamma*torch.tile(torch.reshape(torch.eye(N+1, device=para.device), [1, N+1, N+1]), [nr_of_samples_in, 1, 1]) 
    x = torch.exp(1j*torch.angle(torch.matmul(R, x)))
    
    theta = torch.multiply(x[:, :-1, :], torch.tile(torch.reshape(x[:, -1, :].conj(), [-1, 1, 1]), [1, N, 1]))
    return theta
