#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rescorla Wagner model with policy gradient (PG) for meta-learning epsilon parameter
Contexts:
*Stable: reward probabiliies stay the same during all trials
*Volatile: reward probabilities shuffle every couple of trials
*Reinforced variability: reward dependent on how variable options are chosen
@author: Janne Reynders; janne.reynders@ugent.be
"""
import numpy as np                  
import pandas as pd                 
import matplotlib.pyplot as plt     
import os
import random

#simulate a Rescorla Wagner model with constant epsilon
def simulate_RW(alpha, eps, T, Q_int, reward_prob):
    K=len(reward_prob) #amont of choice options
    #alpha      --->        learning rate
    #eps        --->        epsilon
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #reward prob -->        probabilites to recieve a reward, associated to each option

    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards

    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    
    
    for t in range(T):
        
        # store values for Q
        Q_k_stored[t,:] = Q_k   
              
        # make choice based on choice probababilities
        rand[t] = np.random.choice([0,1], p=[1-eps,eps])
        if rand[t] == 0:
            k[t] = np.argmax(Q_k)
        if rand[t] == 1:
            k[t] = np.random.choice(range(K))
        
        a1 = reward_prob[k[t]]
        a0 = 1-a1
        r[t] = np.random.choice([1, 0], p=[a1, a0])

        # update values
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + alpha * delta_k    

    return k, r, Q_k_stored

#simulation of Rescorla-Wagner model with meta-learning of epsilon 
#meta-learning goes through parameter ML which is transformed to epsilon with a logit transformation
#rewards are baselined
def simulate_RW_MLeps_variable(Q_alpha, ML_alpha, T, Q_int, eps_mean_int, eps_std_int, reward_prob, scale, threshold):
    K=len(reward_prob) #the amount of choice options
    Freq = np.zeros((K), dtype = float)
    #alpha          --->        learning rate
    #ML_alpha       --->        learning rate for meta-learning parameter (list of two LR's: one for mean ML and one for std ML)
    #T              --->        amount of trials for each simulation
    #K              --->        amount of choice options
    #Q_int          --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #eps_mean_int   --->        initial value for the mean of eps
    #eps_std_int    --->        initial value for the standard deviation of eps
    #reward prob    --->        probabilites to recieve a reward, associated to each option
    #scale          --->        scale with which ML-parameter gets transformed to epsilon in a logit transformation
    #threshold      --->        % of least frequently chosen options which will be rewarded

    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on epsilon
    r = np.zeros((T), dtype=float) #vector of rewards
    av_reward = 0.5 #starting baseline reward


    ML_stored = np.zeros((T), dtype=float)
    ML_mean_stored = np.zeros((T), dtype=float)
    ML_std_stored = np.zeros((T), dtype=float)
    eps_var = np.zeros((T), dtype=float)
    eps_var_mean = np.zeros((T), dtype=float)
    eps_var_std = np.zeros((T), dtype=float)
   

    ML_alpha_mean, ML_alpha_std = ML_alpha
    
    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    #initial values of the meta-learning (ML) parameter
    ML_mean_int = (1/scale)*np.log(eps_mean_int/(1-eps_mean_int))
    ML_std_int = (1/scale)*np.log(eps_std_int/(1-eps_std_int))
    ML_int = np.random.normal(loc = ML_mean_int, scale=ML_std_int)
    
    ML_mean = ML_mean_int
    ML_std = ML_std_int
    ML = ML_int   

    log_ML_mean = np.log(ML_mean) #the parameter that gets updated at the end of the loop
    log_ML_std = np.log(ML_std) #the parameter that gets updated at the end of the loop

    for t in range(T):
        #epsilon is calculated with a logit transformation of the ML
        eps = np.exp(scale*ML)/(1+np.exp(scale*ML))
        eps_mean = np.exp(scale*ML_mean)/(1+np.exp(scale*ML_mean))
        eps_std = np.exp(scale*ML_std)/(1+np.exp(scale*ML_std))
        eps_var[t] = eps
        eps_var_mean[t] = eps_mean
        eps_var_std[t] = eps_std
        # store values for Q and epsilon
        Q_k_stored[t,:] = Q_k
        ML_stored[t] = ML
        ML_mean_stored[t] = ML_mean
        ML_std_stored[t] = ML_std

      
        # make choice based on choice probababilities
        rand[t] = np.random.choice(2, p=[1-eps,eps])
        if rand[t] == 0:
         k[t] = np.argmax(Q_k)
        if rand[t] == 1:
         k[t] = np.random.choice(range(K))

        #count frequency
        index = k[t]
        Freq[index] = Freq[index] + 1
        #multiply by weight to account for recency
        Freq = 0.95*Freq

        Freq_sort = np.copy(Freq)
        Freq_sort.sort()
        new_index = np.where(Freq_sort==Freq[index])[0][0]
        threshold_index = np.round(threshold*K) - 1
        if new_index <= threshold_index:
            r[t] = 1
        else: 
            r[t] = 0
     
        # update Q values
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + Q_alpha * delta_k

        # baseline reward
        alpha_reward = 0.25
        av_reward = av_reward + alpha_reward * (r[t] - av_reward)
        baseline_reward = r[t]-av_reward
        
        # update log(mean) and log(std)   
        dif = ML - ML_mean
        if ML > ML_mean:
            #update_log_mean = (ML_alpha_pos*baseline_reward*dif*ML) / ((ML_std)**2) #*ML in nominator ensures update in logspace
            update_ML_mean = (ML_alpha_mean*baseline_reward*dif) / ((ML_std)**2) #*ML in nominator ensures update in logspace
        if ML < ML_mean:
            #update_log_mean = (ML_alpha_neg*baseline_reward*dif*ML) / ((ML_std)**2) #*ML in nominator ensures update in logspace
            update_ML_mean = (ML_alpha_mean*baseline_reward*dif) / ((ML_std)**2) #*ML in nominator ensures update in logspace
            
        #log_ML_mean = log_ML_mean + update_log_mean
        #ML_mean = np.exp(log_ML_mean)
        ML_mean = ML_mean + update_ML_mean
     
        dif2 = ((dif)**2)-((ML_std)**2)
        update_log_std =  (ML_alpha_std*baseline_reward*dif2) /((ML_std)**2) #ML_std^2 in denominator ensures update in logspace (instead of ML_std^3)
        
        log_ML_std = log_ML_std + update_log_std 
        ML_std = np.exp(log_ML_std)
     
        ML = np.random.normal(loc=ML_mean, scale=ML_std)
  
    return k, r, Q_k_stored, eps_var, eps_var_mean, eps_var_std

def simulate_RW_MLeps_stable(Q_alpha, ML_alpha, T, Q_int, eps_mean_int, eps_std_int, reward_prob, scale):
    K=len(reward_prob) #the amount of choice options
    #alpha          --->        learning rate
    #ML_alpha       --->        learning rate for meta-learning parameter (list of two LR's: one for mean ML and one for std ML)
    #T              --->        amount of trials for each simulation
    #K              --->        amount of choice options
    #Q_int          --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #eps_mean_int   --->        initial value for the mean of eps
    #eps_std_int    --->        initial value for the standard deviation of eps
    #reward prob    --->        probabilites to recieve a reward, associated to each option
    #scale          --->        scale with which ML-parameter gets transformed to epsilon in a logit transformation
   
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on epsilon
    r = np.zeros((T), dtype=float) #vector of rewards
    av_reward = 0.5 #starting baseline reward

    ML_stored = np.zeros((T), dtype=float)
    ML_mean_stored = np.zeros((T), dtype=float)
    ML_std_stored = np.zeros((T), dtype=float)
    eps_sta = np.zeros((T), dtype=float)
    eps_sta_mean = np.zeros((T), dtype=float)
    eps_sta_std = np.zeros((T), dtype=float)
    

    ML_alpha_mean, ML_alpha_std = ML_alpha
    
    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    #initial values of the meta-learning (ML) parameter
    ML_mean_int = (1/scale)*np.log(eps_mean_int/(1-eps_mean_int))
    ML_std_int = (1/scale)*np.log(eps_std_int/(1-eps_std_int))
    ML_int = np.random.normal(loc = ML_mean_int, scale=ML_std_int)
    
    ML_mean = ML_mean_int
    ML_std = ML_std_int
    ML = ML_int   

    log_ML_mean = np.log(ML_mean) #the parameter that gets updated at the end of the loop
    log_ML_std = np.log(ML_std) #the parameter that gets updated at the end of the loop

    for t in range(T):
        #epsilon is calculated with a logit transformation of the ML
        eps = np.exp(scale*ML)/(1+np.exp(scale*ML))
        eps_mean = np.exp(scale*ML_mean)/(1+np.exp(scale*ML_mean))
        eps_std = np.exp(scale*ML_std)/(1+np.exp(scale*ML_std))

        eps_sta[t] = eps
        eps_sta_mean[t] = eps_mean
        eps_sta_std[t] = eps_std


        # store values for Q and epsilon
        Q_k_stored[t,:] = Q_k
        ML_stored[t] = ML
        ML_mean_stored[t] = ML_mean
        ML_std_stored[t] = ML_std
      
        # make choice based on choice probababilities
        rand[t] = np.random.choice(2, p=[1-eps,eps])
        if rand[t] == 0:
         k[t] = np.argmax(Q_k)
        if rand[t] == 1:
         k[t] = np.random.choice(range(K))
           
        a1 = reward_prob[k[t]]
        a0 = 1-a1
        r[t] = np.random.choice([1, 0], p=[a1, a0])

        # update Q values
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + Q_alpha * delta_k

        # baseline reward
        alpha_reward = 0.25
        av_reward = av_reward + alpha_reward * (r[t] - av_reward)
        baseline_reward = r[t]-av_reward
        
        # update log(mean) and log(std)   
        dif = ML - ML_mean
        if ML > ML_mean:
            #update_log_mean = (ML_alpha_pos*baseline_reward*dif*ML) / ((ML_std)**2) #*ML in nominator ensures update in logspace
            update_ML_mean = (ML_alpha_mean*baseline_reward*dif) / ((ML_std)**2) #*ML in nominator ensures update in logspace
        if ML < ML_mean:
            #update_log_mean = (ML_alpha_neg*baseline_reward*dif*ML) / ((ML_std)**2) #*ML in nominator ensures update in logspace
            update_ML_mean = (ML_alpha_mean*baseline_reward*dif) / ((ML_std)**2) #*ML in nominator ensures update in logspace
            
        #log_ML_mean = log_ML_mean + update_log_mean
        #ML_mean = np.exp(log_ML_mean)
        ML_mean = ML_mean + update_ML_mean
     
        dif2 = ((dif)**2)-((ML_std)**2)
        update_log_std =  (ML_alpha_std*baseline_reward*dif2) /((ML_std)**2) #ML_std^2 in denominator ensures update in logspace (instead of ML_std^3)
        
        log_ML_std = log_ML_std + update_log_std 
        ML_std = np.exp(log_ML_std)
     
        ML = np.random.normal(loc=ML_mean, scale=ML_std)
  
    return k, r, Q_k_stored, eps_sta, eps_sta_mean, eps_sta_std

def simulate_RW_MLeps_volatile(Q_alpha, ML_alpha, T, Q_int, eps_mean_int, eps_std_int, reward_prob, rot, scale):
    K=len(reward_prob) #the amount of choice options
    #alpha          --->        learning rate
    #ML_alpha       --->        learning rate for meta-learning parameter (list of two LR's: one for mean ML and one for std ML)
    #T              --->        amount of trials for each simulation
    #K              --->        amount of choice options
    #Q_int          --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #eps_mean_int   --->        initial value for the mean of eps
    #eps_std_int    --->        initial value for the standard deviation of eps
    #reward prob    --->        probabilites to recieve a reward, associated to each option
    #rot            --->        amount of trials after which mean reward values rotate among choice options
    #scale          --->        scale with which ML-parameter gets transformed to epsilon in a logit transformation
   
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on epsilon
    r = np.zeros((T), dtype=float) #vector of rewards
    av_reward = 0.5 #starting baseline reward

    ML_stored = np.zeros((T), dtype=float)
    ML_mean_stored = np.zeros((T), dtype=float)
    ML_std_stored = np.zeros((T), dtype=float)
    eps_vol = np.zeros((T), dtype=float)
    eps_vol_mean = np.zeros((T), dtype=float)
    eps_vol_std = np.zeros((T), dtype=float)

    ML_alpha_mean, ML_alpha_std = ML_alpha
    
    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    #initial values of the meta-learning (ML) parameter
    ML_mean_int = (1/scale)*np.log(eps_mean_int/(1-eps_mean_int))
    ML_std_int = (1/scale)*np.log(eps_std_int/(1-eps_std_int))
    ML_int = np.random.normal(loc = ML_mean_int, scale=ML_std_int)
    
    ML_mean = ML_mean_int
    ML_std = ML_std_int
    ML = ML_int   

    log_ML_mean = np.log(ML_mean) #the parameter that gets updated at the end of the loop
    log_ML_std = np.log(ML_std) #the parameter that gets updated at the end of the loop

    for t in range(T):
        #epsilon is calculated with a logit transformation of the ML
        eps = np.exp(scale*ML)/(1+np.exp(scale*ML))
        eps_mean = np.exp(scale*ML_mean)/(1+np.exp(scale*ML_mean))
        eps_std = np.exp(scale*ML_std)/(1+np.exp(scale*ML_std))
        eps_vol[t] = eps
        eps_vol_mean[t] = eps_mean
        eps_vol_std[t] = eps_std
        # store values for Q and epsilon
        Q_k_stored[t,:] = Q_k
        ML_stored[t] = ML
        ML_mean_stored[t] = ML_mean
        ML_std_stored[t] = ML_std
      
        # make choice based on choice probababilities
        rand[t] = np.random.choice(2, p=[1-eps,eps])
        if rand[t] == 0:
         k[t] = np.argmax(Q_k)
        if rand[t] == 1:
         k[t] = np.random.choice(range(K))       

        # generate reward based on normal distribution linked to choice made
        if (t%rot)==0 and t != 0:
            reward_orig = reward_prob.copy()
            while reward_orig == reward_prob:
                np.random.shuffle(reward_prob)
           
        a1 = reward_prob[k[t]]
        a0 = 1-a1
        r[t] = np.random.choice([1, 0], p=[a1, a0])

        # update Q values
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + Q_alpha * delta_k

        # baseline reward
        alpha_reward = 0.25
        av_reward = av_reward + alpha_reward * (r[t] - av_reward)
        baseline_reward = r[t]-av_reward
        
        # update log(mean) and log(std)   
        dif = ML - ML_mean
        if ML > ML_mean:
            #update_log_mean = (ML_alpha_pos*baseline_reward*dif*ML) / ((ML_std)**2) #*ML in nominator ensures update in logspace
            update_ML_mean = (ML_alpha_mean*baseline_reward*dif) / ((ML_std)**2) #*ML in nominator ensures update in logspace
        if ML < ML_mean:
            #update_log_mean = (ML_alpha_neg*baseline_reward*dif*ML) / ((ML_std)**2) #*ML in nominator ensures update in logspace
            update_ML_mean = (ML_alpha_mean*baseline_reward*dif) / ((ML_std)**2) #*ML in nominator ensures update in logspace
            
        #log_ML_mean = log_ML_mean + update_log_mean
        #ML_mean = np.exp(log_ML_mean)
        ML_mean = ML_mean + update_ML_mean
     
        dif2 = ((dif)**2)-((ML_std)**2)
        update_log_std =  (ML_alpha_std*baseline_reward*dif2) /((ML_std)**2) #ML_std^2 in denominator ensures update in logspace (instead of ML_std^3)
        
        log_ML_std = log_ML_std + update_log_std 
        ML_std = np.exp(log_ML_std)
     
        ML = np.random.normal(loc=ML_mean, scale=ML_std)
  
    return k, r, Q_k_stored, eps_vol, eps_vol_mean, eps_vol_std

sim_nr = 5
threshold = 0.5
percentage=50
T=5000
Q_int = 1
Q_alpha = 0.5
amount_of_sim = 50
reward_prob = [0.2,0.2,0.8,0.2,0.2,0.2,0.2,0.2,0.8,0.2]
rot=10
K=10
ML_alpha = [0.01, 0.01, 0] #LR std, LR pos and LR neg

window = 50

eps_mean_int = 0.2
eps_std_int = 0.574

#################################################################
#simulations
#################################################################


total_eps_var = np.zeros(amount_of_sim)
total_eps_var_mean = np.zeros(amount_of_sim)
total_eps_var_std = np.zeros(amount_of_sim)
reward_var = np.zeros(amount_of_sim)
for sim in range(amount_of_sim):
    k, r, Q_k_stored, eps_var, eps_var_mean, eps_var_std = simulate_RW_MLeps_variable(Q_alpha=Q_alpha, ML_alpha=ML_alpha, T=T, Q_int=Q_int, eps_mean_int=eps_mean_int, eps_std_int=eps_std_int, reward_prob=reward_prob, scale=1, threshold = threshold)
    #mean reward over time (cumulative reward divided by current trial)
    total_eps_var[sim] = np.mean(eps_var)
    total_eps_var_mean[sim] = np.mean(eps_var_mean)
    total_eps_var_std[sim] = np.mean(eps_var_std)
    reward_var[sim] = np.mean(r)

av_eps_var = np.mean(total_eps_var)
av_eps_var_mean = np.mean(total_eps_var_mean)
av_eps_var_std = np.mean(total_eps_var_std)
av_reward_var = np.mean(reward_var)

std_eps_var = np.std(total_eps_var)
std_eps_var_mean = np.std(total_eps_var_mean)
std_eps_var_std = np.std(total_eps_var_std)
std_reward_var = np.std(reward_var)



total_eps_sta = np.zeros(amount_of_sim)
total_eps_sta_mean = np.zeros(amount_of_sim)
total_eps_sta_std = np.zeros(amount_of_sim)
reward_sta = np.zeros(amount_of_sim)
for sim in range(amount_of_sim):
    k, r, Q_k_stored, eps_sta, eps_sta_mean, eps_sta_std = simulate_RW_MLeps_stable(Q_alpha=Q_alpha, ML_alpha=ML_alpha, T=T, Q_int=Q_int, eps_mean_int=eps_mean_int, eps_std_int=eps_std_int, reward_prob=reward_prob, scale=1)
    #mean reward over time (cumulative reward divided by current trial)
    total_eps_sta[sim] = np.mean(eps_sta)
    total_eps_sta_mean[sim] = np.mean(eps_sta_mean)
    total_eps_sta_std[sim] = np.mean(eps_sta_std)
    reward_sta[sim] = np.mean(r)

av_eps_sta = np.mean(total_eps_sta)
av_eps_sta_mean = np.mean(total_eps_sta_mean)
av_eps_sta_std = np.mean(total_eps_sta_std)
av_reward_sta = np.mean(reward_sta)

std_eps_sta = np.std(total_eps_sta)
std_eps_sta_mean = np.std(total_eps_sta_mean)
std_eps_sta_std = np.std(total_eps_sta_std)
std_reward_sta = np.std(reward_sta)





total_eps_vol = np.zeros(amount_of_sim)
total_eps_vol_mean = np.zeros(amount_of_sim)
total_eps_vol_std = np.zeros(amount_of_sim)
reward_vol = np.zeros(amount_of_sim)
for sim in range(amount_of_sim):
    k, r, Q_k_stored, eps_vol, eps_vol_mean, eps_vol_std = simulate_RW_MLeps_volatile(Q_alpha=Q_alpha, ML_alpha=ML_alpha, T=T, Q_int=Q_int, eps_mean_int=eps_mean_int, eps_std_int=eps_std_int, reward_prob=reward_prob, rot=rot, scale=1)
    #mean reward over time (cumulative reward divided by current trial)
    total_eps_vol[sim] = np.mean(eps_vol)
    total_eps_vol_mean[sim] = np.mean(eps_vol_mean)
    total_eps_vol_std[sim] = np.mean(eps_vol_std)
    reward_vol[sim] = np.mean(r)

av_eps_vol = np.mean(total_eps_vol)
av_eps_vol_mean = np.mean(total_eps_vol_mean)
av_eps_vol_std = np.mean(total_eps_vol_std)
av_reward_vol = np.mean(reward_vol)

std_eps_vol = np.std(total_eps_vol)
std_eps_vol_mean = np.std(total_eps_vol_mean)
std_eps_vol_std = np.std(total_eps_vol_std)
std_reward_vol = np.std(reward_vol)





save_dir = '/Users/jareynde/OneDrive - UGent/1A_Main_Project/Models/Simulations/Random/Epsilon/Output'

title1 = f'average sampled epsilon over {amount_of_sim} simulations, Q-value learning rate of {Q_alpha}\naverage gained rewards are for stable {round(av_reward_sta, 2)} +/- {round(std_reward_sta,2)}, for volatile {round(av_reward_vol,2)} +/- {round(std_reward_vol,2)} and for variable {round(av_reward_var,2)} +/- {round(std_reward_var,2)}'
fig_name = os.path.join(save_dir, f'{sim_nr}_epsilon_threshold{percentage}')
fig, ax= plt.subplots(figsize=(10, 7))
ax.bar(['stable context', 'volatile context', 'variable context'], [av_eps_sta, av_eps_vol, av_eps_var], yerr=[std_eps_sta, std_eps_vol, std_eps_var])
ax.set_ylabel('epsilon')
plt.title(title1)
plt.savefig(fig_name)
plt.show()

title2 = f'average updated mean epsilon over {amount_of_sim} simulations, Q-value learning rate of {Q_alpha}\naverage gained rewards are for stable {round(av_reward_sta, 2)} +/- {round(std_reward_sta,2)}, for volatile {round(av_reward_vol,2)} +/- {round(std_reward_vol,2)} and for variable {round(av_reward_var,2)} +/- {round(std_reward_var,2)}'
fig_name = os.path.join(save_dir, f'{sim_nr}_mean_epsilon_threshold{percentage}')
fig, ax= plt.subplots(figsize=(10, 7))
ax.bar(['stable context', 'volatile context', 'variable context'], [av_eps_sta_mean, av_eps_vol_mean, av_eps_var_mean], yerr=[std_eps_sta_mean, std_eps_vol_mean, std_eps_var_mean])
ax.set_ylabel('epsilon')
plt.title(title2) 
plt.savefig(fig_name)
plt.show()
