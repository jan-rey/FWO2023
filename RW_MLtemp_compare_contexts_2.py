#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rescorla Wagner model with policy gradient (PG) for meta-learning temperature parameter
Difference between 'RW_MLtemp_compare_contexts_2.py' (dit script) en 'RW_MLtemp_compare_contexts.py', 
is dat ik in dit script ook een logit transformatie van de meta-learning temperature introduceer (net zoals in MLeps)
maar hier dan op een schaal tussen 0 en 10 (voor temp).
This script is now adapted such that the standard deviation is constant. 
Temp_std = ML_std and it is not transformed with logit.
Also, there is a max on ML and ML_mean since otherwise it grows too much
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

#simulate a Rescorla Wagner model with constant temperature
def simulate_RW(alpha, temp, T, Q_int, reward_prob):
    K=len(reward_prob) #amont of choice options
    #alpha      --->        learning rate
    #temp       --->        temperature
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
        sum_soft = 0
        inv_temp = 1/temp

        for j in range(K):
            soft = np.exp(inv_temp*Q_k[j]) #using inverse temperature
            sum_soft = sum_soft + soft
        
        p_k = np.zeros(K) #vector to store probabilities for each choice
        for j in range(K):
            p_k[j] = np.exp(inv_temp*Q_k[j]) / sum_soft
      
        # make noisy choice based on choice probababilities
        k[t] = np.random.choice(range(K), p=p_k)
        
        a1 = reward_prob[k[t]]
        a0 = 1-a1
        r[t] = np.random.choice([1, 0], p=[a1, a0])

        # update values
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + alpha * delta_k    

    return k, r, Q_k_stored

#simulation of Rescorla-Wagner model with meta-learning of temperatre 
#meta-learning goes through parameter ML which is transformed to temperature with a logit transformation
#rewards are baselined
def simulate_RW_MLtemp_variable(Q_alpha, ML_alpha, T, Q_int, temp_mean_int, temp_std_int, reward_prob, scale, threshold):
    Ma = 9.99
    Mi = 0.01
    K=len(reward_prob) #the amount of choice options
    Freq = np.zeros((K), dtype = float)
    #alpha          --->        learning rate
    #ML_alpha       --->        learning rate for meta-learning parameter (list of two LR's: one for mean ML and one for std ML)
    #T              --->        amount of trials for each simulation
    #K              --->        amount of choice options
    #Q_int          --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #temp_mean_int   --->        initial value for the mean of temp
    #temp_std_int    --->        initial value for the standard deviation of temp
    #reward prob    --->        probabilites to recieve a reward, associated to each option
    #scale          --->        scale with which ML-parameter gets transformed to temperature in a logit transformation
    #threshold      --->        % of least frequently chosen options which will be rewarded

    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on temperature
    r = np.zeros((T), dtype=float) #vector of rewards
    av_reward = 0.5 #starting baseline reward


    temp_var = np.zeros((T), dtype=float)
    temp_var_mean = np.zeros((T), dtype=float)
    temp_var_std = np.zeros((T), dtype=float)
   

    ML_alpha_mean, ML_alpha_std = ML_alpha
    
    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    #initial values of the meta-learning (ML) parameter
    ML_mean_int = np.log(((temp_mean_int-Mi)/Ma)/(1-((temp_mean_int-Mi)/Ma)))
    ##################
    #ML_std_int = np.log(((temp_std_int-Mi)/Ma)/(1-((temp_std_int-Mi)/Ma)))
    ML_std_int = temp_std_int
    ML_int = np.random.normal(loc = ML_mean_int, scale=ML_std_int)

    ML_mean = ML_mean_int
    ML_std = ML_std_int
    ML = ML_int   

    log_ML_mean = np.log(ML_mean) #the parameter that gets updated at the end of the loop
    ##################
    #log_ML_std = np.log(ML_std) #the parameter that gets updated at the end of the loop

    for t in range(T):
        #temperature is calculated with a logit transformation of the ML
        temp = Ma*(np.exp(scale*ML)/(1+np.exp(scale*ML))) + Mi
        temp_mean = Ma*(np.exp(scale*ML_mean)/(1+np.exp(scale*ML_mean))) + Mi
        ##################
        #temp_std = Ma*(np.exp(scale*ML_std)/(1+np.exp(scale*ML_std))) + Mi
        temp_std = ML_std

        temp_var[t] = temp
        temp_var_mean[t] = temp_mean
        temp_var_std[t] = temp_std
        # store values for Q
        Q_k_stored[t,:] = Q_k

      
        # make choice based on choice probababilities
        sum_soft = 0
        inv_temp = 1/temp
        #print("we are now at", t, inv_temp)
        #print("we are now at", t, temp_var)
        for j in range(K):
            soft = np.exp(inv_temp*Q_k[j]) #using inverse temperature
            sum_soft = sum_soft + soft
        
        p_k = np.zeros(K) #vector to store probabilities for each choice
        for j in range(K):
            p_k[j] = np.exp(inv_temp*Q_k[j]) / sum_soft
      
        # make noisy choice based on choice probababilities
        k[t] = np.random.choice(range(K), p=p_k)

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
            #update_log_mean = (ML_alpha_mean*baseline_reward*dif*ML) / ((ML_std)**2) #*ML in nominator ensures update in logspace
            update_ML_mean = (ML_alpha_mean*baseline_reward*dif) / ((ML_std)**2) #*ML in nominator ensures update in logspace
        if ML < ML_mean:
            #update_log_mean = (ML_alpha_mean*baseline_reward*dif*ML) / ((ML_std)**2) #*ML in nominator ensures update in logspace
            update_ML_mean = (ML_alpha_mean*baseline_reward*dif) / ((ML_std)**2) #*ML in nominator ensures update in logspace
            
        #log_ML_mean = log_ML_mean + update_log_mean
        #ML_mean = np.exp(log_ML_mean)
        ML_mean = ML_mean + update_ML_mean
        ML_mean = min(ML_mean, 40)
     
        ##################
        #dif2 = ((dif)**2)-((ML_std)**2)
        #update_log_std =  (ML_alpha_std*baseline_reward*dif2) /((ML_std)**2) #ML_std^2 in denominator ensures update in logspace (instead of ML_std^3)
        
        #log_ML_std = log_ML_std + update_log_std 
        #ML_std = np.exp(log_ML_std)
     
        ML = np.random.normal(loc=ML_mean, scale=ML_std)
        #ML = min(ML, 40)
    return k, r, Q_k_stored, temp_var, temp_var_mean, temp_var_std

def simulate_RW_MLtemp_stable(Q_alpha, ML_alpha, T, Q_int, temp_mean_int, temp_std_int, reward_prob, scale):
    Ma = 9.99
    Mi = 0.01
    K=len(reward_prob) #the amount of choice options
    #alpha          --->        learning rate
    #ML_alpha       --->        learning rate for meta-learning parameter (list of two LR's: one for mean ML and one for std ML)
    #T              --->        amount of trials for each simulation
    #K              --->        amount of choice options
    #Q_int          --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #temp_mean_int   --->        initial value for the mean of temp
    #temp_std_int    --->        initial value for the standard deviation of temp
    #reward prob    --->        probabilites to recieve a reward, associated to each option
    #scale          --->        scale with which ML-parameter gets transformed to temperature in a logit transformation
   
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on temperature
    r = np.zeros((T), dtype=float) #vector of rewards
    av_reward = 0.5 #starting baseline reward

    temp_sta = np.zeros((T), dtype=float)
    temp_sta_mean = np.zeros((T), dtype=float)
    temp_sta_std = np.zeros((T), dtype=float)
    

    ML_alpha_mean, ML_alpha_std = ML_alpha
    
    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    #initial values of the meta-learning (ML) parameter
    ML_mean_int = np.log(((temp_mean_int-Mi)/Ma)/(1-((temp_mean_int-Mi)/Ma)))
    ##################
    #ML_std_int = np.log(((temp_std_int-Mi)/Ma)/(1-((temp_std_int-Mi)/Ma)))
    ML_std_int = temp_std_int
    ML_int = np.random.normal(loc = ML_mean_int, scale=ML_std_int)

    ML_mean = ML_mean_int
    ML_std = ML_std_int
    ML = ML_int   

    log_ML_mean = np.log(ML_mean) #the parameter that gets updated at the end of the loop
    ##################
    #log_ML_std = np.log(ML_std) #the parameter that gets updated at the end of the loop

    for t in range(T):
        #temperature is calculated with a logit transformation of the ML
        temp = Ma*(np.exp(scale*ML)/(1+np.exp(scale*ML))) + Mi
        temp_mean = Ma*(np.exp(scale*ML_mean)/(1+np.exp(scale*ML_mean))) + Mi
        ##################
        #temp_std = Ma*(np.exp(scale*ML_std)/(1+np.exp(scale*ML_std))) + Mi
        temp_std = ML_std

        temp_sta[t] = temp
        temp_sta_mean[t] = temp_mean
        temp_sta_std[t] = temp_std


        # store values for Q
        Q_k_stored[t,:] = Q_k
      
        # make choice based on choice probababilities
        sum_soft = 0
        inv_temp = 1/temp

        for j in range(K):
            soft = np.exp(inv_temp*Q_k[j]) #using inverse temperature
            sum_soft = sum_soft + soft
        
        p_k = np.zeros(K) #vector to store probabilities for each choice
        for j in range(K):
            p_k[j] = np.exp(inv_temp*Q_k[j]) / sum_soft
      
        # make noisy choice based on choice probababilities
        k[t] = np.random.choice(range(K), p=p_k)
           
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
            #update_log_mean = (ML_alpha_mean*baseline_reward*dif*ML) / ((ML_std)**2) #*ML in nominator ensures update in logspace
            update_ML_mean = (ML_alpha_mean*baseline_reward*dif) / ((ML_std)**2) #*ML in nominator ensures update in logspace
        if ML < ML_mean:
            #update_log_mean = (ML_alpha_mean*baseline_reward*dif*ML) / ((ML_std)**2) #*ML in nominator ensures update in logspace
            update_ML_mean = (ML_alpha_mean*baseline_reward*dif) / ((ML_std)**2) #*ML in nominator ensures update in logspace
            
        #log_ML_mean = log_ML_mean + update_log_mean
        #ML_mean = np.exp(log_ML_mean)
        ML_mean = ML_mean + update_ML_mean
        ML_mean = min(ML_mean, 40)
     
        ##################
        #dif2 = ((dif)**2)-((ML_std)**2)
        #update_log_std =  (ML_alpha_std*baseline_reward*dif2) /((ML_std)**2) #ML_std^2 in denominator ensures update in logspace (instead of ML_std^3)
        
        #log_ML_std = log_ML_std + update_log_std 
        #ML_std = np.exp(log_ML_std)
     
        ML = np.random.normal(loc=ML_mean, scale=ML_std)
        #ML = min(ML, 40)
    return k, r, Q_k_stored, temp_sta, temp_sta_mean, temp_sta_std

def simulate_RW_MLtemp_volatile(Q_alpha, ML_alpha, T, Q_int, temp_mean_int, temp_std_int, reward_prob, rot, scale):
    Ma = 9.99
    Mi = 0.01
    K=len(reward_prob) #the amount of choice options
    #alpha          --->        learning rate
    #ML_alpha       --->        learning rate for meta-learning parameter (list of two LR's: one for mean ML and one for std ML)
    #T              --->        amount of trials for each simulation
    #K              --->        amount of choice options
    #Q_int          --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #temp_mean_int   --->        initial value for the mean of temp
    #temp_std_int    --->        initial value for the standard deviation of temp
    #reward prob    --->        probabilites to recieve a reward, associated to each option
    #rot            --->        amount of trials after which mean reward values rotate among choice options
    #scale          --->        scale with which ML-parameter gets transformed to temperature in a logit transformation
   
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on temperature
    r = np.zeros((T), dtype=float) #vector of rewards
    av_reward = 0.5 #starting baseline reward

    temp_vol = np.zeros((T), dtype=float)
    temp_vol_mean = np.zeros((T), dtype=float)
    temp_vol_std = np.zeros((T), dtype=float)

    ML_alpha_mean, ML_alpha_std = ML_alpha
    
    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    #initial values of the meta-learning (ML) parameter
    ML_mean_int = np.log(((temp_mean_int-Mi)/Ma)/(1-((temp_mean_int-Mi)/Ma)))
    ##################
    #ML_std_int = np.log(((temp_std_int-Mi)/Ma)/(1-((temp_std_int-Mi)/Ma)))
    ML_std_int = temp_std_int
    ML_int = np.random.normal(loc = ML_mean_int, scale=ML_std_int)

    ML_mean = ML_mean_int
    ML_std = ML_std_int
    ML = ML_int   

    log_ML_mean = np.log(ML_mean) #the parameter that gets updated at the end of the loop
    ##################
    #log_ML_std = np.log(ML_std) #the parameter that gets updated at the end of the loop

    for t in range(T):
        #temperature is calculated with a logit transformation of the ML
        temp = Ma*(np.exp(scale*ML)/(1+np.exp(scale*ML))) + Mi
        temp_mean = Ma*(np.exp(scale*ML_mean)/(1+np.exp(scale*ML_mean))) + Mi
        ##################
        #temp_std = Ma*(np.exp(scale*ML_std)/(1+np.exp(scale*ML_std))) + Mi
        temp_std = ML_std

        temp_vol[t] = temp
        temp_vol_mean[t] = temp_mean
        temp_vol_std[t] = temp_std
        # store values for Q
        Q_k_stored[t,:] = Q_k
      
        # make choice based on choice probababilities
        sum_soft = 0
        inv_temp = 1/temp

        for j in range(K):
            soft = np.exp(inv_temp*Q_k[j]) #using inverse temperature
            sum_soft = sum_soft + soft
        
        p_k = np.zeros(K) #vector to store probabilities for each choice
        for j in range(K):
            p_k[j] = np.exp(inv_temp*Q_k[j]) / sum_soft
      
        # make noisy choice based on choice probababilities
        k[t] = np.random.choice(range(K), p=p_k)      

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
            #update_log_mean = (ML_alpha_mean*baseline_reward*dif*ML) / ((ML_std)**2) #*ML in nominator ensures update in logspace
            update_ML_mean = (ML_alpha_mean*baseline_reward*dif) / ((ML_std)**2) #*ML in nominator ensures update in logspace
        if ML < ML_mean:
            #update_log_mean = (ML_alpha_mean*baseline_reward*dif*ML) / ((ML_std)**2) #*ML in nominator ensures update in logspace
            update_ML_mean = (ML_alpha_mean*baseline_reward*dif) / ((ML_std)**2) #*ML in nominator ensures update in logspace
            
        #log_ML_mean = log_ML_mean + update_log_mean
        #ML_mean = np.exp(log_ML_mean)
        ML_mean = ML_mean + update_ML_mean
        ML_mean = min(ML_mean, 40)

     
        ##################
        #dif2 = ((dif)**2)-((ML_std)**2)
        #update_log_std =  (ML_alpha_std*baseline_reward*dif2) /((ML_std)**2) #ML_std^2 in denominator ensures update in logspace (instead of ML_std^3)
        
        #log_ML_std = log_ML_std + update_log_std 
        #ML_std = np.exp(log_ML_std)
     
        ML = np.random.normal(loc=ML_mean, scale=ML_std)
        #ML = min(ML, 40)
    return k, r, Q_k_stored, temp_vol, temp_vol_mean, temp_vol_std

sim_nr = 3
threshold = 0.5
percentage=50
T=5000
Q_int = 1
Q_alpha = 0.5
amount_of_sim = 100
reward_prob = [0.2,0.2,0.8,0.2,0.2,0.2,0.2,0.2,0.8,0.2]
rot=10
K=10
ML_alpha = [0.05, 0] #LR std, LR pos and LR neg

window = 50

temp_mean_int = 0.5
temp_std_int = 0.01

#################################################################
#simulations
#################################################################


total_temp_var = np.zeros(amount_of_sim)
total_temp_var_mean = np.zeros(amount_of_sim)
total_temp_var_std = np.zeros(amount_of_sim)
reward_var = np.zeros(amount_of_sim)
for sim in range(amount_of_sim):
    k, r, Q_k_stored, temp_var, temp_var_mean, temp_var_std = simulate_RW_MLtemp_variable(Q_alpha=Q_alpha, ML_alpha=ML_alpha, T=T, Q_int=Q_int, temp_mean_int=temp_mean_int, temp_std_int=temp_std_int, reward_prob=reward_prob, scale=1, threshold = threshold)
    #mean reward over time (cumulative reward divided by current trial)
    total_temp_var[sim] = np.mean(temp_var)
    total_temp_var_mean[sim] = np.mean(temp_var_mean)
    total_temp_var_std[sim] = np.mean(temp_var_std)
    reward_var[sim] = np.mean(r)

av_temp_var = np.mean(total_temp_var)
av_temp_var_mean = np.mean(total_temp_var_mean)
av_temp_var_std = np.mean(total_temp_var_std)
av_reward_var = np.mean(reward_var)

std_temp_var = np.std(total_temp_var)
std_temp_var_mean = np.std(total_temp_var_mean)
std_temp_var_std = np.std(total_temp_var_std)
std_reward_var = np.std(reward_var)




total_temp_sta = np.zeros(amount_of_sim)
total_temp_sta_mean = np.zeros(amount_of_sim)
total_temp_sta_std = np.zeros(amount_of_sim)
reward_sta = np.zeros(amount_of_sim)
for sim in range(amount_of_sim):
    k, r, Q_k_stored, temp_sta, temp_sta_mean, temp_sta_std = simulate_RW_MLtemp_stable(Q_alpha=Q_alpha, ML_alpha=ML_alpha, T=T, Q_int=Q_int, temp_mean_int=temp_mean_int, temp_std_int=temp_std_int, reward_prob=reward_prob, scale=1)
    #mean reward over time (cumulative reward divided by current trial)
    total_temp_sta[sim] = np.mean(temp_sta)
    total_temp_sta_mean[sim] = np.mean(temp_sta_mean)
    total_temp_sta_std[sim] = np.mean(temp_sta_std)
    reward_sta[sim] = np.mean(r)

av_temp_sta = np.mean(total_temp_sta)
av_temp_sta_mean = np.mean(total_temp_sta_mean)
av_temp_sta_std = np.mean(total_temp_sta_std)
av_reward_sta = np.mean(reward_sta)

std_temp_sta = np.std(total_temp_sta)
std_temp_sta_mean = np.std(total_temp_sta_mean)
std_temp_sta_std = np.std(total_temp_sta_std)
std_reward_sta = np.std(reward_sta)




total_temp_vol = np.zeros(amount_of_sim)
total_temp_vol_mean = np.zeros(amount_of_sim)
total_temp_vol_std = np.zeros(amount_of_sim)
reward_vol = np.zeros(amount_of_sim)
for sim in range(amount_of_sim):
    k, r, Q_k_stored, temp_vol, temp_vol_mean, temp_vol_std = simulate_RW_MLtemp_volatile(Q_alpha=Q_alpha, ML_alpha=ML_alpha, T=T, Q_int=Q_int, temp_mean_int=temp_mean_int, temp_std_int=temp_std_int, reward_prob=reward_prob, rot=rot, scale=1)
    #mean reward over time (cumulative reward divided by current trial)
    total_temp_vol[sim] = np.mean(temp_vol)
    total_temp_vol_mean[sim] = np.mean(temp_vol_mean)
    total_temp_vol_std[sim] = np.mean(temp_vol_std)
    reward_vol[sim] = np.mean(r)

av_temp_vol = np.mean(total_temp_vol)
av_temp_vol_mean = np.mean(total_temp_vol_mean)
av_temp_vol_std = np.mean(total_temp_vol_std)
av_reward_vol = np.mean(reward_vol)

std_temp_vol = np.std(total_temp_vol)
std_temp_vol_mean = np.std(total_temp_vol_mean)
std_temp_vol_std = np.std(total_temp_vol_std)
std_reward_vol = np.std(reward_vol)






save_dir = '/Users/jareynde/OneDrive - UGent/1A_Main_Project/Models/Simulations/Random/Temperature/Output/logit'

title1 = f'average sampled temperature over {amount_of_sim} simulations, Q-value learning rate of {Q_alpha}\naverage gained rewards are for stable {round(av_reward_sta, 2)} +/- {round(std_reward_sta,2)}, for volatile {round(av_reward_vol,2)} +/- {round(std_reward_vol,2)} and for variable {round(av_reward_var,2)} +/- {round(std_reward_var,2)}'
fig_name = os.path.join(save_dir, f'{sim_nr}_temperature_threshold{percentage}')
fig, ax= plt.subplots(figsize=(10, 7))
ax.bar(['stable context', 'volatile context', 'variable context'], [av_temp_sta, av_temp_vol, av_temp_var], yerr=[std_temp_sta, std_temp_vol, std_temp_var])
ax.set_ylabel('temperature')
plt.title(title1)
plt.savefig(fig_name)
plt.show()

title2 = f'average updated mean temperature over {amount_of_sim} simulations, Q-value learning rate of {Q_alpha}\naverage gained rewards are for stable {round(av_reward_sta, 2)} +/- {round(std_reward_sta,2)}, for volatile {round(av_reward_vol,2)} +/- {round(std_reward_vol,2)} and for variable {round(av_reward_var,2)} +/- {round(std_reward_var,2)}'
fig_name = os.path.join(save_dir, f'{sim_nr}_mean_temperature_threshold{percentage}')
fig, ax= plt.subplots(figsize=(10, 7))
ax.bar(['stable context', 'volatile context', 'variable context'], [av_temp_sta_mean, av_temp_vol_mean, av_temp_var_mean], yerr=[std_temp_sta_mean, std_temp_vol_mean, std_temp_var_mean])
ax.set_ylabel('temperature')
plt.title(title2) 
plt.savefig(fig_name)
plt.show()