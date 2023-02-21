#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rescorla Wagner model with policy gradient (PG) at second level (for learning epsilon parameter)
Formula's in PG come from:
https://towardsdatascience.com/a-minimal-working-example-for-continuous-policy-gradients-in-tensorflow-2-0-d3413ec38c6b
rewards zijn hier gelijk aan 0 of 1 obv probabiliteiten
Dit script is gelijkaardig aan 'sim_MLeps_RW_attempt2.py'
Volgende aanpassingen worden vergeleken tov dit script:
*epsilon update wordt geschaald via logit transformatie naar -oneindig, +oneindig
*reward wordt gebaselined
*een keuzeoptie kan niet meer dezelfde reward prob krijgen bij een shuffle (iets wat eerst volgens law of chance in principe wel kon, en ook gebeurde soms)
    
@author: Janne Reynders; janne.reynders@ugent.be
"""
import numpy as np                  
import pandas as pd                 
import matplotlib.pyplot as plt     
import os
import random

#np.random.seed(2022)                # put seed for reproducibility

#simulate a Rescorla Wagner model with meta-learning of epsilon in discrete amounts
#this function is the same as in 'sim_MLeps_RW_attempt2.py' but with a baseline adaptation of the reward and a rescaling of epsilon
#Policy gradient update happens in logspace --> sigma gets updated too
#No constant sigma as in previous models

#simulate a Rescorla Wagner model with constant epsilon
def simulate_RW(alpha, eps, T, Q_int, reward_prob, rot1, rot2):
    K=len(reward_prob)
    #alpha      --->        learning rate
    #eps        --->        epsilon
    #T          --->        amount of trials for each simulation
    #K          --->        amount of choice options
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #rot        --->        amount of trials after which mean reward values rotate among choice options

    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards

    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    
    
    for t in range(T):
        
        # store values for Q
        Q_k_stored[t,:] = Q_k   
    
        # compute choice probabilities with Softmax function for K choice options
        sum_soft = 0

        for j in range(K):
            soft = Q_k[j]
            sum_soft = sum_soft + soft
                
        p_k = np.zeros(K) #vector to store probabilities for each choice
        for j in range(K):
            p_k[j] = Q_k[j] / sum_soft
              
        # make choice based on choice probababilities
        rand[t] = np.random.choice([0,1], p=[1-eps,eps])
        if rand[t] == 0:
            k[t] = np.argmax(p_k)
        if rand[t] == 1:
            k[t] = np.random.choice(range(K))
       
        rott = rot1
        if t>1000:
            rott = rot2
        if t>2000:
            rott = rot1
        if t>3000:
            rott = rot2
        if t>4000:
            rott = rot1
        if t>5000:
            rott = rot2
        if t>6000:
            rott = rot1
        if t>7000:
            rott = rot2
        if t>8000:
            rott = rot1
        if t>9000:
            rott = rot2
        
        # generate reward based on normal distribution linked to choice made
        if (t%rott)==0 and t != 0:
            reward_orig = reward_prob.copy()
            while reward_orig == reward_prob:
                np.random.shuffle(reward_prob)
        
        a1 = reward_prob[k[t]]
        a0 = 1-a1
        r[t] = np.random.choice([1, 0], p=[a1, a0])

        # update values
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + alpha * delta_k
        

    return k, r, Q_k_stored

def simulate_RW_MLT_adapted_log_random(Q_alpha, ML_alpha, T, Q_int, eps_mean_int, eps_std_int, reward_prob, scale, threshold):
    K=len(reward_prob) #the amount of choice options
    Freq = np.zeros((K), dtype = float)
    #alpha          --->        learning rate
    #alpha_eps      --->        learning rate for epsilon
    #T              --->        amount of trials for each simulation
    #K              --->        amount of choice options
    #Q_int          --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #eps_mean_int   --->        initial value for the mean of eps
    #eps_std_int    --->        initial value for the standard deviation of eps
    #rot            --->        amount of trials after which mean reward values rotate among choice options
   
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on epsilon
    r = np.zeros((T), dtype=float) #vector of rewards
    ML_stored = np.zeros((T), dtype=float)
    eps_stored = np.zeros((T), dtype=float)
    ML_mean_stored = np.zeros((T), dtype=float)
    ML_std_stored = np.zeros((T), dtype=float)
    av_reward = 0.5 #starting baseline reward
    eps_random = np.zeros((T), dtype=float)
    eps_mean_random = np.zeros((T), dtype=float)
   

    ML_alpha_pos, ML_alpha_neg, ML_alpha_std = ML_alpha
    
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
        eps_random[t] = eps
        eps_mean_random[t] = eps_mean
        # store values for Q and epsilon
        Q_k_stored[t,:] = Q_k
        ML_stored[t] = ML
        eps_stored[t] = eps
        ML_mean_stored[t] = ML_mean
        ML_std_stored[t] = ML_std

        # compute choice probabilities with Softmax function for K choice options      
        sum_soft = 0
        for j in range(K):
            soft = Q_k[j]
            sum_soft = sum_soft + soft
        
        p_k = np.zeros(K) #vector to store probabilities for each choice
        for j in range(K):
            p_k[j] = Q_k[j] / sum_soft
      
        # make choice based on choice probababilities
        rand[t] = np.random.choice(2, p=[1-eps,eps])
        if rand[t] == 0:
         k[t] = np.argmax(p_k)
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
            update_ML_mean = (ML_alpha_pos*baseline_reward*dif) / ((ML_std)**2) #*ML in nominator ensures update in logspace
        if ML < ML_mean:
            #update_log_mean = (ML_alpha_neg*baseline_reward*dif*ML) / ((ML_std)**2) #*ML in nominator ensures update in logspace
            update_ML_mean = (ML_alpha_neg*baseline_reward*dif) / ((ML_std)**2) #*ML in nominator ensures update in logspace
            
        #log_ML_mean = log_ML_mean + update_log_mean
        #ML_mean = np.exp(log_ML_mean)
        ML_mean = ML_mean + update_ML_mean
     
        dif2 = ((dif)**2)-((ML_std)**2)
        update_log_std =  (ML_alpha_std*baseline_reward*dif2) /((ML_std)**2) #ML_std^2 in denominator ensures update in logspace (instead of ML_std^3)
        
        log_ML_std = log_ML_std + update_log_std 
        ML_std = np.exp(log_ML_std)
     
        ML = np.random.normal(loc=ML_mean, scale=ML_std)
  
    return k, eps_stored, ML_stored, ML_mean_stored, ML_std_stored, r, Q_k_stored, eps_random, eps_mean_random

def simulate_RW_MLT_adapted_log_stable(Q_alpha, ML_alpha, T, Q_int, eps_mean_int, eps_std_int, reward_prob, scale):
    K=len(reward_prob) #the amount of choice options
    #alpha          --->        learning rate
    #alpha_eps      --->        learning rate for epsilon
    #T              --->        amount of trials for each simulation
    #K              --->        amount of choice options
    #Q_int          --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #eps_mean_int   --->        initial value for the mean of eps
    #eps_std_int    --->        initial value for the standard deviation of eps
    #rot            --->        amount of trials after which mean reward values rotate among choice options
   
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on epsilon
    r = np.zeros((T), dtype=float) #vector of rewards
    ML_stored = np.zeros((T), dtype=float)
    eps_stored = np.zeros((T), dtype=float)
    ML_mean_stored = np.zeros((T), dtype=float)
    ML_std_stored = np.zeros((T), dtype=float)
    av_reward = 0.5 #starting baseline reward
    eps_stable = np.zeros((T), dtype=float)
    eps_mean_stable = np.zeros((T), dtype=float)
    

    ML_alpha_pos, ML_alpha_neg, ML_alpha_std = ML_alpha
    
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
        eps_stable[t] = eps
        eps_mean_stable[t] = eps_mean

        # store values for Q and epsilon
        Q_k_stored[t,:] = Q_k
        ML_stored[t] = ML
        eps_stored[t] = eps
        ML_mean_stored[t] = ML_mean
        ML_std_stored[t] = ML_std

        # compute choice probabilities with Softmax function for K choice options      
        sum_soft = 0
        for j in range(K):
            soft = Q_k[j]
            sum_soft = sum_soft + soft
        
        p_k = np.zeros(K) #vector to store probabilities for each choice
        for j in range(K):
            p_k[j] = Q_k[j] / sum_soft
      
        # make choice based on choice probababilities
        rand[t] = np.random.choice(2, p=[1-eps,eps])
        if rand[t] == 0:
         k[t] = np.argmax(p_k)
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
            update_ML_mean = (ML_alpha_pos*baseline_reward*dif) / ((ML_std)**2) #*ML in nominator ensures update in logspace
        if ML < ML_mean:
            #update_log_mean = (ML_alpha_neg*baseline_reward*dif*ML) / ((ML_std)**2) #*ML in nominator ensures update in logspace
            update_ML_mean = (ML_alpha_neg*baseline_reward*dif) / ((ML_std)**2) #*ML in nominator ensures update in logspace
            
        #log_ML_mean = log_ML_mean + update_log_mean
        #ML_mean = np.exp(log_ML_mean)
        ML_mean = ML_mean + update_ML_mean
     
        dif2 = ((dif)**2)-((ML_std)**2)
        update_log_std =  (ML_alpha_std*baseline_reward*dif2) /((ML_std)**2) #ML_std^2 in denominator ensures update in logspace (instead of ML_std^3)
        
        log_ML_std = log_ML_std + update_log_std 
        ML_std = np.exp(log_ML_std)
     
        ML = np.random.normal(loc=ML_mean, scale=ML_std)
  
    return k, eps_stored, ML_stored, ML_mean_stored, ML_std_stored, r, Q_k_stored, eps_stable, eps_mean_stable

def simulate_RW_MLT_adapted_log_volatile(Q_alpha, ML_alpha, T, Q_int, eps_mean_int, eps_std_int, reward_prob, rot, scale):
    K=len(reward_prob) #the amount of choice options
    #alpha          --->        learning rate
    #alpha_eps      --->        learning rate for epsilon
    #T              --->        amount of trials for each simulation
    #K              --->        amount of choice options
    #Q_int          --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #eps_mean_int   --->        initial value for the mean of eps
    #eps_std_int    --->        initial value for the standard deviation of eps
    #rot            --->        amount of trials after which mean reward values rotate among choice options
   
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on epsilon
    r = np.zeros((T), dtype=float) #vector of rewards
    ML_stored = np.zeros((T), dtype=float)
    eps_stored = np.zeros((T), dtype=float)
    ML_mean_stored = np.zeros((T), dtype=float)
    ML_std_stored = np.zeros((T), dtype=float)
    av_reward = 0.5 #starting baseline reward
    eps_volatile = np.zeros((T), dtype=float)
    eps_mean_volatile = np.zeros((T), dtype=float)

    ML_alpha_pos, ML_alpha_neg, ML_alpha_std = ML_alpha
    
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
        eps_volatile[t] = eps
        eps_mean_volatile[t] = eps_mean
        # store values for Q and epsilon
        Q_k_stored[t,:] = Q_k
        ML_stored[t] = ML
        eps_stored[t] = eps
        ML_mean_stored[t] = ML_mean
        ML_std_stored[t] = ML_std

        # compute choice probabilities with Softmax function for K choice options      
        sum_soft = 0
        for j in range(K):
            soft = Q_k[j]
            sum_soft = sum_soft + soft
        
        p_k = np.zeros(K) #vector to store probabilities for each choice
        for j in range(K):
            p_k[j] = Q_k[j] / sum_soft
      
        # make choice based on choice probababilities
        rand[t] = np.random.choice(2, p=[1-eps,eps])
        if rand[t] == 0:
         k[t] = np.argmax(p_k)
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
            update_ML_mean = (ML_alpha_pos*baseline_reward*dif) / ((ML_std)**2) #*ML in nominator ensures update in logspace
        if ML < ML_mean:
            #update_log_mean = (ML_alpha_neg*baseline_reward*dif*ML) / ((ML_std)**2) #*ML in nominator ensures update in logspace
            update_ML_mean = (ML_alpha_neg*baseline_reward*dif) / ((ML_std)**2) #*ML in nominator ensures update in logspace
            
        #log_ML_mean = log_ML_mean + update_log_mean
        #ML_mean = np.exp(log_ML_mean)
        ML_mean = ML_mean + update_ML_mean
     
        dif2 = ((dif)**2)-((ML_std)**2)
        update_log_std =  (ML_alpha_std*baseline_reward*dif2) /((ML_std)**2) #ML_std^2 in denominator ensures update in logspace (instead of ML_std^3)
        
        log_ML_std = log_ML_std + update_log_std 
        ML_std = np.exp(log_ML_std)
     
        ML = np.random.normal(loc=ML_mean, scale=ML_std)
  
    return k, eps_stored, ML_stored, ML_mean_stored, ML_std_stored, r, Q_k_stored, eps_volatile, eps_mean_volatile

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
eps_std_int = 0.525

#################################################################
#simulations
#################################################################


total_eps_random = np.zeros(amount_of_sim)
total_eps_mean_random = np.zeros(amount_of_sim)
reward_random = np.zeros(amount_of_sim)
for sim in range(amount_of_sim):
    k, eps_stored, ML_stored, ML_mean_stored, ML_std_stored, r, Q_k_stored, eps_random, eps_mean_random = simulate_RW_MLT_adapted_log_random(Q_alpha=Q_alpha, ML_alpha=ML_alpha, T=T, Q_int=Q_int, eps_mean_int=eps_mean_int, eps_std_int=eps_std_int, reward_prob=reward_prob, scale=1, threshold = threshold)
    #mean reward over time (cumulative reward divided by current trial)
    eps_av = np.mean(eps_random)
    eps_mean_av = np.mean(eps_mean_random)
    reward_av = np.mean(r)

    total_eps_random[sim] = eps_av
    total_eps_mean_random[sim] = eps_mean_av
    reward_random[sim] = reward_av

eps_random_done = np.mean(total_eps_random)
eps_mean_random_done = np.mean(total_eps_mean_random)
reward_random_done = np.mean(reward_random)

eps_random_std = np.std(total_eps_random)
eps_mean_random_std = np.std(total_eps_mean_random)
reward_random_std = np.std(reward_random)



total_eps_stable = np.zeros(amount_of_sim)
total_eps_mean_stable = np.zeros(amount_of_sim)
reward_stable = np.zeros(amount_of_sim)
for sim in range(amount_of_sim):
    k, eps_stored, ML_stored, ML_mean_stored, ML_std_stored, r, Q_k_stored, eps_stable, eps_mean_stable = simulate_RW_MLT_adapted_log_stable(Q_alpha=Q_alpha, ML_alpha=ML_alpha, T=T, Q_int=Q_int, eps_mean_int=eps_mean_int, eps_std_int=eps_std_int, reward_prob=reward_prob, scale=1)
    #mean reward over time (cumulative reward divided by current trial)
    eps_av = np.mean(eps_stable)
    eps_mean_av = np.mean(eps_mean_stable)
    reward_av = np.mean(r)

    total_eps_stable[sim] = eps_av
    total_eps_mean_stable[sim] = eps_mean_av
    reward_stable[sim] = reward_av

eps_stable_done = np.mean(total_eps_stable)
eps_mean_stable_done = np.mean(total_eps_mean_stable)
reward_stable_done = np.mean(reward_stable)

eps_stable_std = np.std(total_eps_stable)
eps_mean_stable_std = np.std(total_eps_mean_stable)
reward_stable_std = np.std(reward_stable)





total_eps_volatile = np.zeros(amount_of_sim)
total_eps_mean_volatile = np.zeros(amount_of_sim)
reward_volatile = np.zeros(amount_of_sim)
for sim in range(amount_of_sim):
    k, eps_stored, ML_stored, ML_mean_stored, ML_std_stored, r, Q_k_stored, eps_volatile, eps_mean_volatile = simulate_RW_MLT_adapted_log_volatile(Q_alpha=Q_alpha, ML_alpha=ML_alpha, T=T, Q_int=Q_int, eps_mean_int=eps_mean_int, eps_std_int=eps_std_int, reward_prob=reward_prob, rot=rot, scale=1)
    #mean reward over time (cumulative reward divided by current trial)
    eps_av = np.mean(eps_volatile)
    eps_mean_av = np.mean(eps_mean_volatile)
    reward_av = np.mean(r)

    total_eps_volatile[sim] = eps_av
    total_eps_mean_volatile[sim] = eps_mean_av
    reward_volatile[sim] = reward_av

eps_volatile_done = np.mean(total_eps_volatile)
eps_mean_volatile_done = np.mean(total_eps_mean_volatile)
reward_volatile_done = np.mean(reward_volatile)

eps_volatile_std = np.std(total_eps_volatile)
eps_mean_volatile_std = np.std(total_eps_mean_volatile)
reward_volatile_std = np.std(reward_volatile)





save_dir = '/Users/jareynde/OneDrive - UGent/1A_Main_Project/Data/Neuringer/Self/Output/MLeps_policy_RW_model/Compare_eps_stable_volatile_random'

title1 = f'average sampled epsilon over {amount_of_sim} simulations, Q-value learning rate of {Q_alpha}\naverage gained rewards are for stable {round(reward_stable_done, 2)} +/- {round(reward_stable_std,2)}, for volatile {round(reward_volatile_done,2)} +/- {round(reward_volatile_std,2)} and for variable {round(reward_random_done,2)} +/- {round(reward_random_std,2)}'
fig_name = os.path.join(save_dir, f'{sim_nr}_average epsilon_threshold{percentage}')
fig, ax= plt.subplots(figsize=(10, 7))
ax.bar(['stable context', 'volatile context', 'variable context'], [eps_stable_done, eps_volatile_done, eps_random_done], yerr=[eps_stable_std, eps_volatile_std, eps_random_std])
ax.set_ylabel('epsilon')
plt.title(title1)
plt.savefig(fig_name)
plt.show()

title2 = f'average updated mean epsilon over {amount_of_sim} simulations, Q-value learning rate of {Q_alpha}\naverage gained rewards are for stable {round(reward_stable_done,2)} +/- {round(reward_stable_std,2)}, for volatile {round(reward_volatile_done,2)} +/- {round(reward_volatile_std,2)} and for variable {round(reward_random_done,2)} +/- {round(reward_random_std,2)}'
fig_name = os.path.join(save_dir, f'{sim_nr}_average mean epsilon_threshold{percentage}')
fig, ax= plt.subplots(figsize=(10, 7))
ax.bar(['stable context', 'volatile context', 'variable context'], [eps_mean_stable_done, eps_mean_volatile_done, eps_mean_random_done], yerr=[eps_mean_stable_std, eps_mean_volatile_std, eps_mean_random_std])
ax.set_ylabel('epsilon')
plt.title(title2) 
plt.savefig(fig_name)
plt.show()
