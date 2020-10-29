#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# N is the total population. We are assuming it to be a random constant.
N = 1000

# I represents the infected population and R represents the removed (death or recovered) 
# Initial number of infected and recovered individuals, I0 and R0.
# We are considering the population starts with one infected individual and zero recovered individual. 
I0, R0 = 1, 0

# The rest of the population (Except the infected and removed) are considered to be susceptible to infection.
# S represents the susceptible population, where S0 is the initally susceptible individual.
S0 = N - I0 - R0

# beta represents the contact transmission rate and gamma represents the recovery rate.
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta, gamma = 0.2, 1./10

# Here the total time period is taken for 160 day.
# A grid of time points (in days)
t = np.linspace(0, 160, 160)

# The differential equations of SIR model.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Forming the equations using the initial conditions extending it for the considered time period.
# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

#Ploting the values to indicate the change in population in over different categories and how change in one affects the other.
# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='y')
ax = fig.add_subplot(111, facecolor='#aaaaaa', axisbelow=True)
ax.plot(t, S/1000, 'r', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/1000, 'g', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/1000, 'b', alpha=0.5, lw=2, label='Removed - Recovered/Death')
ax.set_xlabel('Time in days')
ax.set_ylabel('Number in Thousands')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()


# In[49]:


plt.stem(t, I/1000, linefmt=None, markerfmt=None, basefmt=None)


# In[50]:


print(I/1000)


# In[51]:


def quantize(signal, partitions, codebook):
    indices = []
    quanta = []
    for datum in signal:
        index = 0
        while (index < len(partitions) and datum > partitions[index]):
            index += 1
        indices.append(index)
        quanta.append(codebook[index])
    return indices, quanta


xmax=2
xmin=-1
n=3
L=2^n
avg=(xmax-xmin)/L
parti = np.linspace(xmin, avg, xmax)
codebook=np.linspace(xmin-(avg/2), avg, xmax+avg/2)

index, quants = quantize(I/1000, parti, codebook)
print(index)
print(quants)


# In[52]:


plt.stem(t, quants, linefmt=None, markerfmt=None, basefmt=None)

