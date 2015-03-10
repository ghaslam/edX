
# coding: utf-8

## **AM 207**: Homework 4

# _ _ _ _ _
# 
# Verena Kaynig-Fittkau & Pavlos Protopapas <br>
# Handed out: Thursday, February 26th, 2015<br>
# Due: 11.59 P.M. Wednesday March 4th, 2015
# 
# **Instructions**:
# 
# + Upload your answers in an ipython notebook to the dropbox.
# 
# + The notebook filename should have the format AM207_YOURNAME_HW4.ipynb
# 
# + Your code should be in code cells of your ipython notebook. Do not use other languages or formats without permission from the TFs. Any special libraries should be included with your code - the notebook should run "as is". 
# 
# + If you have multiple files (e.g. you've added code files and images) create a tarball or zip for all files in a single file and name it: AM207_YOURNAME_HW4.tar or AM207_YOURNAME_HW4.zip
# 
# + Please remember that your solution is supposed to be a report. Provide clear explanations of your ideas and the way you  approached the solution. Clearly identify the question or part of a question that you are answering. And please comment your code!
# _ _ _ _ _

# In[2]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
import seaborn as sns

from scipy.stats import distributions
from scipy.stats import norm
from pylab import mgrid, contourf, cm, show


#### Question 1: Jumping between mountains

# We use the M-H algorithm to sample from the normal distribution: $p(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}$. 
# 
# 
# 
# (a) Select a proposal distribution of $q(x^*|x) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x^*-x)^2}{2\sigma^2}}$. Let the starting point be $x_0=-1000$ and step size be $\sigma=1$. Now, run the M-H algorithm for 10,000 samples. Show the trace plot of your algorithm. What is an appropriate burn-in sample size?
# 
# (b) Perform Geweke tests, one for all samples and one for samples after burn-in. Do the results differ? If so, explain why.
# 
# Now, we would like to sample from a mixture normal distribution:  $p(x) = \frac{1}{2\sqrt{2\pi}} e^{-\frac{(x-6)^2}{2}} + \frac{1}{2\sqrt{2\pi}} e^{-\frac{(x+6)^2}{2}}$. 
# 
# (c) Please visualize the density function, $p(x)$, from $-10$ to $10$.
# 
# (d) Using a starting point of $x_0=-6$ and a proposal distribution of $q(x^*|x) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x^*-x)^2}{2\sigma^2}}$ with $\sigma=1$, run the M-H algorithm for 10,000 samples.
# Choose an appropriate burn-in sample size, show the trace plot, and perform the Geweke test on the remaining samples. Just looking at these two results, what do you see? Does the MCMC converge?
# 
# (e) Now plot the histogram of the remaining samples. Is the M-H algorithm converging?
# 
# (f) Finally, using $\sigma=4$ for the proposal distribution, re-answer problems (d) and (e). For the Geweke test, try 
# segments of 100 and 500 to calculate the means and variances. If the results differ, please explain why. 

#### Solution to Q.1:

# (a) Select a proposal distribution of $q(x^*|x) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x^*-x)^2}{2\sigma^2}}$. Let the starting point be $x_0=-1000$ and step size be $\sigma=1$. Now, run the M-H algorithm for 10,000 samples. Show the trace plot of your algorithm. What is an appropriate burn-in sample size?

# I could try to use rejection sampling to produce values for $q(x^*|x)$ or else try to take advantage of the fact that the proposal distribution is very similar to the normal distribution. However, I don't really know what I'm doing so can't see how to actually implement that. Below I attempt to do rejection sampling based on $q(x^*|x)$ but I don't really know what the limits should be on my domain or the range limit for y. I'm sure there should also be some Normalisation constant to be bring the CDF to 1, but again, I don't really know.

# In[37]:

#make denominator a constant
denom = np.sqrt(2*np.pi)
sig = 1

#proposal function
q = lambda x,x_prev: (1/(denom*sig))*np.exp(-(x-x_prev)**2/(2*sig**2));

# domain limits
xmin = -2000 # the lower limit of our domain
xmax = 1000 # the upper limit of our domain

# range limit (supremum) for y
ymax = 1

#intitialize the sampling. From question - x0 = -1000 (a long way from function)
x0 = -1000
x_prev = x0

N = 1000 # the total of samples we wish to generate
accepted = 0 # the number of accepted samples
samples = np.zeros(N)
count = 0 # the total count of proposals

# generation loop
while (accepted < N):
    
    # pick a uniform number on [xmin, xmax) (e.g. 0...10)
    x = np.random.uniform(xmin, xmax)

    # pick a uniform number on [0, ymax)
    y = np.random.uniform(0,ymax)
    
    # Do the accept/reject comparison
    if y < q(x,x_prev):
        samples[accepted] = x
        #x_prev = x
        accepted += 1
        x_prev = x
    count +=1
    
print "no of attempts: ",count, "number accepted: ",accepted

# get the histogram info
hinfo = np.histogram(samples,30)

# plot the histogram
plt.hist(samples,bins=30, label=u'Samples', alpha=0.5);

# plot our (normalized) function
xvals=np.linspace(xmin, xmax, 1000)
#plt.plot(xvals, hinfo[0][0]*q(xvals,xvals), 'r', label=u'P(x)')

# turn on the legend
plt.legend()


# The samples are stuck around the origin. Will now try and see if I can use the MH to get them to shift towards x = 0. I will sample x_star randomly from my accepted samples and then generate a new set of samples??

# In[69]:


##--------Functions--------##
#target distribution p(x)
p = lambda x: (1/denom)*np.exp(-(x**2/2));

#or maybe I should be using the log form:
logp = lambda x: norm.logpdf(x);

#q = lambda x: (1/(denom*sig))*np.exp(-(x-x_prev)**2/(2*sig**2))

#intitialize the sampling. From question - x0 = -1000 (a long way from function)
x0 = -1000

x_prev = x0
n = 10000
accepted = np.zeros(n)
k=1
i=0
while i<n:
    
    #choose value from within the range of samples calculated before
    #x = np.random.choice(samples)
    
    
    #choose a proposal step for where to move to next
    #x_star = q(x, x_prev)
    x_star = np.random.normal(x_prev,sig)
    #print x_star
    #work out if it is a good idea to move:
    p_star = logp(x_star)
    p_prev = logp(x_prev)
    
    U= np.random.uniform()
    A = np.min((1,p_star - p_prev)) #acceptance probability. By changing to log dist. of p(x) then
                        #we can subtract and not divide.
    
    if np.log(U) < A: #accept proposal and set x_prev as the accepted value
        accepted[i] = x_star
        i = i + 1
        x_prev = x_star
    else: #reject proposal and stay where you are
        accepted[i] = x_prev
        x_prev = accepted[i]  
        i = i + 1
        k=k+1  
        
# =======
# PLOTTING
plt.subplot(1,2,1)
fig = plt.hist(accepted, 20, alpha=0.3)
#fig = plt.hist(accepted[300:], 20, alpha=0.3)

plt.subplot(1,2,2)
plt.plot(accepted)        


# So the plot converges to zero which is what we want. Looking at the trace plot, it seems that the burn-in is around 2500.

# (b) Perform Geweke tests, one for all samples and one for samples after burn-in. Do the results differ? If so, explain why.

# In[72]:

def Geweke(trace, intervals, length):
    # take two parts of the chain. 
    # subsample lenght 
    nsl=length
    first = 0.1*len(trace)
    z =np.empty(intervals)
    for k in np.arange(0, intervals):
        # beg of each sub samples
        bega=first+k*length
        begb = len(trace)/2 + k*length
        
        sub_trace_a = trace[bega:bega+nsl]
        sub_trace_b = trace[begb:begb+nsl]
        
        theta_a = np.mean(sub_trace_a)
        theta_b = np.mean(sub_trace_b)
        var_a  = np.var(sub_trace_a)
        var_b  = np.var(sub_trace_b)
        
        z[k] = (theta_a-theta_b)/np.sqrt( var_a + var_b)
        
    return z

#print trace.shape
#z = Geweke(trace[:,1], first=2000, length=300)

#plt.plot( [2]*len(z), 'r-.')
#plt.plot(z)
#plt.plot( [-2]*len(z), 'r-.')

burnin = 2500

geweke1 = Geweke(accepted[:], 10, 30)
geweke2 = Geweke(accepted[burnin:], 10, 30)
plt.figure(1, figsize=(14,6))
plt.subplot(121)
plt.plot(geweke1)
plt.subplot(122)
plt.plot(geweke2)
plt.show()


# When including all the samples, many of which have not yet converged, shown in the first plot, the plot varies hugely outside the range (-2, 2) but after taking a burn-in of 2500 and replotting, the trace stays comfortably between (-2, 2), indicating that the trace has converged.

# (c) Please visualize the density function, $p(x)$, from $-10$ to $10$. $p(x) = \frac{1}{2\sqrt{2\pi}} e^{-\frac{(x-6)^2}{2}} + \frac{1}{2\sqrt{2\pi}} e^{-\frac{(x+6)^2}{2}}$.

# In[97]:

#set the range -10 to 10
asp = np.linspace(-10,10, 100)
p = lambda x: 1/(2*np.sqrt(2*np.pi))*np.exp(-(x-6)**2/2) + 1/(2*np.sqrt(2*np.pi))*np.exp(-(x+6)**2/2)
P_x = p(asp)

plt.plot(asp, P_x)
plt.xlabel('x')
plt.title('density function, $p(x)$')


# (d) Using a starting point of $x_0=-6$ and a proposal distribution of $q(x^*|x) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x^*-x)^2}{2\sigma^2}}$ with $\sigma=1$, run the M-H algorithm for 10,000 samples.
# Choose an appropriate burn-in sample size, show the trace plot, and perform the Geweke test on the remaining samples. Just looking at these two results, what do you see? Does the MCMC converge?

# In[106]:

#same code as before (I know I should have made this into a function) but with x0 = -6
x0 = -6.0
x_prev = x0

n = 10000
accepted = np.zeros(n)
k=1
i=0

while i<n:
    
    #choose a proposal step for where to move to next
    #x_star = q(x, x_prev)
    x_star = np.random.normal(x_prev,sig)


    #work out if it is a good idea to move:
    p_star = p(x_star)
    p_prev = p(x_prev)
    p_star = np.log(p_star)
    p_prev = np.log(p_prev)
    
    U= np.random.uniform()
    A = np.min((1,p_star - p_prev)) #acceptance probability. By changing to log dist. of p(x) then
                        #we can subtract and not divide.
    
    if np.log(U) < A: #accept proposal and set x_prev as the accepted value
        accepted[i] = x_star
        i = i + 1
        x_prev = x_star
    else: #reject proposal and stay where you are
        accepted[i] = x_prev
        x_prev = accepted[i]  
        i = i + 1
        k=k+1  
        
# =======
# PLOTTING
burnin = 100

plt.figure(1, figsize=(14,14))
plt.subplot(3,2,1)
fig = plt.hist(accepted[:burnin], 20, alpha=0.3)
plt.tight_layout()
#fig = plt.hist(accepted[300:], 20, alpha=0.3)

plt.subplot(3,2,2)
plt.plot(accepted[:burnin]) 

#Geweke Test


geweke2 = Geweke(accepted[burnin:], 10, 30)
plt.subplot(3,2,3)
plt.plot(geweke2)
plt.show()


# The plot seems to converge almost immediately, this is because -6 as an initial starting point is at the centre of the first mode of p(x) so we expect this to be a very stable position. There is also very little chance that the algorithm will explore the region inbetween the two modes. We would need to increase the step size in order to explore this region. This shows that the Geweke test can be misleading as it suggests that it has converged. 

# (e) Now plot the histogram of the remaining samples. Is the M-H algorithm converging?

# Not sure what is being asked here - plot the histogram of the samples from burn-in to the end? if so, then we will see that a normal distribution with $\mu$ around -6 is formed, due to the mentioned stability of this point and the small value for $\sigma$.

# In[108]:

fig = plt.hist(accepted[burnin:], 20, alpha=0.3)


# (f) Finally, using $\sigma=4$ for the proposal distribution, re-answer problems (d) and (e). For the Geweke test, try 
# segments of 100 and 500 to calculate the means and variances. If the results differ, please explain why. 

# In[111]:

#set sig to 4
sig = 4

#same code as before (I know I should have made this into a function) but with x0 = -6
x0 = -6.0
x_prev = x0

n = 10000
accepted = np.zeros(n)
k=1
i=0

while i<n:
    
    #choose a proposal step for where to move to next
    #x_star = q(x, x_prev)
    x_star = np.random.normal(x_prev,sig)


    #work out if it is a good idea to move:
    p_star = p(x_star)
    p_prev = p(x_prev)
    p_star = np.log(p_star)
    p_prev = np.log(p_prev)
    
    U= np.random.uniform()
    A = np.min((1,p_star - p_prev)) #acceptance probability. By changing to log dist. of p(x) then
                        #we can subtract and not divide.
    
    if np.log(U) < A: #accept proposal and set x_prev as the accepted value
        accepted[i] = x_star
        i = i + 1
        x_prev = x_star
    else: #reject proposal and stay where you are
        accepted[i] = x_prev
        x_prev = accepted[i]  
        i = i + 1
        k=k+1  
        
# =======
# PLOTTING
burnin = 100

plt.figure(1, figsize=(14,14))
plt.subplot(4,2,1)
fig = plt.hist(accepted[burnin:], 20, alpha=0.3)
plt.tight_layout()
#fig = plt.hist(accepted[300:], 20, alpha=0.3)

plt.subplot(4,2,2)
plt.plot(accepted[burnin:]) 

#Geweke Test


geweke2 = Geweke(accepted[burnin:], 10, 100)
plt.subplot(4,2,3)
plt.plot(geweke2)
plt.title('Geweke, interval = 100')
geweke3 = Geweke(accepted[burnin:], 10, 500)
plt.subplot(4,2,4)
plt.plot(geweke3)
plt.title('Geweke, interval = 500')
plt.show()


# Increasing the step size makes the algorithm much better at searching the region beyond the first mode. The periodic nature of the trace plot indicates this hopping from one mode to the other with very little time spent sampling in the valley between the two modes. Not sure how to interpret the Geweke plots - with length = 100, the test is greater than 2 at points, whereas with length = 500, the test passes at all points. The fact that they don't lie within the range (-2,2) suggests that the algorithm has not converged but we know from the first plot that the algorithm has done a good job at reproducing the target distribution. Perhaps the lesson is again, that we cannot rely too much on the Geweke Test.
# 
# Not sure how to calculate var and mean based on Geweke - perhaps use the cumulative values for $\sigma_a$, $\sigma_b$ and $\bar{\theta_A}$, $\bar{\theta_B}$ as defined in the original Geweke function definition from all the different segments. 

#### Question 2: Thinning of Chains

# A good performance test of the ability of a technique to sample the full
# parameter space is to test the sampling of the Rosenbrock 'Banana' Function.
# Consider the following specific form of a Banana PDF,
# 
# $ p(X) \propto {\rm exp} \left[ - \frac{1}{2a^2} \left(\sqrt{x_1^2 + x_2^2} -1 \right)^2 -  \frac{1}{2b^2} \left(x_2  - 1 \right)^2  \right] $ where $a=0.1$ and $b=1$.
# 
# (a) Visualize the Rosenbrock 'Banana' Function.
# 
# (b) Using the proposal function,
# 
# $q(Y|X) = \frac{1}{2 \pi \sigma^2} {\rm exp}\left[-\frac{1}{2 \sigma^2} ((y_1-x_1)^2+(y_2-x_2)^2) \right] $ 
# 
# with $\sigma = 1$, construct a M-H algorithm to produce $N=10,000$ samples from $p(X)$. Let the starting point be $X=(-1,0)$. Plot the results and identify an appropriate burn-in sample size.
# 
# (c) The effective sample size of a M-H result is defined as:
# 
# $\rm{N_{eff}} =\frac{N}{1+2\sum_{t=1}^\infty \rho_t}$ 
# 
# where $\rho_t = cor(X_i, X_{i+t})$  is the autocorrelation of the sequence at lag t. For simplicity, we only consider the chain of $x_2$, that is $\rho_t = cor(x_{i,2}, x_{i+t,2})$ [HINT: Python's numpy.corrcoef computes correlation]. For simplicity, use an upper bound of 100 for lag:
# 
# $\rm{N_{eff}} =\frac{N}{1+2\sum_{t=1}^{100} \rho_t}$. 
# 
# Compute the effective sample size of the M-H samples.
# 
# (d) Compute the sample mean (after burnin)  $\bar{x}_2$. Repeat (b) for 100 times and calculate the variance of $\bar{x}_2$. For simplicity, choose the same burn-in sample size for each run.
# 
# (e) Now, perform  thinning, i.e. using 1 out of every 10 samples. Construct a M-H algorithm that yields $N=10,000$ samples (after thinning). Plot the results and choose an appropriate sample size.
# 
# (f) Compute the effective sample size with thinning and compare it with the result from (c). Do you see any improvement? Explain why.
# 
# (g) Repeat (d) with thinning,  and compare the variances. Do you see any imporvement? Explain why.

#### Solution to Q.2:

# $ p(X) \propto {\rm exp} \left[ - \frac{1}{2a^2} \left(\sqrt{x_1^2 + x_2^2} -1 \right)^2 -  \frac{1}{2b^2} \left(x_2  - 1 \right)^2  \right] $ where $a=0.1$ and $b=1$.
# 
# (a) Visualize the Rosenbrock 'Banana' Function.

# In[8]:

##-----------
##FUNCTION
banana = lambda x1, x2: np.exp(-1/(2*a**2)*(np.sqrt(x1**2+x2**2)-1)**2-1/(2*b**2)*(x2-1)**2)

##-----------
##CONSTANTS
a = 0.1
b = 1.0

x1,x2 = mgrid[-1.5:1.5:50j, 0.7:2:50j]
contourf(x1, x2, banana(x1,x2), cmap=cm.Purples_r)
show()    


# (b) Using the proposal function,
# 
# $q(Y|X) = \frac{1}{2 \pi \sigma^2} {\rm exp}\left[-\frac{1}{2 \sigma^2} ((y_1-x_1)^2+(y_2-x_2)^2) \right] $ 
# 
# with $\sigma = 1$, construct a M-H algorithm to produce $N=10,000$ samples from $p(X)$. Let the starting point be $X=(-1,0)$. Plot the results and identify an appropriate burn-in sample size.

# In[125]:

#probably have to use component-wise updating as this is two dimensional problem
#assuming that proposal function is symmetric and therefore no need to use correction factor

sig = 1.0
#proposal function - I CAN"T WORK OUT HOW TO USE THIS SO WILL USE NORMAL INSTEAD
q = lambda x1, x2, y1, y2: 1/(2*np.pi*sig**2)*np.exp(-1/(2*sig**2)*((y1-x1)**2+(y2-x2)**2))

#target distribution
banana = lambda x1, x2: np.exp(-1/(2*a**2)*(np.sqrt(x1**2+x2**2)-1)**2-1/(2*b**2)*(x2-1)**2)

#number of samples
n=10000

#intitialize the sampling. Start at X = (-1,0)
x_10 = -1
x_20 = 0
x1_prev = x_10
x2_prev = x_20

x=np.zeros(n)
y=np.zeros(n)
k=1
i=0

#
while i<n:
    
    #let's try to do multivariate MH with component-wise updating
    #take a step first in X, according to proposal distribution q_x_y
    #not sure how to select x2 and y2
    #x2 = np.random.normal(x_prev, sig)
    #y2 = np.random.normal(y_prev, sig)
    #x_star = q(x_prev,y_prev,x2,y2)
    x1_star = np.random.normal(x1_prev, sig)
    
    F_star = np.log(banana(x1_star,x2_prev))
    F_prev = np.log(banana(x1_prev,x2_prev))
    U = np.random.uniform()
    
    A = np.min((1, (F_star-F_prev))) #Acceptance probability including correction factor. 
                                   
    #A = F_star/F_prev        
   
    if U < A:
        x[i] = x1_star
        i = i + 1
        x1_prev = x1_star
      
    else:
        x[i] = x1_prev
        x1_prev = x[i]
        i = i + 1
        k = k+1
        
    i=i-1
    
    # now lets do the y step 
    #x_star = q(x_prev,y_prev,x2,y2) #as given by proposal distribution q_x_y(x,y)
    x2_star = np.random.normal(x2_prev,sig)
    
    F_star = np.log(banana(x1_prev, x2_star))
    F_prev = np.log(banana(x1_prev, x2_prev))
    
    #acceptance probability
    A =  np.min((1, (F_star-F_prev)))
    #A = F_star/F_prev
    
    U = np.random.uniform()
     
    if U < A:
        y[i] = x2_star
        i = i + 1
        x2_prev = x2_star
    else: 
        y[i] = x2_prev
        x2_prev = y[i];  
        i = i + 1
        
        
# PLOTTING
plt.figure(figsize=[12,12])
counts,xd,yd,fig=plt.hist2d(x,y,31, cmap="Purples");
#or try plt.hexbin(x,y, cmap="Purples")


xt,yt = np.meshgrid(np.linspace(-3,3, 100),np.linspace(-3,3,100))

plt.contour(xt,yt,  np.exp(-xt**2/2) * np.exp(-yt**2/2))

plt.scatter(x,y, 0.1, alpha=0.2, cmap="BuGn")        


# Yeah, I just can't figure out how to do these multivariate problems - I had the same problem with last week's homework. I think that I'm not setting up my proposal choices of x_star and y_star correctly, especially when the proposal function is not a simple normal or uniform. I also don't really know how the 2D contour plot should be set-up. I just saw the solution from HW3 was posted and tried adapting that but ran out of time.

# In[126]:

x_mean= np.mean(x)
y_mean= np.mean(y)
np.std(x)
np.std(y)


# In[6]:

#target distribution
banana = lambda x1, x2: np.exp(-1/(2*a**2)*(np.sqrt(x1**2+x2**2)-1)**2-1/(2*b**2)*(x2-1)**2)

# First symmetric proposal (Metropolis)

x0=[5.,-5.]     # Starting point
cnt=0           # Initialize counts
Ns=100000       # Number of samples
xsample=np.zeros((2,Ns))
xsample[:,0]=x0
gam=1           # Gaussian parameter for the proposal distribution
acceptcnt=0


# In[9]:

lprop_old = np.log(banana(xsample[0,0],xsample[1,0]))

# Here we implement the Metropolis algorithm. The proposal distribution is Gaussian, and so we use.
# numpy.random.normal

while cnt+1 < Ns:
    xstar = xsample[:,cnt]+np.random.normal(0,gam,2)                # The next step in the chain is sampled randomly from a Gaussian
    lprop_new = np.log(banana(xstar[0],xstar[1]))
    
    if (lprop_new - lprop_old) > np.log(np.random.uniform(0,1,1)):  # Acceptance ratio. Notice that here only the ratio of the evaluated functions matters.
        x0=xstar
        lprop_old=lprop_new
        acceptcnt += 1
    cnt += 1
    xsample[:,cnt]=x0


# In[10]:

# Now let us produce a histogram from our samples.

xout=xsample[:,Ns/10.:] # Just for the sake of it, let's cut 10% of the sample for Burn-in 
plt.hist2d(xout[0,:], xout[1,:], normed=True, bins=100)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.show()

print 'The acceptance rate is:', acceptcnt/float(Ns)


# In[13]:

#proposal function - I CAN"T WORK OUT HOW TO USE THIS SO WILL USE NORMAL INSTEAD
q = lambda x1, x2, y1, y2: 1/(2*np.pi*sig**2)*np.exp(-1/(2*sig**2)*((y1-x1)**2+(y2-x2)**2))
sig =1.0
# And now we plot the proposed transition probability

foo=200
x = np.linspace(-5,5,foo)
y = np.linspace(-5,5,foo)
xx,yy = np.meshgrid(x,y) # Define a mesh grid in the region of interest
xx2,yy2 = np.meshgrid(x,y) # Define a mesh grid in the region of interest
zz=np.zeros((foo,foo))
for i in np.arange(0,foo):
    for j in np.arange(0,foo):
        zz[i,j] = q(xx[i,j],yy[i,j],1,1) # Populate the mesh

levels = np.arange(0,0.35,0.05)        
plt.contour(xx,yy,zz,levels)
axes().set_aspect('equal', 'datalim')



# (c) The effective sample size of a M-H result is defined as:
# 
# $\rm{N_{eff}} =\frac{N}{1+2\sum_{t=1}^\infty \rho_t}$ 
# 
# where $\rho_t = cor(X_i, X_{i+t})$  is the autocorrelation of the sequence at lag t. For simplicity, we only consider the chain of $x_2$, that is $\rho_t = cor(x_{i,2}, x_{i+t,2})$ [HINT: Python's numpy.corrcoef computes correlation]. For simplicity, use an upper bound of 100 for lag:
# 
# $\rm{N_{eff}} =\frac{N}{1+2\sum_{t=1}^{100} \rho_t}$. 
# 
# Compute the effective sample size of the M-H samples.

# In[ ]:




# (d) Compute the sample mean (after burnin)  $\bar{x}_2$. Repeat (b) for 100 times and calculate the variance of $\bar{x}_2$. For simplicity, choose the same burn-in sample size for each run.

# In[ ]:




# (e) Now, perform  thinning, i.e. using 1 out of every 10 samples. Construct a M-H algorithm that yields $N=10,000$ samples (after thinning). Plot the results and choose an appropriate sample size.

# In[ ]:




# (f) Compute the effective sample size with thinning and compare it with the result from (c). Do you see any improvement? Explain why.

# In[ ]:




# (g) Repeat (d) with thinning,  and compare the variances. Do you see any imporvement? Explain why.

# In[ ]:




#### Question 3: IMDB top five

# Suppose we ask individuals to remember the  order of the top five movies on IMDB. When asked afterward, some will not report the correct order, and the mistakes they make can be captured by simple probabilistic models.
#   
# Let's say that the top five movies are:  *The Shawshank Redemption*,*The Godfather*,  *The Godfather: Part II*, *The Dark Knight*  and  *Pulp Fiction*. This true ordering will be represented by a vector $\omega =$ (1,2,3,4,5). 
# 
# When someone ranks the movie as $\theta$ , the Hamming distance between that rank and the true rank is 
#   
# $d(\theta, \omega) = \sum_{i=1}^5 I_{\theta_i\neq \omega_i}$. 
#   
# For example, if $\theta$= (2,3,5,4,1), then $d(\theta, \omega)=4$, because only *The Dark Knight* is ranked correctly. 
# 
# Suppose the probability of a guess (expressed as $\theta$) is 
# 
# $ p(\theta | \omega, \lambda) \propto  e^{-\lambda d(\theta, \omega)} $.
# 
# (a) Implement an M-H algorithm to produce sample guesses from 500 individuals with different $\lambda=0.2, 0.5, 1.0$. What are the top five possible guesses?
# 
# (b) Compute the probability that *The Shawshank Redemption* is ranked as the top 1 movie by the M-H algorithm. Compare the results for different $\lambda$. What do you find?

#### Solution to Q.3:

# (a) Implement an M-H algorithm to produce sample guesses from 500 individuals with different $\lambda=0.2, 0.5, 1.0$. What are the top five possible guesses?

# In[ ]:




# (b) Compute the probability that *The Shawshank Redemption* is ranked as the top 1 movie by the M-H algorithm. Compare the results for different $\lambda$. What do you find?

# In[ ]:



