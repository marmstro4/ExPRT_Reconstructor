#Blame: Michael Armstrong  
#Contact: marmstro@lbl.gov  


This is a python code that fits pairs of straight lines through proton tracks recorded as up to 4 hits in the ExPRT tracker to obtain reaction verticies.

The reconstruction algorithm studies the residual between an extrapolation of simulatenously fitted lines to the z axis with a Gaussian penality in the transverse plane depending on the beam profile.

#Pre-requisites
python3 and the following modules: (most of which usually come preinstalled)  
math  
numpy  
scipy.optimize  
matplotlib  
optimize  
sklearn  
itertools  
scipy.integrate  
mpl_toolkits  
csv  
sys  
multiprocessing as mp  
time  

#Running

1. Input data is a .csv of the following format:
event_id,hit1x,hit1y,hit1z,hit2x,hit2y,hit2z,hit3x,hit3y,hit3z,hit4x,hit4y,hit4z

Then use the following command to run:
python3 main.py input.csv

2. This code is parrallelized, edit the main function parameter "nthreads" appropriately

3. The beam_spot parameter provides as essential boundary limit for the reconstruction. Assuming a 2D Gaussian profile this is the x,y sigma in mm

4. For analying simulations a Gaussian smear for hist is available in the simres parameters. Otherwise leave these 0.

#Results

Reconstructed verticies are stored in the 4D "vertex" object:
event_id,x,y,z

Analysis.py contains some functions for plotting these results if desired.
