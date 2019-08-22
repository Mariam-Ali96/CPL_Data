import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
###########
#DATA READ#
###########

#defining the directories because os.walk might loop over hidden not needed directories. 
direcs = ['M4025_D2.3_Z0.002', 'M10507_D2.3_Z0.002', 'plots', 'M24444_D2.3_Z0.002', 'M2001_D2.3_Z0.002', 'M8129_D2.3_Z0.002', 'M6029_D2.3_Z0.002', 'M15331_D2.3_Z0.002', 'M3003_D2.3_Z0.002', 'M1500_D2.3_Z0.002', 'M1000_D2.3_Z0.002']

#col_se, bev, bdat, bwdat columns of each file defined in the manula of the simulation
col_sev=["TIME[NB]", "I", "NAME", "K*", "RI[RC]", "M[M*]", "log10(L[L*])",
"log10(RS[R*])", "log10(Teff[K])"]

col_bev=["TIME[NB]", "I1", "I2", "NAME(I1)", "NAME(I2)", "K*(I1)", "K*(I2)", "K*(ICM)",
"RI[RC]", 'ECC', 'log10(P[days])', 'log10(SEMI[R*])',' M(I1)[M*]',
'M(I2)[M*]', 'log10(L(I1)[L*])', 'Log10(L(I2)[L*])', 'Log10(RS(I1)[R*])',
'Log10f(RS(I2)[R*])', 'Log10(Teff(I1)[K])', "Log10(Teff(I2)[K])"]



col_bdat = ['NAME(I1)', 'NAME(I2)', 'M1[M*]', 'M2[M*]', 'E[NB]', 'ECC', 'P[days]',
'SEMI[AU]', 'RI[PC]', 'VI[km/s]', 'K*(I1)', 'K*(I2)', 'ZN[NB]', 'RP[NB]',
'STEP(I1)[NB]', 'NAME(ICM)', 'ECM[NB]', 'K*(ICM)']





col_bwdat = ['NAME(I1)', 'NAME(I2)', 'M1[M*]', 'M2[M*]', 'E[NB]',
'ECC', 'P[days]','SEMI[AU]', 'RI[PC]', 'VI[km/s]', 'K*(I1)', 'K*(I2)']



mass_dict_sev = {}
mass_dict1 = {}
mass_dict2={}
mass_dict3={}
mass_all_bev = {}
mass_all_bdat = {}
mass_all_bwdat={}
mass_all= {}

#defining a dictionary of dictionaries to read all sev files and appending them with keys to be the time steps for the samall dictionaries
# the big dictionary have the keys to be the directory names
#then Stacking all dictionaries in all directories together according to their time steps
#Finally we have an array of arrays: "final" that's of the length of the time steps 305
#so each element in "final" array is an array of all masses of blackholes at that time step
for root, dirs, files in os.walk("../"):
	for d in direcs:
		if 'M' in d:
    			for f in files:
        			if "sev.83_" in f:
            				x = os.path.join(root,f)
            				df_sin = pd.read_csv(x,sep='\s+',names=col_sev)
            				black_holes=df_sin[df_sin['K*']==14.0]
            				mass_blackholes= black_holes['M[M*]']
            				mass = [float(i) for i in mass_blackholes]
            				mass_dict_sev[int(f[7:])]= mass
		mass_all[d] = mass_dict_sev
Mass = np.zeros(305)
Mass = [[] for i in Mass]
steps = np.arange(0,4880,16)
for i in mass_all:
	for j in steps:
		m = mass_all[i][j]
		indx = list(steps).index(j)
		Mass[indx].append(m)

final = []
for i in range(0,len(Mass)):
	final.append(np.hstack(Mass[i]))
################################################################################################################################################
#same done with binary files but with an extra steps of stacking the three final arrays of arrays together
#the final array "binary" has all masses of blackholes in binary system


for root, dirs, files in os.walk("../"): 
	for d in direcs: 
		if 'M' in d:
			for f in files:
				if "bev.82_" in f:
					x1 = os.path.join(root,f)
					df_bin1 = pd.read_csv(x1,sep='\s+',names=col_bev, skiprows = 1)
					k11=df_bin1[df_bin1["K*(I1)"]==14.0]
					k12 = df_bin1[df_bin1["K*(I2)"]==14.0]
					mass11= k11[' M(I1)[M*]'].values
					mass12 = k12[' M(I1)[M*]'].values
					mass11= np.array(mass11)
					mass12 = np.array(mass12)
					mass1 = np.concatenate((mass11,mass12))
					mass1 = np.array(mass1).flatten()
					mass_dict1[int(f[7:])]=mass1
			mass_all_bev[d] = mass_dict1
################################################################################################################################################
Mass_bev = np.zeros(305)
Mass_bev = [[] for i in Mass_bev]
steps = np.arange(0,4880,16)
for i in mass_all_bev:
	for j in steps:
		m = mass_all_bev[i][j]
		indx = list(steps).index(j)
		Mass_bev[indx].append(m)

final_bev = []
for i in range(0,len(Mass_bev)):
	final_bev.append(np.hstack(Mass_bev[i]))
################################################################################################################################################
for root, dirs, files in os.walk("../"): 
	for d in direcs: 
		if 'M' in d: 
			for f in files: 
				if "bdat.9" in f: 
					x2 = os.path.join(root,f) 
					df_bin2 = pd.read_csv(x2,sep='\s+',names=col_bdat, skiprows=4) 
					k21=df_bin2[df_bin2["K*(I1)"]==14.0] 
					k22 = df_bin2[df_bin2["K*(I2)"]==14.0] 
					mass21 = k21['M1[M*]'].values 
					mass21 = np.array(mass21) 
					mass22 = k22['M2[M*]'].values 
					mass22 = np.array(mass22) 
					mass2 = np.concatenate((mass21,mass22)) 
					mass2 = np.array(mass2).flatten() 
					mass_dict2[int(f[7:])]=mass2 
			mass_all_bdat[d] = mass_dict2




Mass_bdat = np.zeros(305)
Mass_bdat = [[i] for i in Mass_bdat]
steps = np.arange(0,4880,16)
for i in mass_all_bdat:
	for j in steps:
		m = mass_all_bdat[i][j]
		indx = list(steps).index(j)
		Mass_bdat[indx].append(m)

final_bdat = []
for i in range(0,len(Mass_bdat)):
	final_bdat.append(np.hstack(Mass_bdat[i]))
################################################################################################################################################
for root, dirs, files in os.walk("../"):
	for d in direcs:
		if 'M' in d:
			for f in files:
				if "bwdat.19_" in f:
					x3 = os.path.join(root,f)
					df_bin3 = pd.read_csv(x3,sep='\s+',names=col_bwdat,skiprows=2)
					k31=df_bin3[df_bin3["K*(I1)"]==14]
					k32 = df_bin3[df_bin3["K*(I2)"]==14]
					mass31 = k31['M1[M*]'].values
					mass31 = np.array(mass31)
					mass32 = k32['M2[M*]'].values
					mass32 = np.array(mass32)
					mass3 = np.concatenate((mass31,mass32))
					mass3 = np.array(mass3).flatten()
					mass_dict3[int(f[9:])]=mass3
			mass_all_bwdat[d] = mass_dict3
			
Mass_bwdat = np.zeros(305)
Mass_bwdat = [[i] for i in Mass_bwdat]
steps = np.arange(0,4880,16)
for i in mass_all_bwdat:
	for j in steps:
		m = mass_all_bwdat[i][j]
		indx = list(steps).index(j)
		Mass_bwdat[indx].append(m)

final_bwdat = []
for i in range(0,len(Mass_bwdat)):
	final_bwdat.append(np.hstack(Mass_bwdat[i]))
################################################################################################################################################
binary = []
for i in range(0,len(final_bdat)):
	m1= final_bev[i]
	m2 = final_bdat[i]
	m3 = final_bwdat[i]
	mass_con = np.concatenate((m1,m2,m3)).flatten()
	binary.append(mass_con)
	
	
length = []
for i in range(0,len(binary)):
	length.append(len(binary[i]))
	
##############################################################################################################################################
length_all = []
for i in range(0,len(final)):
	length_all.append(len(final[i]))
###############################################################################################################################################
##############
#Queation one#
##############
#Mass function of Blackholes
#I will plot the mass function of all black holes and those in binary system
# I plot elements at the first time steps
# at time steps at the middle :110,115,..
#then at the end 250...



keys1 = [50,55,60,75]
keys2 = [150,160,170,180]
keys3 = [205,230,275,290]


(fig, subplots) = plt.subplots(3, 4, figsize=(25, 25))
for j in keys1:
    indx = list(keys1).index(j)
    ax0 = subplots[0][indx]
    x = final[j]
    y = binary[j]
    if len(x)!=0:
        ax0.hist([x,y])
        legend = ['Blackholes', 'Binary Blackholes']
        ax0.legend(legend)
        fig.suptitle('BlackHoles MassFunction', fontsize=20)
        ax0.set_title('mass function of time:'+str(steps[j]), fontsize=10)
    #ax0.set_xlim(0,5)
    #ax0.set_ylim(0,1000)
        ax0.set_xlabel('Mass in Solar mass', fontsize=10)
        ax0.set_ylabel('Counts', fontsize=10)




for k in keys2:
    indx1 = list(keys2).index(k)
    ax1 = subplots[1][indx1]
    x = final[k]
    y = binary[k]
    if len(x)!=0:
        ax1.hist([x,y])
        legend = ['Blackholes', 'Binary Blackholes']
        ax1.legend(legend)
    #fig.suptitle('BlackHoles MassFunction_'+str(i), fontsize=15)
        ax1.set_title('mass function of time:'+str(steps[k]), fontsize=10)
    #ax1.set_xlim(0,5)
    #ax1.set_ylim(0,1000)
        ax1.set_xlabel('Mass in Solar mass', fontsize=10)
        ax1.set_ylabel('Distribution', fontsize=10)

for l in keys3:
    indx2 = list(keys3).index(l)
    ax2 = subplots[2][indx2]
    x = final[l]
    y = binary[l]
    ax2.hist([x,y])
    legend = ['Blackholes', 'Binary Blackholes']
    ax2.legend(legend)
    #print(final_massall[i])
    #fig.suptitle('BlackHoles MassFunction_'+str(i), fontsize=15)
    ax2.set_title('mass function of time:'+str(steps[l]), fontsize=10)
    #ax2.set_xlim(0,5)
    #ax2.set_ylim(0,1000)
    ax2.set_xlabel('Mass in Solar mass', fontsize=10)
    ax2.set_ylabel('Distribution', fontsize=10)

fig.savefig('Blackholes_Mass_function.png')
################################################################################################################################################
##############
#Queation two#
##############
#How many of the blackholes are in binary system
#length and length all, pre-defined lengthes of all masses arrays in both "final" and "binary"

length = np.array(length)
length_all = np.array(length_all)
denum = length+length_all
fraction  = length/denum		



print("Fraction of blackholes in binary system:", fraction, sep = "\n")

##############################################################################################################################################
################
#Question Three#
################
#I wil plot these fractions with Crossing time

#I will remove the zero value of number of black holes
#and 1. value in the fraction because they don't show much information
#in the evolution plot
fraction = [float(i) for i in fraction]
fraction = np.array(fraction)
frax = [i for i in fraction if i != 1.0]
#the I will try to know what are the corresponding indexes of tee removed values
indx_rm = np.where(fraction==1.0)
new_steps= np.delete(steps,indx_rm)
#these index values correspond to 1.0 in fraction and 0.0 in length_all "OBVIOUSLY"
new_len = [float(i) for i in length_all if i != 0]

avrg = [np.mean(frax)]*len(new_steps)
(fig1, subplots1) = plt.subplots(2,1, figsize=(25, 25))
fig1.suptitle('Evoltion of Black holes with Crosing time', fontsize=28)
ax01 = subplots1[0]
ax01.plot(new_steps, frax,'ro', label='binary_blackholes_evolution', linestyle='dashed')
ax01.plot(steps, avrg, label='Mean of binary fractions',color ='blue', linestyle='--')
ax01.set_xlabel('time', fontsize=20)
ax01.set_ylabel('fraction of blackholes in binary systems', fontsize=20)
ax01.legend()
  
ax02 = subplots1[1]
ax02.plot(new_steps, new_len,'go', label='Black_holes with time', linestyle='dashed')
ax02.set_xlabel('time', fontsize=20)
ax02.set_ylabel('NUMBER OF BLACK HOLES', fontsize=20)
ax02.legend()
fig1.savefig('EVOLUTION_PLOT.png')








