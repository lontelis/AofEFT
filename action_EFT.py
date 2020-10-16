import numpy as np
import matplotlib.pyplot as plt
import subprocess
from cosmopit import numMath,fitting,cosmology
import mypymcLib


subprocess.call(['mkdir','NPZ'])
subprocess.call(['mkdir','PDF'])

def model(x,pars):
	beta_s1  = pars[0]
	phi_1    = pars[1]
	n_1      = pars[2]
	alpha_s2 = pars[3]
	phi_2    = pars[4]
	n_2      = pars[5]
	result   = beta_s1*np.exp(-0.5*((phi_1*x-0.1)/n_1)**2.) + alpha_s2 + phi_2*x + n_2
	return(result)

Theta = [1,2,1,1,2,1]
Theta_for_sigma_sim = list(np.array(Theta)*0.1)


xvec = np.linspace(-2,2,100)

d_sim     = model(xvec,Theta)
sigma_sim = model(xvec,Theta_for_sigma_sim)


plt.ion()
plt.figure(1,figsize=(10,6)),plt.clf()
plt.errorbar(xvec,d_sim,yerr=sigma_sim,fmt='.',label='Simulated data')
plt.plot(xvec,model(xvec,Theta),'r-',label='Model')
plt.xlabel('x',size=15)
plt.ylabel('$S_{\mathrm{EFT}}^{\mathrm{Simplified,2}}(x)$',size=15)
plt.legend()
plt.draw()
plt.show()
plt.savefig('PDF/ActionEFT_model_data_error.pdf')


xvec_representation = np.linspace(-2,2,50)

d_sim_representation     = model(xvec_representation,Theta)
sigma_sim_representation = model(xvec_representation,Theta_for_sigma_sim)
plt.ion()
plt.figure(2,figsize=(10,6)),plt.clf()
plt.plot(xvec_representation,d_sim_representation,'.',label='Simulated data')
plt.fill_between(xvec_representation,d_sim_representation-2.*sigma_sim_representation,d_sim_representation+2.*sigma_sim_representation,color='#1f77b4',label='2$\sigma_{\mathrm{sim}}$',alpha=0.5)
plt.fill_between(xvec_representation,d_sim_representation-sigma_sim_representation,d_sim_representation+sigma_sim_representation,color='#1f77b4',label='1$\sigma_{\mathrm{sim}}$',alpha=0.8)
plt.plot(xvec,model(xvec,Theta),'r-',label='Model')
plt.xlabel('x',size=15)
plt.ylabel('$S_{\mathrm{EFT}}^{\mathrm{Simplified,2}}(x)$',size=15)
plt.legend()
plt.draw()
plt.show()
plt.savefig('PDF/ActionEFT_model_data_error_representation.pdf')


print('##### MCMC calculations and plots ')

def totoPlot(toto,color2='blue',Blabel=None,KMIN=20000,NsigLim=2,plotCorrCoef=True,labels_in=None,kk=0,Blabelsize=20,plotNumberContours='12',normalize_om_ol=False,paper2=False):   
	vars = toto['vars']
	chains = toto['chains'].item()         
	nchains = mypymcLib.burnChains(chains,kmin=KMIN)
	smth=10.0      
	mypymcLib.matrixplot(nchains, vars, color2, smth, labels=labels_in, Blabel=Blabel,NsigLim=NsigLim,Bpercentile=False,
	plotCorrCoef=plotCorrCoef,kk=kk,plotNumberContours=plotNumberContours,Blabelsize=Blabelsize,paper2=paper2) #,'$\Omega_k$'



if_doplot_MCMC=True
if_calculate_MCMC=True
if_consider_gaussian_prior='Prior_n1n2' # No: '' Yes: 'Prior_n1n2'
if if_doplot_MCMC:

	niter=50000
	nburn=0e3
	KMIN=int(0.25*niter)


	variables = ['beta_s1','phi_1','n_1','alpha_s2','phi_2','n_2'] 
	data      = mypymcLib.Data(
		xvals = xvec,
		yvals = d_sim,
		errors= sigma_sim,
		model = model)

	if if_consider_gaussian_prior=='Prior_n1n2':
		print('if_consider_gaussian_prior=',if_consider_gaussian_prior)
		def gaussian_prior(x,pars):
			beta_s1  = pars[0]
			phi_1    = pars[1]
			n_1      = pars[2]
			alpha_s2 = pars[3]
			phi_2    = pars[4]
			n_2      = pars[5]

			n_1_prior_mean     = 1.3
			n_2_prior_mean     = 2.0

			n_1_prior_sigma    = 0.1
			n_2_prior_sigma    = 0.1

			result_chi_prior   = ( (n_1-n_1_prior_mean)/n_1_prior_sigma )**2. + ( (n_2-n_2_prior_mean)/n_2_prior_sigma )**2.
			return(result_chi_prior)
		data_gaussian_prior=mypymcLib.Data( model=gaussian_prior     , prior=True)
	elif if_consider_gaussian_prior=='': 
		print('No prior',if_consider_gaussian_prior)
	if if_calculate_MCMC:
		if if_consider_gaussian_prior=='Prior_n1n2':			
			print('if_consider_gaussian_prior=',if_consider_gaussian_prior)
			chains   = mypymcLib.run_mcmc(
				[data,data_gaussian_prior],
				variables=variables,
				niter=niter,
				nburn=nburn,
				w_ll_model='ActionEFT')
		elif if_consider_gaussian_prior=='':
			print('No prior',if_consider_gaussian_prior)
			chains   = mypymcLib.run_mcmc(
				data,
				variables=variables,
				niter=niter,
				nburn=nburn,
				w_ll_model='ActionEFT')

		chi2_chain = -2.*data([chains[0]['beta_s1'][-KMIN:].mean(),chains[0]['phi_1'][-KMIN:].mean(),chains[0]['n_1'][-KMIN:].mean(), chains[0]['alpha_s2'][-KMIN:].mean(),chains[0]['phi_2'][-KMIN:].mean(),chains[0]['n_2'][-KMIN:].mean()])

		np.savez('NPZ/ActionEFTofLSS'+if_consider_gaussian_prior+'.npz',
			data=data,chains=chains[0],vars=variables,niter=niter,nburn=nburn,chi2_chain=chi2_chain,chi2_chain_PYMC=chains[1])

	CHAIN_ref = np.load('NPZ/ActionEFTofLSS'+if_consider_gaussian_prior+'.npz',allow_pickle=True,encoding='bytes')
		
	ndf_CHAIN_ref = len(xvec) - len(variables)
	sigma_ndf_CHAIN_ref= np.sqrt(2.*ndf_CHAIN_ref)

	Blabel_in = '$\chi^2\pm\sqrt{2 ndf}$=%0.2f$\pm$%0.2f ,ndf=%0.2f'%(CHAIN_ref['chi2_chain_PYMC'],sigma_ndf_CHAIN_ref,ndf_CHAIN_ref)
	labels_variables_in = ['$\\beta^{(S_1)}$','$\\phi_1$','$n_1$','$\\alpha^{(S_2)}$','$\\phi_2$','$n_2$']


	plt.rcParams.update({'font.size': 15})
	plt.tick_params(labelsize=15)
	plt.ion()
	plt.figure(3,figsize=(15,8)),plt.clf() 
	totoPlot(CHAIN_ref,color2='black' ,Blabel=Blabel_in,labels_in=labels_variables_in,normalize_om_ol=False,NsigLim=5,plotCorrCoef=True,KMIN=KMIN)
	plt.draw()
	plt.show()
	plt.savefig('PDF/ActionEFT_MCMC'+if_consider_gaussian_prior+'.pdf')





