#!/usr/bin/env python
####################################################################################################
# IXS Spectrum fitter
# Gael Goret, Bjorn Wehinger and Alessandro Mirone for the European Synchrotron Radiation Facility
# gael.goret@esrf.fr
####################################################################################################
#
#  regarder Gefit pour voir si d'autres option [integration constrains fait !]
#
#  Convolution avec numpy ?? [fait!]
#
#  ajuster spacing sur largeur convolution [fait!]
#  ajuster spacing pour avoir 2**n [fait!]
#
#  Calculer en avance la fft de la fonction de convolution -> a faire
#  Interpolation aussi -> a faire
#
#  Calculer analytiquement la fft des  fonctions modele-> a faire
#  Voir comment utiliser le derivee analytiques.
#
#  FFTW en option avec les plan et l' option measure -> a faire
#
#  Voir widget Pymca-> a faire
#
#  Reintroduire l' option de simplex downhill.-> a faire
#


import sys, traceback, string, os, time, h5py
import numpy as np
from types import *
from matplotlib import pyplot as plt
from PyMca import Gefit 


#--------------------------------------------------------------------------------------------------------
# Basic function Model
#--------------------------------------------------------------------------------------------------------

def line(x,x0,A,w):
	""" normalised area lorenzian modeling inelastic line (integral = 1)"""
	return (2.*A/np.pi)*(w/(4.*(x-x0)**2+w**2))

def gauss(x,x0,A,w):
	""" normalised area gaussian"""
	return A*np.sqrt(4.*np.log(2)/np.pi)*(1./w)*np.exp(-((4.*np.log(2))/(w**2))*(x-x0)**2)

def pseudo_voigt(E,mu,wL,wG):
	"""amplitude pseudo-voigt used as resolution function"""
	return mu * line(E,0.,1.,wL) + (1-mu) * gauss(E,0.,1.,wG)

#--------------------------------------------------------------------------------------------------------
# Basic FT function Model
#--------------------------------------------------------------------------------------------------------


def ft_line(k,x0,A,w):
	""" Fourier transform of normalised area lorenzian modeling elastic line (integral = 1)"""
	return A*np.exp(-1.0j*k*(x0) - w/2.*abs(k))
	
def ft_gauss(k,x0,A,w):
	""" Fourier transform of normalised area gaussian"""
	return A*((np.exp(-(k**2*w**2)/(16*np.log(2))))/(np.sqrt(1./w**2)*w))*np.exp(-1.0j*k*(x0))

def ft_pseudo_voigt(S,mu,wL,wG):
	"""Fourier transform of amplitude pseudo-voigt used as resolution function"""
	return mu * ft_line(S,0,1.,wL) + (1-mu) * ft_gauss(S,0,1.,wG)


#--------------------------------------------------------------------------------------------------------
# Fitting function Model
#--------------------------------------------------------------------------------------------------------

class model:
	def __init__(self,T,E,res_param,d):
		self.T = T
		self.kb = 1.3806504e-23/1.602176462e-22
		self.E = E
		self.xmin= E[0]
		self.xmax= E[-1]
		self.norm = (self.xmax - self.xmin)
		
		self.mun,self.wGn,self.wLn = res_param[int(d)]
		self.npts = sup_pow_2(self.norm/(self.wGn/5.))
		
		self.Ech = np.arange(self.xmin,self.xmax,float((self.xmax-self.xmin))/self.npts)
		self.Sch = np.fft.fftshift(np.arange(-self.npts*np.pi/self.norm,self.npts*np.pi/self.norm,2*np.pi/self.norm))
		
		self.count=0
		
#--------------------------------------------------------------------------------------------------------

	def ft_inel_lines(self,S,param,peak_id):
		""" 
		inelastic lines (pair of peaks) !! Display only !! 
		based on lorentizan with amplitude factor = exp(-(pj[0]-Ec)/(k*T)) for the anti-stokes
		"""
		kb = self.kb
		T = self.T
		return ft_line(S,param.get_inel(peak_id).pos-self.xmin,param.get_inel(peak_id).amp,param.get_inel(peak_id).wid) + ft_line(S,-param.get_inel(peak_id).pos+2*param.Ec-self.xmin,param.get_inel(peak_id).amp,param.get_inel(peak_id).wid)*np.exp(-(param.get_inel(peak_id).pos-param.Ec)/(kb*T))


	def ft_inel_contrib(self,S,param):
		""" 
		inelastic contribution of I(S) 
		based on lorentizan with amplitude factor = exp(-(pj[0]-Ec)/(k*T)) for the anti-stokes
		"""
		kb = self.kb
		T = self.T
		
		return sum([ft_line(S,param.get_inel(i).pos-self.xmin,param.get_inel(i).amp,param.get_inel(i).wid) + ft_line(S,-param.get_inel(i).pos+2*param.Ec-self.xmin,param.get_inel(i).amp,param.get_inel(i).wid)*np.exp(-(param.get_inel(i).pos-param.Ec)/(kb*T)) for i in range(param.nb_inel_peaks)])

	def ft_i(self,S,param):
		"""
		ixs model function : 
		elastic_contribution + inelastic_contribution cf e_line and inel_contrib
		"""
		return ft_line(S,param.Ec-self.xmin,param.get_el().amp,param.get_el().wid)  + self.ft_inel_contrib(S,param)
	
	def Ft_I(self,p, E, interpolation=1): 
		""" Interface """
		return self.ft_I(E,p,interpolation=interpolation)
		
	def ft_I(self,E,p,interpolation=1):
		"""
		i(E) convoluted with resolution (pseudo voigt function)
		p = [Av,Ec,Ael,wel,xn,An,wn,...for all n]
		"""
		self.count+=1
		param = parameter_proxy(p)
		
		xmin = self.xmin
		xmax = self.xmax
		npts = self.npts
		norm = self.norm

		Sch = self.Sch
		Ech = self.Ech
		conv = np.fft.ifft(self.ft_i(Sch,param)*ft_pseudo_voigt(Sch,self.mun,self.wLn,self.wGn))*npts
		conv = conv.real
		if interpolation:
			return np.interp(E,Ech,conv)
		else :
			return conv

#--------------------------------------------------------------------------------------------------------
# Fitting Class	
#--------------------------------------------------------------------------------------------------------

class fitting:
	"""
	Return the point which minimizes the sum of squares of M (non-linear) equations 
	in N unknowns given a starting estimate using a modification of the Levenberg-Marquardt algorithm.
	"""
	def __init__(self, param, x, y,yerr):
		self.x=x
		self.y=y
		self.yerr=yerr
		self.param = [p for p in param]

	def define_func(self,func):
		self.f = func

	def rms(self,param_from_fit):
		self.param=param_from_fit[:]
		return np.sqrt(np.sum(((self.y - self.f(self.param,self.x))**2)))/len(self.x)

	def optimize(self,const=[]):
		p1, chisq, sigmapar = Gefit.LeastSquaresFit(self.f , self.param, constrains=const ,xdata=self.x , ydata= self.y ,sigmadata=self.yerr)
		return p1,chisq,sigmapar

#--------------------------------------------------------------------------------------------------------
# Misc Function    
#--------------------------------------------------------------------------------------------------------

def sup_pow_2(x):
	p=1
	while(p<x):
		p=p*2
	return p

def print_logo():
	print "\n"
	print "    ____ _  __ _____    ______ _  __   __             "
	print "   /  _/| |/ // ___/   / ____/(_)/ /_ / /_ ___   _____"
	print "   / /  |   / \__ \   / /_   / // __// __// _ \ / ___/"
	print " _/ /  /   | ___/ /  / __/  / // /_ / /_ /  __// /    "
	print "/___/ /_/|_|/____/  /_/    /_/ \__/ \__/ \___//_/ v1.5\n\n"
	

def extract_Ec_Ael_peaks(xy,noel):
	if noel:
		Ec=float(raw_input('Enter overall scan shift (Ec) : '))
		Ael=float(raw_input('Enter intensity of elastic line (Ael) : '))
		return Ec,Ael,xy
	else : 
		Ec,Ael = xy.pop(0)
		return Ec,Ael,xy


def get_xy(event):
	if event.button == 1:
	        if event.inaxes:
	        	global xy,cid
		    	print 'Peak added -> ', event.xdata, event.ydata
		    	xy += [[event.xdata,event.ydata ]]


def zeroinv(v):
	return np.sign(v[0])==np.sign(v[-1])

	
def print_params(param,T,sigma=None):
	if sigma == None:
		sigma =  list(np.zeros(param.nb_params))
	sigEc,sigAel,sigwel = sigma[:3]
	sigppk = sigma[3:]
	sigtmp = sigppk[:]
	sigppk = []
	
	for i in range(0,param.nb_inel_peaks*3,3):
		sigppk += [[sigtmp[i],sigtmp[i+1],sigtmp[i+2]]]


	print'-------------------------------------------'
	print'temperature : %.2f'%T
	print'-------------------------------------------'
	print'elastic line :'
	print'overall scan shift        [Ec]  = %.4f (%.4f)'%(param.get_el().Ec,sigEc)
	print'intensity of elastic line [Ael] = %.4f (%.4f)'%(param.get_el().amp,sigAel)
	print'width of elastic line     [wel] = %.4f (%.4f)'%(param.get_el().wid,sigwel)
	print'-------------------------------------------'

	for i in range(param.nb_inel_peaks):
		print'inelastic line %d :'%(i+1)
		print'position of exitation    [E%d] = %.4f (%.4f)'%(i+1,param.get_inel(i).pos,sigppk[i][0])
		print'intensity of exitation   [A%d] = %.4f (%.4f)'%(i+1,param.get_inel(i).amp,sigppk[i][1])
		print'width of inelastic lines [w%d] = %.4f (%.4f)'%(i+1,param.get_inel(i).wid,sigppk[i][2])
	print '--------------------------------------------------------------'

#--------------------------------------------------------------------------------------------------------

def interactive_extract_data_from_h5(hdf):
	scans = np.int32(hdf.keys())
	scans.sort()
	print '--------------------------------------------------------------'
	print scans
	print '--------------------------------------------------------------'
	counterror=0
	while 1:
		s = raw_input('Enter scan number : ')
		if s not in hdf.keys():
			if(counterror<4):
				print  'Scan %s not found in file %s  Try Again you have still %d try '%(s,hdf.filename, 4-counterror)
			else:
				raise  Exception , ('Scan %s not found in file %s'%(s,hdf.filename))
			counterror+=1
		else:
			break

	detas = np.int32(hdf[s].keys())
	detas.sort()
	print '--------------------------------------------------------------'
	print detas
	print '--------------------------------------------------------------'
	d = raw_input('Enter detector number : ')
	print '--------------------------------------------------------------'
	if d not in hdf[s].keys():
		raise  Exception , ('Detector %s not found for scan %s'%(d,s))
	E = np.array(hdf[s][d]['DE'],dtype=float).round(3)#why rounded is mandatory ? 
	A = np.array(hdf[s][d]['NI'],dtype=float).round(2)
	Err = np.array(hdf[s][d]['NErr'],dtype=float).round(2)
	return s,d,E,A,Err

#--------------------------------------------------------------------------------------------------------
	
def GUI_get_init_peak_params(E,A):
	plt.ion()
	
	plt.plot(E,A,label='Experimental data')
	plt.xlabel("Energy")
	plt.ylabel("Intensity")
	plt.title("IXS Spectrum")
	fig  = plt.gcf()
	fig.set_size_inches(8*1.4,6*1.2,forward=True) 
	ax = fig.gca()
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
	plt.legend(loc='upper center', bbox_to_anchor=(1.1, 1), fancybox=True, shadow=True, ncol=1)
	plt.draw()

	global xy,cid
	xy=None
	noel = zeroinv(E)
	if noel:
		print "Select ONLY the anti-stocks maxima for each exitation."
		print "Close the plot to start the fit."
	else:
		print "Pick first the maximum of the elastic line."
		print "Select the anti-stocks maxima for each exitation."
		print "Close the plot to start the fit."
	cid = plt.connect('button_press_event', get_xy)
	xy=[]
	plt.ioff()
	plt.show()
	plt.disconnect(cid)
	return xy,noel

#--------------------------------------------------------------------------------------------------------	
	
def build_params_list(E,A,xy,noel):
	wel,wj =0.1,0.1 #widths of elastic and exitation peaks (initial guess)
	Ec,Ael,peaks=extract_Ec_Ael_peaks(xy,noel)
	ppk = []
	for pk in peaks:
		ppk +=[pk[0],pk[1],wj]
	
	xmin= E[0]
	xmax= E[-1]
	init_spacing = (xmax-xmin)/len(E)		
	p = np.array([Ec,Ael,wel] + ppk)
	return p
	
#--------------------------------------------------------------------------------------------------------

def build_constrains(nb_params,position=0,prange=[],intensity=0,irange=[],width=0,wrange=[]):
	"""
	preparing constrains for optimizing amplitudes
	0 = Free       1 = Positive     2 = Quoted
	3 = Fixed      4 = Factor       5 = Delta	
	6 = Sum        7 = ignored
	"""
	c = list(np.zeros((3,nb_params)))
	for i in range(nb_params-2):
		if i%3==0:
			if position == 2:
				c[0][i]=position
				if prange != []:
					c[1][i]=prange[0]
					c[2][i]=prange[1]
				else:
					c[1][i]=0.
					c[2][i]=400.
			else : 
				c[0][i]=position
			if intensity == 2:
				c[0][i+1]=intensity
				if irange != []:
					c[1][i+1]=irange[0]
					c[2][i+1]=irange[1]
				else:
					c[1][i+1]=0.
					c[2][i+1]=1000.
			else : 
				c[0][i+1]=intensity
			
			if width == 2:
				c[0][i+2]=width
				if wrange != []:
					c[1][i+2]=wrange[0]
					c[2][i+2]=wrange[1]
				else:
					c[1][i+2]=0.
					c[2][i+2]=10.
			else : 
				c[0][i+2]=width
			if i == 0:#Ec
				c[0][i]=0
    				c[1][i]=0
				c[2][i]=0
	return c
    	
#--------------------------------------------------------------------------------------------------------
  	
def Fit(mod,p,E,A,Err,extconst=None):
	fit = fitting(p.get_list(),E,A,Err)#
	fit.define_func(mod.Ft_I)
	if extconst != None:
		print 'Optimization based on user-define parameter constrains, please wait ...'
		mod.count = 0
		t0=time.time()
		p,chisq,sigma = fit.optimize(const=extconst)
		t1=time.time()
		print 'Exec time for calculation : %f'%(t1-t0)
		print 'number of iteration in Levenberg-Marquardt : %d'%mod.count
		print 'Exec time per iteration : %f'%((t1-t0)/mod.count)
		print 'root-mean-square deviation : %.4f'%fit.rms(p)
		p = parameter_proxy(p)
		return p,chisq,sigma,extconst

	else :
		c1 = build_constrains(p.nb_params,position=3,intensity=2,irange=[0.,max(A)*1.5],width=3)
		print 'Amplitude refinement in progress'
		p0,chisq,sigma = fit.optimize(const=c1)
		print 'number of iteration in Levenberg-Marquardt : %d'%mod.count
		print 'root-mean-square deviation : %.4f'%fit.rms(p0)
		print '-------------------------------------------'
	
		print 'Optimization of all parameters in progress, please wait ...'
		mod.count = 0
		fit2 = fitting(p0,E,A,Err)
		fit2.define_func(mod.Ft_I)
		
		c2 = build_constrains(p.nb_params,position=2,prange=[0+p0[0],E[-1]*1.2],intensity=2,irange=[0.,max(A)*1.5],width=2,wrange=[0.,2.5])#XXX 
		t0=time.time()
		p1,chisq,sigma = fit2.optimize(const=c2)
		t1=time.time()
		print 'Exec time for calculation : %f'%(t1-t0)
		print 'number of iteration in Levenberg-Marquardt : %d'%mod.count
		print 'Exec time per iteration : %f'%((t1-t0)/mod.count)
		print 'root-mean-square deviation : %.4f'%fit2.rms(p1)
		p1 = parameter_proxy(p1)
		return p1,chisq,sigma,c2

#--------------------------------------------------------------------------------------------------------	
	
def Plot(mod,param,E,A):

	Ech = mod.Ech
	Sch = mod.Sch
		
	plt.plot(E-param.Ec,A,'blue',label='Experimental data')# plot : exp data
	plt.xlabel("Energy")
	plt.ylabel("Intensity")
	plt.title("IXS Spectrum Fitting")
	
	
	for i in range(param.nb_inel_peaks):
		peak = np.fft.ifft(mod.ft_inel_lines(Sch,param,i)*ft_pseudo_voigt(Sch,mod.mun,mod.wLn,mod.wGn))*mod.npts
		plt.plot(Ech-param.Ec,peak.real,'Cyan',label='Inelastic Contrib %d'%(i+1))
	
	convel = np.fft.ifft(ft_line(Sch,param.Ec-mod.xmin,param.get_el().amp,param.get_el().wid)*ft_pseudo_voigt(Sch,mod.mun,mod.wLn,mod.wGn))*mod.npts
	plt.plot(Ech-param.Ec,convel.real,'magenta',label='Elastic Contrib.')
	
	App = mod.Ft_I(param.get_list(),E,interpolation=0)
	plt.plot(Ech-param.Ec,App,'red',label='Fitted model')#plot : fitted model	

	fig  = plt.gcf()
	fig.set_size_inches(8*1.4,6*1.2,forward=True) 
	ax = fig.gca()
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
	plt.legend(loc='upper center', bbox_to_anchor=(1.1, 1), fancybox=True, shadow=True, ncol=1)
	return

#--------------------------------------------------------------------------------------------------------

def save_params(dirfn, fn, s,d,param,const,T,sigma):
	if not os.path.exists(dirfn):
		os.mkdir(dirfn)

	sigEc,sigAel,sigwel = sigma[:3]

	sigppk = sigma[3:]
	sigtmp = sigppk[:]
	sigppk = []
	readable_const = []
	for a in const:
		readable_const += [list(a)]
	
	for i in range(0,param.nb_inel_peaks*3,3):
		sigppk += [[sigtmp[i],sigtmp[i+1],sigtmp[i+2]]]

	out = open('%s/%s_%s_%s.param'%(dirfn,fn,s,d),'w')
	out.write('# Output parameters file\n')
	out.write('T = %.2f\n'%T)
	out.write('param = %s\n'%str(param.get_list()))
	out.write('constrains = %s\n'%str(readable_const))
	out.write('sigma = %s\n'%str(sigma))
	out.write('"""\n')
	out.write('--------------------------------------------------------------\n')
	out.write('Output parameters :\n')
	out.write('-------------------------------------------\n')
	out.write('temperature : %.2f\n'%T)
	out.write('-------------------------------------------\n')
	out.write('elastic line :\n')
	out.write('overall scan shift        [Ec]  = %.4f (%.4f)\n'%(param.Ec,sigEc))
	out.write('intensity of elastic line [Ael] = %.4f (%.4f)\n'%(param.get_el().amp,sigAel))
	out.write('width of elastic line     [wel] = %.4f (%.4f)\n'%(param.get_el().wid,sigwel))
	out.write('-------------------------------------------\n')
	for i in range(param.nb_inel_peaks):
		out.write('inelastic line %d :\n'%(i+1))
		out.write('position of exitation    [E%d] = %.4f (%.4f)\n'%(i+1,param.get_inel(i).pos,sigppk[i][0]))
		out.write('intensity of exitation   [A%d] = %.4f (%.4f)\n'%(i+1,param.get_inel(i).amp,sigppk[i][1]))
		out.write('width of inelastic lines [w%d] = %.4f (%.4f)\n'%(i+1,param.get_inel(i).wid,sigppk[i][2]))
	out.write('--------------------------------------------------------------\n')
	out.write('"""')
	out.close()

#--------------------------------------------------------------------------------------------------------

def save_data(dirfn,fn, s,d,param,mod,E,A,Err,T,sigma):
	if not os.path.exists(dirfn):
		os.mkdir(dirfn)

	Ech = mod.Ech
	Sch = mod.Sch
	
	fit = mod.Ft_I(param.get_list(),E,interpolation=1)
	
	convel = np.fft.ifft(ft_line(Sch,param.Ec-mod.xmin,param.get_el().amp,param.get_el().wid)*ft_pseudo_voigt(Sch,mod.mun,mod.wLn,mod.wGn))*mod.npts
	elcontrib = np.interp(E,Ech,convel.real)
	
	inelcontrib = []
	for i in range(param.nb_inel_peaks):
		inel = np.fft.ifft(mod.ft_inel_lines(Sch,param,i)*ft_pseudo_voigt(Sch,mod.mun,mod.wLn,mod.wGn))*mod.npts
		inelinterp = np.interp(E,Ech,inel.real)
		inelcontrib +=[inelinterp]
	
	Ldat = [E-param.Ec,A,Err,fit,elcontrib]
	for inel in inelcontrib:
		Ldat+=[inel]
	datout = np.column_stack(np.array(Ldat))
	np.savetxt('%s/%s_%s_%s.dat'%(dirfn,fn,s,d), datout, fmt='%14.4f', delimiter=' ')

	
def file_print(dirfn,fn, s,d):
	if not os.path.exists(dirfn):
		os.mkdir(dirfn)
	plt.savefig('%s/%s_%s_%s.png'%(dirfn,fn,s,d),format = 'png')#format among : png, pdf, ps, eps, svg
	
#--------------------------------------------------------------------------------------------------------

def read_configuration_file(cfgfn,allowed_keys={} ):
	"""
	cfgfn is the filename of the configuration file.
	the function return an object containing information from configuration file (cf inside cfg file).
	"""
	
	try:
		s=open(cfgfn,"r")
	except:
		print " Error reading configuration file " ,  cfgfn			
		exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
		print "*** print_exception:"
		traceback.print_exception(exceptionType, exceptionValue, exceptionTraceback,
                              limit=None, file=sys.stdout)
		raise Exception
	class Config():
		exec(s)
	cfg = Config()
	
	
	for key in allowed_keys.keys() :
		if key not in dir(cfg):
			raise  Exception , ("Key not found in config file : %s"%key)
	for key in dir(cfg):
		if key not in allowed_keys.keys():
			if key[0] != "_":
				raise  Exception , ("Config file has spurious key %s"%key)
		else :
			if not (type(getattr(cfg,key)) is allowed_keys[key]):
				raise  Exception , ("Wrong type for key %s in config file"%key)
	return cfg

#--------------------------------------------------------------------------------------------------------

def define_ext_constrains(param_list,constrains):
	consttype = {0 : 'Free' ,1 : 'Positive' ,2 : 'Quoted' ,3 : 'Fixed'}
	pname_dict = build_param_name_dict(len(param_list))
	
	
	print '--------------------------------------------------------------'
	print 'Parameters & associated constrains   '
	print '-------------------------------------'
	print 'key : [param_name] = (param_value) -> const_flag ; [bound1,bound2]'
	for k in pname_dict.keys():
		print k,' : ','[%4s] = (%10.4f) -> '%(pname_dict[k],param_list[k]),int(constrains[0][k]),';','[%.4f,%.4f]'%(constrains[1][k],constrains[2][k])
	print '--------------------------------------------------------------'
	
	try :
		pnum = int(raw_input('Enter key number of the parameter that you would contrain : ').strip())
	except :
		print 'Error : Entry is not an allowed key'
		define_ext_constrains(param_list,constrains)
	if pnum not in pname_dict.keys() :
		print 'Error : value %s is not an allowed key'%pnum
		define_ext_constrains(param_list,constrains)
	
	print '--------------------------------------------------------------'
	print 'Constrains type : '
	for k in consttype.keys():
		print k,' : ',consttype[k]
	print '--------------------------------------------------------------'
	try :
		ctype = int(raw_input('What kind of contrain would you like to impose on parameter %s : '%(pname_dict[pnum])).strip()) 
	except:
		print 'Error : Entry is not an allowed key'
		define_ext_constrains(param_list,constrains)
	if ctype not in consttype.keys():
		print 'Error : value %s is not an allowed key'%ctype
		define_ext_constrains(param_list,constrains)
	
	if ctype == 0:  # Free
		constrains[0][pnum]=0		
		constrains[1][pnum]=0
		constrains[2][pnum]=0
	elif ctype == 1:# Positive
		constrains[0][pnum]=1		
		constrains[1][pnum]=0
		constrains[2][pnum]=0
	elif ctype == 2:# Quoted
		try:
			bounds = list(eval(raw_input('Enter bounds for the parameter (2 int or float separated by ",") : ').strip()))
		except:
			print 'Error : Last entry has not been written in the correct format'
			define_ext_constrains(param_list,constrains)
		constrains[0][pnum]=2		
		constrains[1][pnum]=bounds[0]
		constrains[2][pnum]=bounds[1]
	elif ctype == 3:# Fixed
		constrains[0][pnum]=3		
		constrains[1][pnum]=0
		constrains[2][pnum]=0
	else:
		print 'impossible case! sure ??'
		
	asknew = raw_input('Would you like to assigne a new start-value for parameter %s ? (y) or (n) [y] ?'%pname_dict[pnum]).strip()
	if asknew in ['','y','Y']:
		newval = raw_input('Enter a new value for parameter %s [%s] : '%(pname_dict[pnum],param_list[pnum])).strip()
		if newval != '':
		 	try:
		 		newval = float(newval)
		 	except:
		 		print 'Error : Last entry has not been written in the correct format'
		 		define_ext_constrains(param_list,constrains)
		 	param_list[pnum]=newval
		else:
			pass		 
	elif asknew in ['n','N']:
		pass
		
	redo = raw_input('Would you like to setup another constrain ? (y) or (n) [n] ? :\n((n) to refine with the new constrains/parameters)').strip()
	if redo == '':
		redo = 'n' 
	if redo in ['y','Y']:
		define_ext_constrains(param_list,constrains)
	elif redo in ['n','N']:
		return param_list,constrains
	else : 
		'Error : Entry is not matching any case, restarting constrains procedure'
		define_ext_constrains(param_list,constrains)

#--------------------------------------------------------------------------------------------------------

def build_param_name_dict(nbparam):
	pname_dict = {0:'Ec',1:'Ael',2:'Wel'}
	for i in range(3,nbparam,3):
		nbcontrib = i/3
		pname_dict[i] = 'E%d'%nbcontrib
		pname_dict[i+1] ='A%d'%nbcontrib
		pname_dict[i+2] ='W%d'%nbcontrib
	return pname_dict

#--------------------------------------------------------------------------------------------------------
# Class peak params
#--------------------------------------------------------------------------------------------------------

class peak_params(object):
	def __init__(self,peak_param_list,peak_id=None):
		self.peak_id=peak_id
		if peak_id==None:
			self.Ec = peak_param_list[0]	
		else : 
			self.pos= peak_param_list[0]
		self.amp=peak_param_list[1]
		self.wid=peak_param_list[2]

#--------------------------------------------------------------------------------------------------------
# Class parameters proxy
#--------------------------------------------------------------------------------------------------------

class parameter_proxy(object):
	def __init__(self,param_list):
		self.param_list = param_list
		self.nb_params = len(param_list)
		self.Ec = param_list[0]
		self.elastic_params = peak_params(param_list[0:3])
		self.nb_inel_peaks = (len(param_list)-3)/3
		self.inelastic_params = self.build_inel_object_params_list(param_list[3:])
	
	def build_inel_object_params_list(self,inel_param_list):
		inelastic_params = []
		for i in range(0,len(inel_param_list),3):
			inelastic_params += [peak_params(inel_param_list[i:i+3],peak_id = i)]
		return inelastic_params
	
	def get_list(self):
		return self.param_list
	
	def get_inel(self,peak_id):
		return self.inelastic_params[peak_id]
	
	def get_el(self):
		return self.elastic_params

	def normalise(self,mod):
		norm = pseudo_voigt(0,mod.mun,mod.wLn,mod.wGn)
		self.get_el().amp *= norm
		for i in range(self.nb_inel_peaks):
			self.get_inel(i).amp *= norm

def get_dotstripped_path_name( name ):
	posslash=name.rfind("/")
	posdot  =name.rfind(".")
	if posdot>posslash:
		name= name[:posdot]
	else:
		pass
	if(posslash>-1):
		return name, name[posslash+1:]
	else:
		return name, name
		

#--------------------------------------------------------------------------------------------------------
# Main
#--------------------------------------------------------------------------------------------------------

def main(argv):
	print_logo()
	fn = argv[1]
	cfg_filename = argv[2]
	hdf = h5py.File(fn,'r')
	
	# the allowed keys will be available as cfg members after reading parameter file 
	allowed_keys={"res_param":DictType,"T":FloatType}
	# EXAMPLE OF CFG FILE :
	""""
	# Parameters for resolution function
	# usage : res_param = {detector_number:[mu,wG,wL],...,n:[mun,wGn,wLn]}
	res_param ={
	1:[0.6552,2.604,4.53],
	2:[0.6319,2.603,4.013],
	..........................
	}
	#Temperature (important : floating type is mandatory)
	T = 297.0
	"""
	
	cfg = read_configuration_file(cfg_filename,allowed_keys= allowed_keys)
	interactive_Entry = True 
	mod=None
	const=None
	while(1):
		if interactive_Entry:
			s, d, E, A, Err = interactive_extract_data_from_h5(hdf)
			mod = model(cfg.T,E,cfg.res_param,d)
			xy, noel = GUI_get_init_peak_params(E,A)
			skip = (xy == [])

			param_list = build_params_list(E,A,xy,noel)
			param  = parameter_proxy(param_list)
			param.normalise(mod) 
			print '--------------------------------------------------------------'
			print 'Input parameters :'
			print_params(param,cfg.T)			
		else:
			skip=False
		
		if not skip:

			refined_param,chisq,sigma,const = Fit(mod,param,E,A,Err,extconst=const)
			Plot(mod,refined_param,E,A)

			print '--------------------------------------------------------------'
			print 'Output parameters :'
			print_params(refined_param,cfg.T,sigma)
			
			output_dir, output_stripped_name  = get_dotstripped_path_name(hdf.filename)

			save_params( output_dir, output_stripped_name       , s,d,refined_param,const,cfg.T,sigma=sigma)
			save_data  ( output_dir, output_stripped_name       , s,d,refined_param,mod,E,A,Err,cfg.T,sigma=sigma)
			file_print ( output_dir, output_stripped_name       , s,d)

			param = parameter_proxy(refined_param.get_list())
			plt.show(block=False)

			interactive_Entry=True
			r = raw_input('Would you like to fit another spectrum (y) or (n) default : [y] ?\nor change temperature (t) ?\nor refine again the previous fit with different constrains (r) ?\n')
			plt.close()
			if r in ['n','N']:
				print 'Bye Bye'
				break
			elif r in ['t','T']:
				T = raw_input('Temperature ? [297.0]: ')
				if T == '':
					cfg.T = 297.0
				else :
					cfg.T = float(T)
			elif r in ['r','R']:
				param_list = param.get_list()
				new_param_list,const = define_ext_constrains(param_list,const)
				param = parameter_proxy(new_param_list)
				interactive_Entry=False
			else:
				pass # will continue as default
	# now we exit from the main
	hdf.close()

#--------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
	argv = sys.argv
	if len(argv) in [3]:
		sys.exit(main(argv))
	else:
		print '\nusage : python ixs_fitter.py input_file.h5 file.conf\n'


		
