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
from numpy import array
import h5py



#--------------------------------------------------------------------------------------------------------
# Basic function Model
#--------------------------------------------------------------------------------------------------------


class LineModel:
	def nofMyParams(self):
		return 3
	def parNames(self):
		return ["Center",
			"Height",
			"Width" ]

	def __init__(self, arr_of_pars):
		self.Center = arr_of_pars[0:1]   # Note: this is a View . It changes when arr_of_pars changes
		self.Height = arr_of_pars[1:2] 
		self.W      = arr_of_pars[2:3]

	def get_Center(self):
		return self.Center
	def get_Height(self):
		return self.Height

	def ft_and_derivatives(self, reciprocal_grid, real_grid_origin, Stokes=+1 , ElC=None):
		result=np.zeros(   [   1+self.nofMyParams()  , len(reciprocal_grid),    ] ,"D"  )

		if Stokes==-1:
			Center = 2*ElC-self.Center - real_grid_origin
		else:
			Center =  self.Center - real_grid_origin

		result [2,:] = ft_line( reciprocal_grid, Center ,1.0, self.W    )
		result [0,:] = self.Height*result [2,:]
		result [1,:] = self.Height*result [0,:]*( -1.0j*reciprocal_grid)
		result [3,:] = self.Height*result [0,:]*( -1.0/2*abs(reciprocal_grid))
		

		return result
#--------------------------------------------------------------------------------------------------------
# Basic FT function Model
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



def ft_line(k,x0,A,w):
	""" Fourier transform of normalised area lorenzian modeling elastic line (integral = 1)"""
	return np.array(A*np.exp(-1.0j*k*(x0) - w*abs(k)/2.0) ,"D")
	
def ft_gauss(k,x0,A,w):
	""" Fourier transform of normalised area gaussian"""
	return np.array(A*np.exp(-((k**2)*(w**2))/(16*np.log(2)) )  *np.exp(-1.0j*k*(x0)) ,"D")

def ft_pseudo_voigt(S,mu,wL,wG):
	"""Fourier transform of amplitude pseudo-voigt used as resolution function"""
	return mu * ft_line(S,0,1.,wL) + (1-mu) * ft_gauss(S,0,1.,wG)

# --------------------------------------------------------------------------------------------------------
#  Convolution Models ( all have the same interface(excepted init function )
# ---------------------------------------------------------------------------------------------------------

class PseudoVoigt:
	def __init__(self ,  mu,lorentz_w ,gaussian_w , base_line):
		self.mu=mu
		self.gaussian_w=gaussian_w
		self.lorentz_w=lorentz_w
		self.base_line = base_line

	def safe_scale_length(self):
		return  min(self.lorentz_w/40,self.gaussian_w/40) 
	def safe_margin(self):
		 return (5*self.lorentz_w+3*self.gaussian_w)*2
	 
	def values_on_real_points(self,x ):
		 return   pseudo_voigt(x ,  self.mu,self.lorentz_w ,self.gaussian_w )+self.base_line
	def values_on_reciprocal_points(self,x ):
		 return   ft_pseudo_voigt(x ,  self.mu,self.lorentz_w ,self.gaussian_w )+self.base_line*len(x)
	 



#--------------------------------------------------------------------------------------------------------
# Fitting function Model
#--------------------------------------------------------------------------------------------------------

class Model:
	def __init__(self,T,E, convolution_function):
		self.convolution_function = convolution_function
		safe_step = self.convolution_function.safe_scale_length() 
		safe_margin = self.convolution_function.safe_margin() 
		
		self.T = T
		self.kb = 1.3806504e-23/1.602176462e-22
		self.E = E
		self.orig_xmin= E[0]  
		self.orig_xmax= E[-1] 
		self.xmin= E[0]  - safe_margin
		self.xmax= E[-1] + safe_margin

		self.norm = (self.xmax - self.xmin)
		self.npts = sup_pow_2(self.norm/(safe_step))

		self.fact4fft = self.npts/( self.xmax- self.xmin )

		self.Ech = np.arange(self.xmin,self.xmax,float((self.xmax-self.xmin))/self.npts)
		self.Sch = np.fft.fftshift(np.arange(-self.npts*np.pi/self.norm,self.npts*np.pi/self.norm,2*np.pi/self.norm))
		
		self.resolution_fft = self.fact4fft* self.convolution_function.values_on_reciprocal_points(self.Sch)  
		self.reso_ifft_max = np.fft.ifft(self.resolution_fft)[0]

		self.count=0
	
	def	set_Params_and_Functions(self, params_and_functions):
		self.params_and_functions=params_and_functions

#--------------------------------------------------------------------------------------------------------

	
	def Ft_I(self,p, E, interpolation=1, mask=None, convolution=1): 
		""" Interface """
		self.count+=1
		# update variables in self.params_and_functions
		self.params_and_functions.par_array[:] = p   # Note : we update internal values. We dont change the object reference value 
		if mask is None:
			mask=np.ones(len(self.params_and_functions.shapes) )

		icontribution=0
		ipar=0
		result=0.0
		for contribution in self.params_and_functions.shapes:
			npars = contribution.nofMyParams()
			parnames = contribution.parNames()
			fact=self.fact4fft*mask[icontribution]
			if icontribution==0:
				value_and_deri =  contribution.ft_and_derivatives( self.Sch, self.xmin , Stokes=0 )
				result=result+value_and_deri [0]*fact  # so far we exploit only function itself not derivatives..
				el_center = contribution.get_Center()
			else:
				inel_center = contribution.get_Center()
				value_and_deri =  contribution.ft_and_derivatives( self.Sch, self.xmin , Stokes=+1 )

				fact_stokes =  np.exp(+(inel_center-el_center)/(self.kb*self.T))
				

				result=result+value_and_deri [0]*fact * (fact_stokes/(fact_stokes-1))

				
				value_and_deri =  contribution.ft_and_derivatives( self.Sch, self.xmin , Stokes=-1, ElC=el_center )

				result=result+value_and_deri [0]*fact * (  1.0 /(fact_stokes-1))

			icontribution+=1
			ipar+=npars


		if convolution:
			result = np.fft.ifft(result*   self.resolution_fft )/self.fact4fft
		else:
			result = np.fft.ifft(result  ) 

		result = result.real
		if interpolation:
			return np.interp(E,self.Ech,result)
		else :
			return result


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
	

def get_xy(event):
	if event.button == 1:
	        if event.inaxes:
	        	global xy,cid
		    	print 'Peak added -> ', event.xdata, event.ydata
		    	xy += [[event.xdata,event.ydata ]]


def zeroinv(v):
	return np.sign(v[0])==np.sign(v[-1])


#--------------------------------------------------------------------------------------------------------

def interactive_extract_data_from_h5(hdf):
	global RECORD, REPLAY
	if not REPLAY==0:
		exec(getinstruction(REPLAY))
	else:
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
		if RECORD:
			open("interactive_session.log","a").write("s,d=%s   # in function interactive_extract_data_from_h5 \n"%str((s,d)))


	E = np.array(hdf[s][d]['DE'],dtype=float).round(3)#why rounded is mandatory ? 
	A = np.array(hdf[s][d]['NI'],dtype=float).round(2)
	Err = np.array(hdf[s][d]['NErr'],dtype=float).round(2)
	
	return s,d,E,A,Err

#--------------------------------------------------------------------------------------------------------

def get_data_from_txt(fn,detector):
        
    dat = open(fn,'r')
    
    E = []
    A = []
    Err = []
        
    for l in dat:
        E += [np.float32(l.split()[0])]
        A += [np.float32(l.split()[1])]
        Err += [np.float32(l.split()[2])]


    s = 0
    d = detector
    E = np.array(E)
    A = np.array(A)
    Err = np.array(Err)

    return s,d,E,A,Err
        
	
#--------------------------------------------------------------------------------------------------------

def interactive_GUI_get_init_peak_params(E,A):
	global RECORD, REPLAY
	if not REPLAY==0:
		s= getinstruction(REPLAY)
		exec(s, locals(), globals())
		noel=pippo
	else:
		#plt.ion()
		plt.plot(E,A,label='Experimental data')
		plt.xlabel("Energy transfer [meV]")
		plt.ylabel("Intensity [arb. units]")
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
			print "Select ONLY the anti-stokes maxima for each exitation."
			print "Close the plot to start the fit."
		else:
			print "Pick first the maximum of the elastic line."
			print "Select the anti-stokes maxima for each exitation."
			print "Close the plot to start the fit."
		cid = plt.connect('button_press_event', get_xy)
		xy=[]
		#plt.ioff()
		plt.show(block=True)
		plt.disconnect(cid)
		if RECORD:
			open("interactive_session.log","a").write("xy,pippo=%s   # in function interactive_GUI  \n"%str((xy,noel)))
	return xy,noel

#--------------------------------------------------------------------------------------------------------

def default_build_constrains( params_and_functions   ,position=0,prange=[],intensity=0,irange=[],width=0,wrange=[]):
	"""
	preparing constrains for optimizing amplitudes
	0 = Free       1 = Positive     2 = Quoted
	3 = Fixed      4 = Factor       5 = Delta	
	6 = Sum        7 = ignored
	"""
	if prange ==[]: prange=0.0,400.0
	if irange ==[]: prange=0.0,1000.0
	if wrange ==[]: prange=0.0,10.0

	indicators = {"Center":position ,"Height":intensity,"Width":width }
	indicators_limits = {"Center":prange ,"Height":irange,"Width":wrange }

	c = list(np.zeros((3, len(params_and_functions.par_array)  )))
	icontribution=0
	ipar=0
	for contribution in params_and_functions.shapes:
		npars = contribution.nofMyParams()
		parnames = contribution.parNames()
		for k in range(npars):
			if ipar+k == 0:
				c[0][ipar+k],c[1][ipar+k], c[2][ipar+k] =2,-prange[1]/2.0,prange[1]/2.0  # Free
			else:
				if indicators.has_key(parnames[k]) and indicators[parnames[k]]: 
					c[0][ipar+k]=indicators[parnames[k]]
					if indicators[parnames[k]]==2:
						c[1][ipar+k],c[2][ipar+k] =indicators_limits[parnames[k]]
					elif indicators[parnames[k]]==0:
						c[0][ipar+k],c[1][ipar+k], c[2][ipar+k] =0,0,0 # Free

		icontribution+=1
		ipar+=npars
	# c[1][0]=-c[2][0]
	return c
    	
#--------------------------------------------------------------------------------------------------------	
	
def Plot(mod,par_array,E,A, Err, show_graph=1):
	plt.xlim((min(E),max(E)))
	Ech = mod.Ech
	Sch = mod.Sch
	mod.params_and_functions.par_array[:] = par_array   # Note : we update internal values. We dont change the object reference value 
	Center = mod.params_and_functions.shapes[0].get_Center()
	if show_graph : 
		plt.errorbar(E-Center, A, Err,0, 'black',label = 'Errors on Exp data')
		plt.plot(E-Center,A,'blue',label='Experimental data')# plot : exp data
		
		plt.xlabel("Energy transfer [meV]")
		plt.ylabel("Intensity [arb. units]")
		plt.title("IXS Spectrum Fitting")

	mask=np.zeros(len(mod.params_and_functions.shapes) )

	Ldat = [E , A, Err]

	mask[:]=1	
	total_model = mod.Ft_I(par_array , Ech, interpolation=0, mask=mask) # with interpolation=0 Ech is dummy
	if show_graph : plt.plot(Ech-Center,total_model,'red',label='Fitted model')	

	Ldat.append(mod.Ft_I(par_array, E, interpolation=1, mask=mask))


	icontribution=0
	for contribution in mod.params_and_functions.shapes:
		mask[:]=0
		mask[icontribution]=1
		partial_model = mod.Ft_I(par_array, Ech, interpolation=0, mask=mask) # with interpolation=0 Ech is dummy
		if icontribution>0:
			if show_graph : plt.plot(Ech-Center,partial_model,'Cyan',label='Inelastic Contrib %d'%(icontribution))	
		else:
			if show_graph : plt.plot(Ech-Center,partial_model,'magenta',label='Elastic Contrib.')

		partial_model = mod.Ft_I(par_array, E, interpolation=1, mask=mask) # with interpolation=0 Ech is dummy
		Ldat.append(partial_model)
		icontribution+=1

	if show_graph :
		fig  = plt.gcf()
		fig.set_size_inches(8*1.4,6*1.2,forward=True) 
		ax = fig.gca()
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
		plt.legend(loc='upper center', bbox_to_anchor=(1.1, 1), fancybox=True, shadow=True, ncol=1)

	print Ech[0]
	print Center
	print mod.xmin
	print mod.xmax
	Ldat.append(list(np.interp( mod.xmin  + ((E-Center) %(mod.xmax-mod.xmin) ),Ech,np.fft.ifft(mod.resolution_fft).real)))
	return Ldat
	
def file_print(dirfn,fn, s,d):
	if not os.path.exists(dirfn):
		os.mkdir(dirfn)
	plt.savefig('%s/%s_%s.png'%(dirfn,fn,d),format = 'png')#format among : png, pdf, ps, eps, svg
	
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

	for key in cfg.res_param.keys():
		params = cfg.res_param[key]
		if(len(params)==3): params.append(0.0)
		
	return cfg

#--------------------------------------------------------------------------------------------------------

def build_param_name_dict(params_and_functions):
	icontribution=0
	ipar=0
	res={}
	for contribution in params_and_functions.shapes:
		npars = contribution.nofMyParams()
		parnames = contribution.parNames()
		if icontribution==0:
			# 'elastic line :'
			for k in range(npars):
				res[ipar+k] = parnames[k]+"_el"
		else:
			# 'inelastic line %d :'%(icontribution+1)
			for k in range(npars):
				res[ipar+k] = parnames[k]+("_%d"%icontribution)
		icontribution+=1
		ipar+=npars
	return res

def interactive_define_ext_constrains(params_and_functions,constrains):
  global RECORD, REPLAY
  consttype = {0 : 'Free' ,1 : 'Positive' ,2 : 'Quoted' ,3 : 'Fixed'}
  param_list = params_and_functions.par_array

  s="# Parameter constrains for optimization : \n# 0 = Free ; 1 = Positive ; 2 = Quoted ; 3 = Fixed\n"
  pname_dict = build_param_name_dict(params_and_functions)
  for k in pname_dict.keys():
	  s=s+("# ------------------------------------------   %s ----------------------------\n" % ( pname_dict[k], ))
	  s=s+("constrains[0][%d]= %d \n"%(k,int(constrains[0][k]) ))
	  s=s+("param_list[%d],constrains[1][%d],constrains[2][%d] = %10.4f,%.4f,%.4f  \n"% (k,k,k,param_list[k], constrains[1][k],constrains[2][k]  ))
	  
  filename="/tmp/inpufitter"
  open(filename,"w").write(s)
  os.system("emacs  %s >/dev/null < /dev/null " % (filename),)
  s= open(filename,"r").read()
  exec(s)
  return constrains

  if not REPLAY==0:
	  exec(getinstruction(REPLAY))
	  exec(getinstruction(REPLAY))
	  return constrains

  while(1):
	pname_dict = build_param_name_dict(params_and_functions)
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
		continue
	if pnum not in pname_dict.keys() :
		print 'Error : value %s is not an allowed key'%pnum
		continue
	
	print '--------------------------------------------------------------'
	print 'Constrains type : '
	for k in consttype.keys():
		print k,' : ',consttype[k]
	print '--------------------------------------------------------------'
	try :
		ctype = int(raw_input('What kind of contrain would you like to impose on parameter %s : '%(pname_dict[pnum])).strip()) 
	except:
		print 'Error : Entry is not an allowed key'
		continue
	if ctype not in consttype.keys():
		print 'Error : value %s is not an allowed key'%ctype
		continue
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
			continue

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
				continue
		 	param_list[pnum]=newval
		else:
			pass		 
	elif asknew in ['n','N']:
		pass
		
	redo = raw_input('Would you like to setup another constrain ? (y) or (n) [n] ? :\n((n) to refine with the new constrains/parameters)').strip()
	if redo == '':
		redo = 'n' 
	if redo in ['y','Y']:
		continue
	elif redo in ['n','N']:
			
		if RECORD:
			open("interactive_session.log","a").write("constrains=%s   # in function interactive-define-ext-constraints \n"%repr(constrains))
			open("interactive_session.log","a").write("param_list[:]=%s   # in function interactive-define-ext-constraints \n"%repr(param_list))
		return constrains
	else : 
		'Error : Entry is not matching any case, restarting constrains procedure'
		continue

#--------------------------------------------------------------------------------------------------------
# Class parameters proxy
#--------------------------------------------------------------------------------------------------------
class Params_and_Functions:
	def __init__(self):
		pass
	def setParams(self, par_array):
		self.par_array=par_array
		self.NusedPar=0
		self.shapes=[]
	def setContribution(self, shape_class = None):
		newshape = shape_class(self.par_array[self.NusedPar:])
		self.shapes.append(newshape)
		self.NusedPar+=newshape.nofMyParams()
	
	def normalise(self,mod):
		norm = mod.convolution_function.values_on_real_points (0)   
		npar=0
		for shape in self.shapes:
			self.par_array[npar+1] /= norm # or by  mod.reso_ifft_max : must be the same
			npar += shape.nofMyParams()

	def  maxheight(self):
		maxres=0
		for shape in self.shapes:
			maxres=max(maxres,shape.get_Height())
		return maxres

	def print_params_h5(self, T,sigma=None, File=None,datasetname=None ):
		if sigma == None:
			sigma =  list(np.zeros(self.par_array.shape ))
		
		File = h5py.File(File)
		group =  File .require_group(datasetname)
		group.require_dataset("temperature", shape=(1,), dtype=np.float32)
		group['temperature'][:] = T
		icontribution = 0 
		ipar=0
		for contribution in self.shapes:
			
			npars = contribution.nofMyParams()
			parnames = contribution.parNames()
			if icontribution==0:
				elC = contribution.get_Center()
                                shift=0
			else:
				shift = - elC
			for k in range(npars):
				if parnames[k]=="Center":
					group.require_dataset("Center"+str(icontribution), shape=(1,), dtype=np.float32)
					group.require_dataset("CenterSigma"+str(icontribution), shape=(1,), dtype=np.float32)
					group["Center"+str(icontribution)][:] = self.par_array[ipar+k]+shift
					group["CenterSigma"+str(icontribution)][:] = sigma[ipar+k]
				else:
					group.require_dataset(parnames[k] +str(icontribution), shape=(1,), dtype=np.float32)
					group.require_dataset(parnames[k]+"_sigma"+str(icontribution), shape=(1,), dtype=np.float32)
					group[parnames[k] +str(icontribution)][:]=  self.par_array[ipar+k]
					group[parnames[k]+"_sigma"+str(icontribution) ][:]=   sigma[ipar+k]

			icontribution+=1
			ipar+=npars

		return elC


	def print_params(self, T,sigma=None, File=sys.stdout):
		if sigma == None:
			sigma =  list(np.zeros(self.par_array.shape ))
		
		File.write('-------------------------------------------\n')
		File.write('temperature : %.2f\n'%T)
		File.write('-------------------------------------------\n')
		icontribution=0
		ipar=0
		elC=None
		for contribution in self.shapes:
			
			npars = contribution.nofMyParams()
			parnames = contribution.parNames()
			if icontribution==0:
				elC = contribution.get_Center()
				File.write('Elastic line :\n')
				for k in range(npars):
					# meV2K=1  
					if parnames[k]=="Center":
						File.write( '%s   = %.4f (%.4f) meV (relative to original E grid) \n'%( parnames[k] ,    self.par_array[ipar+k] ,sigma[ipar+k]))
					# File.write( '%s   = %.4f   K (%.4f)   K\n' %( parnames[k] ,    self.par_array[ipar+k]*meV2K ,sigma[ipar+k]*meV2K))
					else:
						File.write( '%s   = %.4f (%.4f) \n'%( parnames[k] ,    self.par_array[ipar+k] ,sigma[ipar+k]))
				File.write('-------------------------------------------\n')
			else:
				File.write('Inelastic line %d :\n'%(icontribution))
				for k in range(npars):
					if parnames[k]=="Center":
						File.write( '%s[%d] = %.4f (%.4f)  meV (relative to elastic) \n'%( parnames[k],icontribution, self.par_array[ipar+k]-elC ,sigma[ipar+k]))
					else:
						File.write( '%s[%d] = %.4f (%.4f)\n'%( parnames[k] ,  icontribution,  self.par_array[ipar+k] ,sigma[ipar+k]))
			icontribution+=1
			ipar+=npars

		File.write('--------------------------------------------------------\n')
		return elC
			
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

def main(argv, SHOW, BLOCK):
	global RECORD, REPLAY
	print_logo()
	fn = argv[1]
	cfg_filename = argv[2]
	detector = argv[3]
#	hdf = h5py.File(fn,'r')
	
	# the allowed keys will be available as cfg members after reading parameter file 
	allowed_keys={"res_param":DictType,"T":FloatType}
	# EXAMPLE OF CFG FILE :
	""""
	# Parameters for resolution function
	# usage : res_param = {detector_number:[mu,wG,wL],...,n:[mun,wGn,wLn]}
	res_param ={
	1:[0.6552,2.604,4.53],
	2:[0.6319,2.603,4.<013],
	..........................
	}
	#Temperature (important : floating type is mandatory)
	T = 297.0
	"""
	
	cfg = read_configuration_file(cfg_filename,allowed_keys= allowed_keys)
	interactive_Entry = True 
	mod=None
	const=None
	CONVOLUTION_METHOD="PSEUDOVOIGT"

	while(1):
		if interactive_Entry:
#			( scan_num , detect_num, 
#			  Ene_array ,Intens_array, Intens_Err) = interactive_extract_data_from_h5(hdf)
			( scan_num , detect_num, 
			  Ene_array ,Intens_array, Intens_Err) = get_data_from_txt(fn,detector)

			if  CONVOLUTION_METHOD=="PSEUDOVOIGT":
				# we build hera a pseudo_voigt for convolution, based on configuration parameters peculiar to the detector 
				mu,gaussian_w,lorentz_w , base_line = cfg.res_param[int(detect_num)]
				convolution_Function = PseudoVoigt( mu,lorentz_w ,gaussian_w , base_line)
			else:
				raise Exception, (" I dont know your convolution model=%s, develop it in the code "%CONVOLUTION_METHOD)

			mod = Model(cfg.T,Ene_array,convolution_Function )

			xy, noel = interactive_GUI_get_init_peak_params(Ene_array,Intens_array)
			const=None
			skip = (xy == [])  # xy is a list : [ e0, height0, e1, height....]
			if noel :   # means : energy range was not containing zero , and elastic peak has not been set  by
                                    # the above GUI routine. We are going to ask for it now and prepend Ec, Ael to xy
				if not REPLAY==0:
					exec(getinstruction(REPLAY))
					xy=[[Ec,Ael]]+xy
				while(1):
					try:
						Ec=float(raw_input('Enter overall scan shift (Ec) : '))
						Ael=float(raw_input('Enter intensity of elastic line (Ael) : '))
						xy=[[Ec,Ael]]+xy
						if RECORD:
							open("interactive_session.log","a").write("Ec,Ael=.%s   # in completion for noel=True \n"%str((Ec,Ael)))

						break
					except:
						print " INPUT ERROR, TRY AGAIN "
						pass
			if not skip:
				# setting up parameter list  : ( position, height, width, position, height.... )
				param_list = np.zeros([len(xy),3    ],"d")
				param_list[:,:2]=xy
				wel,wj =0.1,0.1 #widths of elastic and excitation peaks (initial guess)
				param_list[0,2]=wel
				param_list[1:,2]=wj

				# setting up the model
				params_and_functions = Params_and_Functions()
				params_and_functions.setParams(param_list.flatten())
				# //////////////////////////// contributions
				params_and_functions.setContribution(shape_class=LineModel) # elastic line
				for i in range(len(xy)-1):
					params_and_functions.setContribution(shape_class=LineModel)
				params_and_functions.normalise(mod) 
				print '--------------------------------------------------------------'
				print 'Input parameters :'
				params_and_functions.print_params(cfg.T, File=sys.stdout)

				mod.set_Params_and_Functions(params_and_functions)
			
		else:
			skip=False
		
		if not skip:
			mod.count = 0
			t0=time.time()
			
			if const is not None:
				refined_param, chisq, sigmapar = Gefit.LeastSquaresFit(mod.Ft_I ,params_and_functions.par_array ,
										       constrains=const ,xdata=Ene_array , 
										       ydata= Intens_array,
										       sigmadata=Intens_Err)
			else:
				# Plot(mod,params_and_functions.par_array,Ene_array,Intens_array, Intens_Err, show_graph=1)
				# plt.show()

				const = default_build_constrains(params_and_functions ,position=3,intensity=2,irange=[0.,params_and_functions.maxheight()*2],width=3)

				refined_param, chisq, sigmapar = Gefit.LeastSquaresFit(mod.Ft_I ,params_and_functions.par_array ,
										       constrains=const ,xdata=Ene_array , 
										       ydata= Intens_array,
										       sigmadata=Intens_Err)

				const = default_build_constrains(params_and_functions,position=2, # refined_param[0] est suppose etre le centre de la ligne elastic
							  prange=[0+(refined_param[0]),Ene_array[-1]*1.2],intensity=2,irange=[0.,params_and_functions.maxheight()*2],width=2,wrange=[0.,2.5])#XXX

				refined_param, chisq, sigmapar = Gefit.LeastSquaresFit(mod.Ft_I ,params_and_functions.par_array ,
										       constrains=const ,xdata=Ene_array , 
										       ydata= Intens_array,
										       sigmadata=Intens_Err)
			t1=time.time()
			print 'Exec time for calculation : %f'%(t1-t0)
			print 'number of iteration in Levenberg-Marquardt : %d'%mod.count
			print 'Exec time per iteration : %f'%((t1-t0)/mod.count)
			mod.params_and_functions.par_array[:] =  refined_param  # Note : we update internal values. We dont change the object reference value 
			print 'root-mean-square deviation : %.4f'%  (np.sqrt(np.sum(((Intens_array-mod.Ft_I(refined_param,Ene_array ))**2)))/len(Ene_array))

			plotted_datas = Plot(mod,refined_param,Ene_array,Intens_array,Intens_Err , show_graph=SHOW) # this function would be used also just
			# for grabbing data columns :  Ldat = [E-Center , A, Err,tot, el, inel1, inel2 ...]

			print '--------------------------------------------------------------'
			print 'Output parameters :'
			params_and_functions.print_params(cfg.T,sigmapar, File=sys.stdout)   # on the screen
#			output_dir =  fn[:-3] + '_fit'
#			output_stripped_name  = fn[:-3]
			output_dir =  fn[:-4] + '_fit'
			output_stripped_name  = fn[:-4]
			if not os.path.exists(output_dir):
				os.mkdir(output_dir)
			out_name = '%s/%s.h5'%(output_dir,output_stripped_name )	
			datasetname = "data_%s_%s"%(scan_num,detect_num )

#			out = open('%s/%s_%s_%s.param'%(output_dir,output_stripped_name ,scan_num,detect_num),'w')
			out = open('%s/%s_%s.param'%(output_dir,output_stripped_name, detect_num),'w')
			elC = params_and_functions.print_params_h5(cfg.T,sigmapar, File=out_name,datasetname= datasetname)  # on file
			elC = params_and_functions.print_params(cfg.T,sigmapar, File=out)  # on file
			out=None
			cs = np.column_stack(plotted_datas)
			cs[:,0] -= elC
			np.savetxt('%s/%s_%s.dat'%(output_dir,output_stripped_name,detect_num), cs , fmt='%12.4e', delimiter=' ')

			file_print ( output_dir, output_stripped_name       ,  scan_num , detect_num)

			try:
				plt.show(block=BLOCK)
			except:
				plt.show()

			interactive_Entry=True
			if not REPLAY==0:
				exec(getinstruction(REPLAY))
			else:
				r = raw_input('Would you like to fit another spectrum (y) or (n) default : [y] ?\nor change temperature (t) ?\nor refine again the previous fit with different constrains (r) ?\n')
				if RECORD:
					open("interactive_session.log","a").write("r='%s'   # in asking Would you like to fit another spectrum :y,n,r,t \n"%r)
			if r in ['n','N']:
				print 'Bye Bye'
				break
			elif r in ['t','T']:
				
				if not REPLAY==0:
					exec(getinstruction(REPLAY))
				else:
					T = raw_input('Temperature ? [297.0]: ')
					if RECORD :
						open("interactive_session.log","a").write("T='%s'   # in asking Temperature ? [297.0] \n"%T)

				if T == '':
					cfg.T = 297.0
				else :
					cfg.T = float(T)
			elif r in ['r','R']:
				const = interactive_define_ext_constrains(params_and_functions,const) # this function might change internal values
				                                                                      # of params_and_functions.par_array
				interactive_Entry=False
			else:
				pass # will continue as default
			plt.close()

	# now we exit from the main
	# hdf.close()

#--------------------------------------------------------------------------------------------------------
def getinstruction(REPLAY):
	s=""
	while("#" not in s):
		s=s+REPLAY.readline()
	return s

if __name__ == '__main__':
	RECORD=0
	REPLAY=0
	SHOW=1
	BLOCK=0
	argv = sys.argv
	if len(argv) in [4,5]:
		if len(argv)==5:
			if sys.argv[4]=="RECORD":
				open("interactive_session.log","w")
				RECORD=1
			elif "REPLAY" in sys.argv[4]:
				print   sys.argv[4]
				exec(sys.argv[4])
				if "REPLAYSHOW" in sys.argv[4] :
					REPLAY=open( REPLAYSHOW,"r")
					SHOW=1
					BLOCK=1
				else:
					REPLAY=open( REPLAY,"r")
					SHOW=0
			else:
				raise Exception, " Thrid argument, if present ,  must be either RECORD or REPLAY REPLAYSHOW" 
		sys.exit(main(argv, SHOW, BLOCK))
	else:
		print '\nusage : ixs_fitter data_file.dat resolution.param detector\n'


		
