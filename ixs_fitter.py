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

	def ft_and_derivatives(self, reciprocal_grid, real_grid_origin, Stokes=True ):
		result=np.zeros(   [  len(reciprocal_grid),    1+self.nofMyParams()  ]   )
		if Stokes:
			Center =  self.Center - real_grid_origin
		else:
			Center = -self.Center - real_grid_origin

		result [:,2] = ft_line( reciprocal_grid, Center ,1.0, self.W    )
		result [:,0] = self.Height*result [:,2]
		result [:,1] = self.Height*result [:,1]*( -1.0j*reciprocal_grid)
		result [:,3] = self.Height*result [:,1]*( -1.0/2*abs(reciprocal_grid))
		

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

# --------------------------------------------------------------------------------------------------------
#  Convolution Models ( all have the same interface(excepted init function )
# ---------------------------------------------------------------------------------------------------------

class PseudoVoigt:
	def __init__(self ,  mu,lorentz_w ,gaussian_w ):
		self.mu=mu
		self.gaussian_w=gaussian_w
		self.lorentz_w=lorentz_w

	def safe_scale_length(self):
		return  min(self.lorentz_w/5,self.gaussian_w) 
	def safe_margin(self):
		 return 10*self.lorentz_w+3*self.gaussian_w
	 
	def values_on_real_points(self,x ):
		 return   pseudo_voigt(x ,  self.mu,self.lorentz_w ,self.gaussian_w )
	def values_on_reciprocal_points(self,x ):
		 return   ft_pseudo_voigt(x ,  self.mu,self.lorentz_w ,self.gaussian_w )
	 



#--------------------------------------------------------------------------------------------------------
# Fitting function Model
#--------------------------------------------------------------------------------------------------------

class Model:
	def __init__(self,T,E,res_param, convolution_function):

		self.convolution_function = convolution_function
		safe_step = self.convolution_function.safe_scale_length() 
		safe_margin = self.convolution_function.safe_margin() 
		

		self.T = T
		self.kb = 1.3806504e-23/1.602176462e-22
		self.E = E
		self.orig_xmin= E[0]  - safe_margin
		self.orig_xmax= E[-1] + safe_margin
		self.xmin= E[0]  - safe_margin
		self.xmax= E[-1] + safe_margin

		self.norm = (self.xmax - self.xmin)
		self.npts = sup_pow_2(self.norm/(safe_step))

		
		self.Ech = np.arange(self.xmin,self.xmax,float((self.xmax-self.xmin))/self.npts)
		self.Sch = np.fft.fftshift(np.arange(-self.npts*np.pi/self.norm,self.npts*np.pi/self.norm,2*np.pi/self.norm))
		
		self.resolution_fft = self.convolution_function.values_on_reciprocal_points(self.Sch)  

		self.count=0
	

	def	set_Params_and_Functions(self, params_and_functions):
		self.params_and_functions=params_and_functions

#--------------------------------------------------------------------------------------------------------

	
	def Ft_I(self,p, E, interpolation=1, mask=None, convolution=1): 
		""" Interface """
		
		# update variables in self.params_and_functions
		self.params_and_functions.par_array[:] = p   # Note : we update internal values. We dont change the object reference value 
		if mask is None:
			mask=np.ones(len(self.params_and_functions.shapes) )

		icontribution=0
		ipar=0
		for contribution in self.params_and_functions.shapes:
			npars = contribution.nofMyParams()
			parnames = contribution.parNames()
			result=0.0
			fact=1.0*mask[icontribution]
			if icontribution==0:
				value_and_deri =  contribution.ft_and_derivatives( reciprocal_grid.Sch, self.xmin ,self.T, Stokes=0 )
				result=result+value_and_deri [0]*fact  # so far we exploit only function itself not derivatives..
				el_center = contribution.get_Center()
			else:
				value_and_deri =  contribution.ft_and_derivatives( reciprocal_grid.Sch, self.xmin , Stokes=+1 )
				result=result+value_and_deri [0]*fact

				inel_center = contribution.get_Center()
				fact = fact* np.exp(-(inel_center-el_center)/(self.kb*self.T))
				value_and_deri =  contribution.ft_and_derivatives( reciprocal_grid.Sch, self.xmin , Stokes=-1 )*fact

				result=result+value_and_deri [0]

			icontribution+=1
			ipar+=npars


		if convolution:
			result = np.fft.ifft(result*   self.resolution_fft )*npts
		else:
			result = np.fft.ifft(result  ) 

		result = result.real
		if interpolation:
			return np.interp(E,Ech,result)
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
	indicators_limit = {"Center":prange ,"Height":irange,"Width":wrange }

	c = list(np.zeros((3, len(params_and_functions.par_array)  )))
	icontribution=0
	ipar=0
	for contribution in self.shapes:
		npars = contribution.nofMyParams()
		parnames = contribution.parNames()
		if icontribution==0:
			# 'elastic line : everything is free'			
			for k in range(npars):
				c[0][ipar+k],c[1][ipar+k], c[2][ipar+k] =0 # Free
		else:
			# 'inelastic line %d :'%(icontribution+1)
			for k in range(npars):
				if indicators.has_key( parnames[k]) and indicators[parnames[k]]: 
					c[0][ipar+k]=indicators[parnames[k]]
					if indicators[parnames[k]]==2:
						c[1][ipar+k],c[2][ipar+k] =indicators_limits[parnames[k]]
					else:
						c[0][ipar+k],c[1][ipar+k], c[2][ipar+k] =0 # Free
		icontribution+=1
		ipar+=npars
	return c
    	
#--------------------------------------------------------------------------------------------------------	
	
def Plot(mod,par_array,E,A, Err, show_graph=1):

	Ech = mod.Ech
	Sch = mod.Sch
	mod.params_and_functions.par_array[:] = par_array   # Note : we update internal values. We dont change the object reference value 
	Center = mod.params_and_functions.shapes[0].get_Center()
	if show_graph : plt.plot(E-Center,A,'blue',label='Experimental data')# plot : exp data

	plt.xlabel("Energy")
	plt.ylabel("Intensity")
	plt.title("IXS Spectrum Fitting")

	mask=np.zeros(len(mod.params_and_functions.shapes) )

	Ldat = [E-Center , A, Err]

	mask[:]=1	
	total_model = mod.Ft_I(param, Ech, interpolation=0, mask=mask) # with interpolation=0 Ech is dummy
	if show_graph : plt.plot(Ech-Center,App,'red',label='Fitted model')	

	Ldat.append(mod.Ft_I(param, E, interpolation=1, mask=mask))


	icontribution=0
	for contribution in mod.params_and_functions.shapes:
		mask[:]=0
		mask[icontribution=1]
		partial_model = mod.Ft_I(param, Ech, interpolation=0, mask=mask) # with interpolation=0 Ech is dummy
		if icontribution==0:
			if show_graph : plt.plot(Ech-Center,partial_model,'Cyan',label='Inelastic Contrib %d'%(i+1))	
		else:
			if show_graph : plt.plot(Ech-Center,partial_model,'magenta',label='Elastic Contrib.')

		partial_model = mod.Ft_I(param, E, interpolation=1, mask=mask) # with interpolation=0 Ech is dummy
		Ldat.append(partial_model)
		icontribution+=1

	fig  = plt.gcf()
	fig.set_size_inches(8*1.4,6*1.2,forward=True) 
	ax = fig.gca()
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
	plt.legend(loc='upper center', bbox_to_anchor=(1.1, 1), fancybox=True, shadow=True, ncol=1)
	return Ldat
	
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

def build_param_name_dict(params_and_functions):
	icontribution=0
	ipar=0
	res={}
	for contribution in self.shapes:
		npars = contribution.nofMyParams()
		parnames = contribution.parNames()
		if icontribution==0:
			# 'elastic line :'
			for k in range(npars):
				res[ipar+k] = parnames[k]+"_el"
		else:
			# 'inelastic line %d :'%(icontribution+1)
			res[ipar+k] = parnames[k]+("_%d"%icontribution)
		icontribution+=1
		ipar+=npars
	return res

def interactive_define_ext_constrains(params_and_functions,constrains):
  consttype = {0 : 'Free' ,1 : 'Positive' ,2 : 'Quoted' ,3 : 'Fixed'}
  param_list = params_and_functions.par_array
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
	def setContribution(self, shape_class = shape_class):
		newshape = shape_class(par_array[self.NusedPar:])
		self.shapes.append(newshape)
		self.NusedPar+=newshape.nofMyParams()
	
	def normalise(self,mod):
		norm = mod.convolution_function.values_on_real_points (0)   
		npar=self.shapes[0].nofMyParams()
		for shape in self.shapes[1:]:
			self.par_array[npar] *=  norm
			npar += shape.nofMyParams()

	def print_params(self, T,sigma=None, File=sys.stdout):
		if sigma == None:
			sigma =  list(np.zeros(self.par_array.shape ))

		
		print'-------------------------------------------'
		print'temperature : %.2f'%T
		print'-------------------------------------------'
		icontribution=0
		ipar=0
		for contribution in self.shapes:
			npars = contribution.nofMyParams()
			parnames = contribution.parNames()
			if icontribution==0:
				print'elastic line :'
				for k in range(npars):
					print 'Elastic %s   = %.4f (%.4f)'%( parnames[k] ,    self.par_array[ipar+k] ,sigma[ipar+k])
				print'-------------------------------------------'
			else:
				print'inelastic line %d :'%(icontribution+1)
				for k in range(npars):
					print '%s  inelastic[%d] = %.4f (%.4f)'%( parnames[k] ,  icontribution +1,  self.par_array[ipar+k] ,sigma[ipar+k])
			icontribution+=1
			ipar+=npars

		print'--------------------------------------------------------'
			
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
	CONVOLUTION_METHOD="PSEUDOVOIGT"

	while(1):
		if interactive_Entry:
			( scan_num , detect_num, 
			  Ene_array ,Intens_array, Intens_Err) = interactive_extract_data_from_h5(hdf)
			
			if  CONVOLUTION_METHOD=="PSEUDOVOIGT":
				# we build hera a pseudo_voigt for convolution, based on configuration parameters peculiar to the detector 
				mu,gaussian_w,lorentz_w = cfg.res_param[int(detect_num)]
				convolution_Function = PseudoVoigt( mu,lorentz_w ,gaussian_w )
			else:
				raise Exception, (" I dont know your convolution model=%s, develop it it the code "%CONVOLUTION_METHOD)

			mod = Model(cfg.T,Ene_array,cfg.res_param,convolution_Function )

			xy, noel = GUI_get_init_peak_params(Ene_array,Intens_array)
			skip = (xy == [])  # xy is a list : [ e0, height0, e1, height....]
			if noel :   # means : energy range was not containing zero , and elastic peak has not been set  by
                                    # the above GUI routine. We are going to ask for it now and prepend Ec, Ael to xy
				while(1):
					try:
						Ec=float(raw_input('Enter overall scan shift (Ec) : '))
						Ael=float(raw_input('Enter intensity of elastic line (Ael) : '))
						xy=[[Ec,Ael]]+xy
						break
					except:
						print " INPUT ERROR, TRY AGAIN "
						pass
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
			params_and_functions.setContribution(shape_class=Line) # elastic line
			for i in range(len(xy)-1):
				params_and_functions.setContribution(shape_class=Line)
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
				const1 = default_build_constrains(params_and_functions ,position=3,intensity=2,irange=[0.,max(Intens_array)*1.5],width=3)
				refined_param, chisq, sigmapar = Gefit.LeastSquaresFit(mod.Ft_I ,params_and_functions.par_array ,
										       constrains=const1 ,xdata=Ene_array , 
										       ydata= Intens_array,
										       sigmadata=Intens_Err)
				const2 = default_build_constrains(params_and_functions,position=2, # refined_param[0] est supposedly le centre de elastic
							  prange=[0+refined_param[0],Ene_array[-1]*1.2],intensity=2,irange=[0.,max(Intens_array)*1.5],width=2,wrange=[0.,2.5])#XXX 
				refined_param, chisq, sigmapar = Gefit.LeastSquaresFit(mod.Ft_I ,params_and_functions.par_array ,
										       constrains=const2 ,xdata=Ene_array , 
										       ydata= Intens_array,
										       sigmadata=Intens_Err)
			t1=time.time()
			print 'Exec time for calculation : %f'%(t1-t0)
			print 'number of iteration in Levenberg-Marquardt : %d'%mod.count
			print 'Exec time per iteration : %f'%((t1-t0)/mod.count)
			mod.params_and_functions.par_array[:] =  refined_param  # Note : we update internal values. We dont change the object reference value 
			print 'root-mean-square deviation : %.4f'%  np.sqrt(np.sum(((Intens_array-mod.Ft_I(refined_param,Ene_array ))**2)))/len(Ene_array)

			plotted_datas = Plot(mod,refined_param,Ene_array,Intens_array, Err, show_graph=1) # this function xould be used also just
			# for grabbing data columns :  Ldat = [E-Center , A, Err,tot, el, inel1, inel2 ...]

			print '--------------------------------------------------------------'
			print 'Output parameters :'
			params_and_functions.print_params(cfg.T,sigma, File=sys.stdout)   # on the screen
			output_dir, output_stripped_name  = get_dotstripped_path_name(hdf.filename)
			if not os.path.exists(output_dir):
				os.mkdir(dirfn)
			out = open('%s/%s_%s_%s.param'%(output_dir,output_stripped_name ,scan_num,detect_num),'w')
			params_and_functions.print_params(cfg.T,sigma, File=out)  # on file
			out=None

			np.savetxt('%s/%s_%s_%s.dat'%(dirfn,fn,s,d), np.array(Ldat), fmt='%14.4f', delimiter=' ')

			file_print ( output_dir, output_stripped_name       ,  scan_num , detect_num)

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
				const = interactive_define_ext_constrains(params_and_functions,const) # this function might change internal values
				                                                          # of params_and_functions.par_array
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


		
