#!/usr/bin/env python
#------------------------------------------------------------------------------
# File: dnnwrite.py
# Purpose: Write a mlp-function created by sklearn or fbm
# Created: 03-Mar-2018 Harrison B. Prosper
#          21-Apr-2018 HBP adapt for fbm
#          24-Apr-2018 HBP fix Python interface
#------------------------------------------------------------------------------
import os, sys, re, optparse
import pickle as pc
from math import *
from string import *
from time   import *
try:
    from sklearn.externals import joblib
except:
    pass
#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------
USAGE = '''
Usage:
  dnnwrite.py <options> mlp-file-name

  options:
       -p   fbm/scikit-learn [fbm,100]
       -o   output file name [name of mlp-file + '.cc']
'''
VERSION="v1.0.2"
#------------------------------------------------------------------------------
# C++ template
#------------------------------------------------------------------------------
HEADER= '''#ifndef %(basename)s_h
#define %(basename)s_h
// ----------------------------------------------------------------------
// File:    %(basename)s%(inputs)s
// Created: %(date)s by dnnwrite.py %(version)s
// ----------------------------------------------------------------------
#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
// ----------------------------------------------------------------------
struct %(basename)s
{
  struct Layer
  {
    double (*activation)(double); 
    std::vector<double> B;
    std::vector<std::vector<double> > W;    
  };

  inline static double identity(double x) { return x; }
  inline static double relu(double x)     { return fmax(0, x); }
  inline static double logistic(double x) { return 1.0/(1 + exp(-x)); }
  inline static double sigmoid(double x)  { return tanh(x); }

  struct NetWeights
  {
    NetWeights() {}

    NetWeights(std::vector<Layer>&  weights_)
     : weights(weights_),
       outputs(std::vector<double>(%(maxwidth)s)),
       I(std::vector<double>(%(maxwidth)s)) {}

    ~NetWeights() {}

    std::vector<Layer>  weights;
    std::vector<double> outputs;
    std::vector<double> I;
  };

  %(basename)s();
  ~%(basename)s();
  %(rtype)s operator()(std::vector<double>& inputs);
  %(rtype)s operator()(double* inputs);

#ifdef WITH_PYTHON
  %(rtype)s operator()(PyObject* o);
#endif

  void compute(int netid=0);
%(softmaxdef)s
%(selectdef)s
  int ninputs;
  int noutputs;
  int maxwidth;
  int first;
  int last;

  std::vector<double> mean;
  std::vector<double> scale;
  std::vector<double> I;
  std::vector<NetWeights> nn;
  std::vector<double> p;
};
#endif
'''

SOFTMAXDEF = '''
  void softmax(int netid=0);
'''

SOFTMAXIMPL = ''' 
void %(basename)s::softmax(int netid)
{
  NetWeights& nw = nn[netid]; // NB: a reference, not a copy!
  double softsum = 0;
  for(size_t c=0; c < nw.outputs.size(); c++)
    {
      softsum += nw.I[c];
      nw.outputs[c] = nw.I[c];
    }
  for(size_t c=0; c < nw.outputs.size(); c++) nw.outputs[c] /= softsum;	
}'''


SELECTDEF = '''
  void select(int first_=-1, int last_=-1);
'''

SELECTIMPL = '''
void %(basename)s::select(int first_, int last_)
{
  first = first_;
  last  = last_;
  int maxid = (int)nn.size() - 1;
  if ( first < 0 )     first = 0;
  if ( first > maxid ) first = maxid;
  if ( last  > maxid ) last  = maxid;
  if ( last  < first ) last  = first;
}
'''


SOURCE='''// ----------------------------------------------------------------------
// File:    %(name)s%(inputs)s
// Created: %(date)s by dnnwrite.py %(version)s
// ----------------------------------------------------------------------
#include "%(basename)s.h"
// ----------------------------------------------------------------------
%(basename)s::%(basename)s()
  : ninputs(%(ninputs)d),
    noutputs(%(noutputs)d),
    maxwidth(%(maxwidth)d),
    first(0),
    last(%(maxid)s),
    mean(std::vector<double>(%(ninputs)d)),
    scale(std::vector<double>(%(ninputs)d)),
    I(std::vector<double>(%(ninputs)d)),
    nn(std::vector<NetWeights>()),
    p(std::vector<double>(%(nnetworks)d))
{
%(cppsrc)s
}

%(basename)s::~%(basename)s() {}

%(rtype)s %(basename)s::operator()(std::vector<double>& inputs)
{
  // standardize inputs
  for(size_t c=0; c < mean.size(); c++) I[c] = (inputs[c] - mean[c]) / scale[c];
  std::vector<double> y(%(noutputs)d, 0);
  int j=0;
  for(int netid=first; netid<=last; netid++)
    {
      NetWeights& nw = nn[netid];
      compute(netid);
      for(int c=0; c < noutputs; c++) y[c] += nw.outputs[c];
      p[j] = nw.outputs[0]; j++;
    }
  int N = last - first + 1;
  for(int c=0; c < noutputs; c++) y[c] /= N;
  return %(output)s
}

%(rtype)s %(basename)s::operator()(double* inputs_)
{
  std::vector<double> inputs(%(ninputs)d);
  for(size_t c=0; c < inputs.size(); c++) inputs[c] = inputs_[c];
  return (*this)(inputs);
}

#ifdef WITH_PYTHON
%(rtype)s %(basename)s::operator()(PyObject* o)
{
  std::vector<double> inputs(ninputs, 0);
  if ( PySequence_Check(o) )
    {
      int n = PySequence_Length(o);
      if (n != ninputs)
        {
          std::cout << "%(basename)s - sequence argument has wrong length: "
                    << n << std::endl;
          exit(0);
        }
      for(int c=0; c < n; c++)
        { 
          PyObject* item = PySequence_GetItem(o, c);
          // assume we have floats
          inputs[c] = PyFloat_AsDouble(item);
          // since we own item and no longer need it,
          // we must decrement ownership (reference) count
          Py_DECREF(item);
        }
    }
  else
    {
      std::cout << "%(basename)s - argument must be either a list or a tuple" 
                << std::endl;
      exit(0);
    }
  return (*this)(inputs);    
}
#endif
%(softmaximpl)s
%(selectimpl)s
void %(basename)s::compute(int netid)
{
  NetWeights& nw = nn[netid];

  for(size_t layer=0; layer < nw.weights.size(); layer++)
    {
      std::vector<double>& B = nw.weights[layer].B; // reference not a copy!
      std::vector<std::vector<double> >& W = nw.weights[layer].W;
      for(size_t j=0; j < B.size(); j++)
        {
          nw.outputs[j] = B[j];
          if ( layer == 0 )
            for(size_t i=0; i < W.size(); i++) nw.outputs[j] += I[i] * W[i][j];
          else
            for(size_t i=0; i < W.size(); i++) nw.outputs[j] += nw.I[i] * W[i][j];
          nw.outputs[j] = nw.weights[layer].activation(nw.outputs[j]);
        }
      for(size_t j=0; j < B.size(); j++) nw.I[j] = nw.outputs[j];
     }
}
'''

BUFFERS = '''%(means)s

%(scales)s

  { // NETWORK: %(netid)s

    std::vector<Layer>  weights;
%(weightimpl)s
    nn.push_back( NetWeights(weights) );
  }
'''

WEIGHTS = '''
    {       // layer %(layer)d
      weights.push_back(Layer());
      weights.back().activation = %(activation)s;
      weights.back().B = std::vector<double>(%(nout)d);    
      std::vector<double>& B = weights.back().B;
%(biases)s

      weights.back().W = std::vector<std::vector<double> >
        (%(ninp)d, std::vector<double>(%(nout)d));
      std::vector<std::vector<double> >& W = weights.back().W;
%(weights)s
     }
'''
#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------
def usage():
    sys.exit(USAGE)

def quit(s):
    sys.exit("\n**error** %s" % s)

def nameonly(s):
    import posixpath
    return posixpath.splitext(posixpath.split(s)[1])[0]

def decodeCommandline():
    parser = optparse.OptionParser(usage=USAGE, version=VERSION)

    parser.add_option("-n", "--netname",
                      action="store",
                      dest="netname",
                      type="string",
                      default='',
                      help="name of neural network")

    parser.add_option("-p", "--package",
                      action="store",
                      dest="package",
                      type="string",
                      default='',
                      help="name of mlp package")
    
    options, args = parser.parse_args()
    if len(args) == 0:
        sys.exit(USAGE)

    stripno = re.compile('^[0-9_]+')
    dnnfilename = args[0]
    netname = options.netname

    if netname == '':
        name = nameonly(dnnfilename)
    else:
        name = nameonly(netname)
    name = stripno.sub('', name)
    
    package = options.package
    
    if package == '':
        t = split(dnnfilename, '.')
        if t[-1] == 'bin':
            package = 'fbm,100'
        else:
            package = 'scikit-learn'
            
    return (name, dnnfilename, package, args[1:])
#------------------------------------------------------------------------------
class Parameter:
    def __init__(self, activation='tanh'):
        # B and W are lists of arrays by layer
        # B[layer][j],    j = 1, nout/layer
        # W[layer][i][j]  i = 1, ninp/layer
        self.intercepts_ = []
        self.coefs_      = []
        self.activation_ = activation
        self.out_activation_ = ''
        
    def __del__(self):
        pass

class Scaler:
    def __init__(self, name, mean, scale):
        self.name_  = name
        self.mean_  = mean
        self.scale_ = scale
        
    def __del__(self):
        pass
    
class MLPinfo:
    def __init__(self, mlpfilename, package):
        if package[0] == 's':
            try:
                t = joblib.load(mlpfilename)
            except:
                quit('unable to load %s' % mlpfilename)

            # load trained MLP and optional scaler object
            self.name = None
            if len(t) > 1:
                self.mlp   = t[0]
                self.mean  = t[1].mean_
                self.scale = t[1].scale_
            else:
                self.mlp   = t
                self.mean  = [0.0]*len(mlp.coefs_[0])
                self.scale = [1.0]*len(mlp.coefs_[0])
        else:
            # assume we have an ensemble of NN
            self.mlp, t = pc.load(open(mlpfilename, 'rb'))
            self.name   = t.name_
            self.mean   = t.mean_
            self.scale  = t.scale_

    def __del__(self):
        pass
    
    def __call__(self, netid=-1):
        if netid < 0:
            mlp = self.mlp
            activation = mlp.get_params()['activation']
            out_activation = mlp.out_activation_
        else:
            mlp = self.mlp[netid]
            activation = mlp.activation_
            out_activation = mlp.out_activation_
            
        B = []
        W = []
        nlayers = len(mlp.intercepts_)
        func  = []
        for layer in range(nlayers):   
            func.append(activation)
            B.append(mlp.intercepts_[layer])
            W.append(mlp.coefs_[layer])
        func[-1] = out_activation

        # width of network
        width = max(map(len, mlp.intercepts_))
        
        return (B, W, func, self.name, self.mean, self.scale, width)
#------------------------------------------------------------------------------    
def writeCPP(names, details):
    B, W, activations, iname, mean, scale, maxwidth = details
    
    if 'softmax' in activations:
        names['softmaximpl'] = SOFTMAXIMPL % names
        names['softmaxdef']  = SOFTMAXDEF  % names

    if names['nnetworks'] > 1:
        names['selectimpl'] = SELECTIMPL   % names
        names['selectdef']  = SELECTDEF    % names
        
    # B and W are lists of arrays by layer
    # B[layer][j],    j = 1, nout/layer
    # W[layer][i][j]  i = 1, ninp/layer
    
    ninputs = len(W[0]) # length of layer 0 gives range of "i"
    nlayers = len(B)
    noutputs= len(W[-1][0]) # last year, first node gives range of "j"
        
    names['ninputs'] = ninputs
    names['nlayers'] = len(B)
    names['maxwidth'] = maxwidth
    names['maxid'] = names['nnetworks'] - 1    
    if len(B[-1]) == 1:
        names['rtype'] = 'double'
        names['noutputs'] = 1
        names['output'] = 'y[0];'
    else:
        names['rtype'] = 'std::vector<double>&'
        names['noutputs'] = noutputs
        names['output'] = 'y;'
    
    # write out C++ code
    tab2 = ' '*2
    tab4 = ' '*4
    tab6 = ' '*6

    if names['netid'] == 0:
        
        # MEANS
        means = []
        rec  = tab2
        means.append(rec)
        for i in range(ninputs):
            s = 'mean[%d]=%12.5e; ' % (i, mean[i])
            if len(rec + s) < 80:
                rec += s
                means[-1] = rec
            else:
                rec = tab2 + s
                means.append(rec)
        names['means'] = joinfields(means, '\n')

        # SCALES
        scales = []
        rec  = tab2
        scales.append(rec)
        for i in range(ninputs):
            s = 'scale[%d]=%12.5e; ' % (i, scale[i])
            if len(rec + s) < 80:
                rec += s
                scales[-1] = rec
            else:
                rec = tab2 + s
                scales.append(rec)
        names['scales'] = joinfields(scales, '\n')    
    else:
        names['means'] = ''
        names['scales'] = ''
        
    # BIASES & WEIGHTS
    cppsrc = ''
    for layer in range(nlayers):
        ninp = len(W[layer])
        nout = len(W[layer][0])
        names['layer'] = layer
        names['activation'] = activations[layer]
        names['nout']  = nout
        names['ninp']  = ninp

        biases = []
        rec = tab6
        biases.append(rec)
        for j in range(nout):
            s = 'B[%d]=%12.5e; ' % (j, B[layer][j])
            if len(rec + s) < 80:
                rec += s
                biases[-1] = rec
            else:
                rec = tab6 + s
                biases.append(rec)
        names['biases'] = joinfields(biases, '\n') 

        weights= []
        rec = tab6
        weights.append(rec)
        for i in range(ninp):
            for j in range(nout):
                s = 'W[%d][%d]=%12.5e; ' % (i, j, W[layer][i][j])
                if len(rec + s) < 80:
                    rec += s
                    weights[-1] = rec
                else:
                    rec = tab6 + s
                    weights.append(rec)
        names['weights'] = joinfields(weights, '\n')        
        
        rec = WEIGHTS % names
        cppsrc += rec
        
    names['weightimpl'] = cppsrc
    cppsrc = BUFFERS % names
    return cppsrc
#------------------------------------------------------------------------------
def writeLinkdefAndMakefile(names, extranames=[]):
    # --------------------------------------------------------------------------
    # write linkdef
    # --------------------------------------------------------------------------
    srcs = names['srcs']
    rec  = '#pragma link C++ class %s+;' % names['basename']
    for namen in extranames:
       rec  += '#pragma link C++ class %s+;' % namen
       srcs += '\t$(srcdir)/%s.cc \\\n' % namen
    srcs = srcs[:-2]
    
    record = '''
#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link C++ class std::pair<double, std::string>+;
#pragma link C++ class std::vector<std::pair<double, std::string> >+;
%s
#endif
''' % rec
    
    outfilename = '%(dnndir)s/include/linkdef.h' % names
    print '\t=> creating file: %s' % outfilename    
    open(outfilename, 'w').write(record)

    # --------------------------------------------------------------------------
    # write makefile
    # --------------------------------------------------------------------------
    names['srcs'] = srcs
    names['percent'] = '%'
    
    record = '''# ------------------------------------------------------------------------------
# build shared library lib%(libname)s
# created: %(date)s by dnnwrite.py %(version)s
# ------------------------------------------------------------------------------
ifndef ROOTSYS
	$(error *** Please set up Root)
endif
# ----------------------------------------------------------------------------
dnndir  := %(dnndir)s
name	:= %(libname)s
srcdir	:= src
libdir	:= lib
incdir  := include

# create directories 
$(shell mkdir -p $(srcdir) $(libdir) $(incdir))

# get lists of sources
SRCS:= %(srcs)s

CINTSRCS:= $(wildcard $(srcdir)/*_dict.cc)

OTHERSRCS:= $(filter-out $(CINTSRCS) $(SRCS),$(wildcard $(srcdir)/*.cc))

# list of dictionaries to be created
DICTIONARIES:= $(SRCS:.cc=_dict.cc)

# get list of objects
OBJECTS		:= $(SRCS:.cc=.o) $(OTHERSRCS:.cc=.o) $(DICTIONARIES:.cc=.o)
# ----------------------------------------------------------------------------
ROOTCINT	:= rootcint
CXX\t:= g++
LD\t:= g++
CPPFLAGS:= -I. -I$(incdir) $(shell root-config --cflags)

ifeq ($(WITH_PYTHON),1)
CPPFLAGS+= -DWITH_PYTHON $(shell python-config --cflags)
endif

CXXFLAGS:= -O2 -Wall -fPIC -ansi -Wshadow -Wextra 
LDFLAGS	:=
# ----------------------------------------------------------------------------
# which operating system?
OS := $(shell uname -s)
ifeq ($(OS),Darwin)
\tLDFLAGS += -dynamiclib
\tLDEXT	:= .so
else
\tLDFLAGS	+= -shared
\tLDEXT	:= .so
endif
LDFLAGS += $(shell root-config --ldflags) -Wl,-rpath,$(ROOTSYS)/lib

ifeq ($(WITH_PYTHON),1)
LDFLAGS += $(shell python-config --ldflags)
endif

# libraries
LIBS	:= $(shell root-config --libs)
LIBRARY	:= $(libdir)/lib$(name)$(LDEXT)
# ----------------------------------------------------------------------------
all: $(LIBRARY)

$(LIBRARY)\t: $(OBJECTS)
\t@echo ""
\t@echo "=> Linking shared library $@"
\t$(LD) $(LDFLAGS) $^ $(LIBS)  -o $@

$(OBJECTS)\t: %(percent)s.o	: %(percent)s.cc
\t@echo ""
\t@echo "=> Compiling $<"
\t$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(DICTIONARIES)	: $(srcdir)/%(percent)s_dict.cc	: $(incdir)/%(percent)s.h $(incdir)/linkdef.h
\t@echo ""
\t@echo "=> Building dictionary $@"
\t$(ROOTCINT) -f $@ -c $(CPPFLAGS) $+
\tfind $(srcdir) -name "*.pcm" -exec mv {} $(libdir) \;

tidy:
\trm -rf $(srcdir)/*_dict*.* $(srcdir)/*.o 

clean:
\trm -rf $(libdir)/* $(srcdir)/*_dict*.* $(srcdir)/*.o 
''' % names
    
    outfilename = '%(dnndir)s/Makefile' % names
    print '\t=> creating file: %s' % outfilename    
    open(outfilename, 'w').write(record)    
#------------------------------------------------------------------------------    
def main():
    name, dnnfilename, package, extranames = decodeCommandline()

    # make sure file(s) exist
    if not os.path.exists(dnnfilename):
        quit("file %s not found" % dnnfilename)

    names = {}
    names['basename'] = name
    names['name']     = name
    names['srcs']     = '\t$(srcdir)/%(basename)s.cc \\\n' % names
    names['date']     = ctime(time())
    names['dnndir']   = name
    names['libname']  = name
    names['version']  = VERSION
    names['inputs']   = ''
    names['softmaxdef']  = ''
    names['softmaximpl'] = ''
    names['selectdef']   = ''
    names['selectimpl']  = ''    
    
    os.system('mkdir -p %(dnndir)s/lib' % names)
    os.system('mkdir -p %(dnndir)s/src' % names)
    os.system('mkdir -p %(dnndir)s/include' % names)


    cppsrc = ''
    if package[0] == 's':
        
        # SCIKIT-LEARN
        
        # extract MLP information from file
        mlpinfo = MLPinfo(dnnfilename, package)
        
        names['netid'] = 0
        names['nnetworks'] = 1
        details = mlpinfo()
        cppsrc = writeCPP(names, details)
        
    else:

        # FBM

        # check for net-display
        answer = os.popen('which net-display').read()
        if answer == '':
            quit('please execute setup.sh to gain access to FBM programs')
        
        # first convert to a pickle file
        cmd = 'netpickle.py %s .bnn.pkl' % dnnfilename
        os.system(cmd)

        # extract MLP information from file
        mlpinfo = MLPinfo('.bnn.pkl', package)
        
        t = split(package, ',')
        if len(t) > 1:
            nnets = atoi(t[1])
        else:
            nnets = 100

        maxnets = len(mlpinfo.mlp)
        nnets   = min(nnets, maxnets)
        names['nnetworks'] = nnets
        details = mlpinfo(0)
        
        # add input names to C++ file
        iname   = details[3]
        rec = '\n// Inputs\n'
        for i, n in enumerate(iname):
            ii = i + 1
            rec += '//   %4d %s\n' % (ii, n)
        inames = rec[:-1]                    
        names['inputs'] = inames
        
        firstid= maxnets - nnets
        for netid in range(nnets):
            names['netid'] = netid

            nid = firstid + netid 
            if netid % 50 == 0: print "\t%(netid)5d\t%(name)s" % names, nid

            details = mlpinfo(nid)
            names['inputs'] = ''
            cppsrc += writeCPP(names, details)
            
    # -------------------------------------------------------------------------
    # WRITE C++ SOURCE
    print 'create C++ source and header files'    
    cppfilename = '%(dnndir)s/src/%(basename)s.cc' % names
    print '\t=> creating file: %s' % cppfilename
    names['cppsrc'] = cppsrc    
    source = SOURCE % names
    open(cppfilename, 'w').write(source)

    # WRITE C++ HEADER
    hdrfilename = '%(dnndir)s/include/%(basename)s.h' % names
    print '\t=> creating file: %s' % hdrfilename
    header = HEADER % names
    open(hdrfilename, 'w').write(header)

    # WRITE LINKDEF and MAKEFILE
    writeLinkdefAndMakefile(names, extranames)
#------------------------------------------------------------------------------
try:
    main()
except KeyboardInterrupt:
    print '\nciao!'

















