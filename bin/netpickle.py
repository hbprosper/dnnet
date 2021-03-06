#!/usr/bin/env python
#------------------------------------------------------------------------------
# File: netpickle.py
# Purpose: Extract parameters from an FBM file and write them out to a pickle
#          file in the same format as scikit-learn's MLPClassifer.
# Created: 20-Apr-2018 Harrison B. Prosper
#------------------------------------------------------------------------------
import os, sys
import pickle as pc
from string import *
from time   import *
#------------------------------------------------------------------------------
DEBUG = False

USAGE = '''
Usage:
  netpickle.py <bnn-log-file-name>.bin [output file-name]
'''
#------------------------------------------------------------------------------
def usage():
    sys.exit(USAGE)

def quit(s):
    sys.exit("\n**error** %s" % s)

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
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def main():
    global DEBUG
    
    # Make sure we have a network log-file
    inputs = sys.argv[1:]
    if len(inputs) == 0: usage()

    # Strip away extension
    netname  = os.path.splitext(inputs[0])[0] # Name of network
    nnlogfile= netname + ".bin"               # Name of (binary) log file
    nnvarfile= netname + ".var"               # Name of variables file

    if len(inputs) > 1:
        nnpklfile = inputs[1]
    else:
        nnpklfile= netname + ".pkl"           # Name of pickle file
    
    # Make sure bnn-logfile and bnn-varfile exist
    if not os.path.exists(nnlogfile): quit("File %s not found" % nnlogfile)
    if not os.path.exists(nnlogfile): quit("File %s not found" % nnvarfile)
    
    # Get number of networks
    inp = os.popen("net-display -h %s" % nnlogfile)
    record  = inp.readline() # Skip blank line
    record  = inp.readline()
    t = split(record)
    try:
        numberNetworks = atoi(t[6])
    except:
        quit("Unable to decode:\n%s" % record)
    
    #---------------------------------------------------
    # Get network structure.
    #---------------------------------------------------        
    net = os.popen("net-spec %s" % nnlogfile).readlines()
    ninputs = 0
    nhidden = []
    noutputs= 0

    numberParams = 0
    for record in net:
        t = split(record)
        if len(t) == 0: continue

        keyword = t[0]
        if keyword == "Input":
            ninputs = atoi(t[3])
            lhidden = ninputs
            
        elif keyword == "Hidden":
            value = atoi(t[4])
            nhidden.append(value)
            numberParams += (lhidden + 1) * value
            lhidden = value
            
        elif keyword == "Output":
            value = atoi(t[3])
            nhidden.append(value)
            numberParams += (lhidden + 1) * value
                
        elif keyword == "Prior":
            break
         
    #---------------------------------------------------
    # Get network type (real or binary)
    #---------------------------------------------------        
    net = os.popen("model-spec %s" % nnlogfile).read()
    print
    print "extract network parameters and write to a pickle file"
    if find(net, "real") > -1:
        model = "real"
        print "  model type: regressor"
    else:
        model = "binary"
        print "  model type: classifier"
    print "  number of networks:     %6d" % numberNetworks
    print "  number of inputs:       %6d" % ninputs
    print "  number of nodes/layer:  %s"  % nhidden
    print "  number of parameters:   %6d" % numberParams

    #---------------------------------------------------
    # Call net-display for each index
    #---------------------------------------------------
    netnumber = 0
    nhidden.append(ninputs)

    network = [0]*numberNetworks
    for n in range(numberNetworks):
        index = n + 1
        if index % 50 == 0:
            print "\t=> network %5d" % index
            
        #---------------------------------------------------
        # Read records
        #---------------------------------------------------
        records = os.popen("net-display -p %s %d" % \
                                          (nnlogfile, index)).readlines()
        layer = 0
        irec  = 0
        mlp   = Parameter()
        if model == 'real':
            mlp.out_activation_ = 'identity'
        else:
            mlp.out_activation_ = 'logistic'
            
        while irec < len(records):
            tokens = split(records[irec])

            try:
                keyword = tokens[-2]
            except:
                keyword = None
                
            if keyword == 'Weights':
                if DEBUG:
                    print "\t\tlayer %5d" % layer
                    print "\t\t\tweights"
                    
                ninp = nhidden[layer-1]
                nout = nhidden[layer]
                if DEBUG:
                    print "\t\tninp: %d" % ninp
                    print "\t\tnout: %d" % nout

                irec += 1 # blank line
                W = ninp * [0]
                for i in range(ninp):
                    recs = []
                    while (len(recs) < nout) and (irec < len(records)):
                        irec+= 1
                        recs += map(atof, split(records[irec]))
                    W[i] = recs
                    if DEBUG:
                        print "%3d %s" % (i, W[i])
                        
                # now get biases
                irec += 2 # skip blank line
                tokens  = split(records[irec])
                try:
                    keyword = tokens[-2]
                except:
                    quit('problem decoding record\n%s\n' % records[irec])
                    
                if keyword != 'Biases':
                    quit('expected keyword Biases not found on\n%s' % records[irec])
                    
                irec+= 1 # blank line
                recs = []
                while (len(recs) < nout) and (irec < len(records)):
                    irec += 1
                    recs += map(atof, split(records[irec]))                
                B = recs
                if len(B) != nout:
                    quit('mismatch in bias count len(B) = %d != %d in layer %d' % \
                             (len(B), nout, layer))
                if DEBUG:
                    print "\t\t\tbiases"
                    print "%3s %s" % ("", B)
                    
                # update weights and biases for current layer
                mlp.coefs_.append(W)
                mlp.intercepts_.append(B)

                if DEBUG:
                    print "\t<== weight and biases cached ==>"
                    
                # update layer number
                layer += 1
                DEBUG = False
                
            # remember to update record number
            irec += 1

        network[n] = mlp

    # cache for mean and scale
    name  = []
    mean  = []
    scale = []
    records = map(lambda x: (x[0], atof(x[1]), atof(x[2])),
                      map(split,
                              open(nnvarfile).readlines()))
    for n, m, s in records:
        name.append(n)
        mean.append(m)
        scale.append(s)
    scaler = Scaler(name, mean, scale)
    t = (network, scaler)

    print "  saving to file: %s" % nnpklfile
    pc.dump(t, open(nnpklfile, 'wb'), pc.HIGHEST_PROTOCOL)
    print
#------------------------------------------------------------------------------
main()


















