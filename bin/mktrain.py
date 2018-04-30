#!/usr/bin/env python
#------------------------------------------------------------------------------
# File: mktrain.py
# Description: Create files <BNN>.sh and <BNN>.var to run FBM code
# Created: 06-Dec-2005 Harrison B. Prosper
# Updated: 20-Oct-2006 HBP & Daekwang Kau
#          02-Apr-2008 HBP some minor fixes for Serban
#          11-Nov-2010 HBP fix sh script when -mr is specified
#          24-Apr-2018 HBP generalize to MLP
#------------------------------------------------------------------------------
import os, sys
from string import *
from time import time, ctime
from random import shuffle
from getopt import getopt, GetoptError
from math import *
#------------------------------------------------------------------------------
USAGE = '''
Usage:
   mktrain.py [options] net-name

   options:
         -h   print this
         -m   model type (b=binary or r=real) [b]
         -N   number of training events [min(5000, all)]
         -H h0[,h1[,h2]]   number of hidden nodes [10]
         -I   number of iterations [300]


   mktrain.py needs two input files:

   1. <BNN-name>.dat              file containing training data
   2. <BNN-name>.var              file containing names of variables
                                  offsets and scale factors        

   In the training file, all columns except the last must be input variables
   and the last column must be the targets. The first row should be a header
   of column names.
'''
SHORTOPTIONS = 'hm:N:H:I:'
COUNT = 5000
ITERATIONS = 250
HIDDEN= 10

template = '''#------------------------------------------------------------------------------
# Description: This file contains the commands to run the BNN training.
#              Note: the data-spec format assumes that all inputs in the
#                    training file are used and the last column is the
#                    target.
# Created: %(time)s by mktrain.py
#------------------------------------------------------------------------------
%(varlist)s
#------------------------------------------------------------------------------
echo "File: %(name)s"

net-spec	%(name)s.bin %(I)d%(H)s 1 / - 0.05:0.5 0.05:0.5 - %(Hprior)s x0.05:0.5 - 100

model-spec	%(name)s.bin %(modeltype)s

data-spec	%(name)s.bin %(I)s %(datatype)s / %(datafile)s@2:%(count)d . %(datafile)s@2:%(count)d .

net-gen		%(name)s.bin fix 0.5

mc-spec		%(name)s.bin repeat 20 heatbath hybrid 100:10 0.2

net-mc		%(name)s.bin 1

mc-spec %(name)s.bin repeat 20 sample-sigmas heatbath 0.95 hybrid 100:10 0.2

echo "Start chain"
echo "Use"
echo "   net-display -h %(name)s.bin"
echo "periodically to check the progress of the chain"

time net-mc	%(name)s.bin %(iter)s

echo ""
echo "Use"
echo "   dnnwrite.py -f100 %(name)s.bin"
echo "to create the BNN function %(name)s.cpp using the last 100 points"
'''

#------------------------------------------------------------------------------
def error(message):
    print "** %s" % message
    sys.exit(0)

def usage():
    print USAGE
    sys.exit(0)

def nameonly(x):
    return os.path.splitext(os.path.basename(x))[0]
#------------------------------------------------------------------------------
def readData(datafile, varfile, count):

    print "\nReading training file: %s" % datafile
    records= map(lambda x: rstrip(x), open(datafile).readlines())
    header = records[0]
    records= records[1:]

    # Make sure we don't ask for more than the number of
    # events per file

    count = min(count, len(records))
    records = records[:count]
    #---------------------------------------------------
    # Convert data to float
    #---------------------------------------------------
    data = map(lambda row:
               map(atof, split(row)), records)

    colnames = split(header)

    # Get variable names, means and sigmas

    vars = map(split, open(varfile).readlines())
    offset = []
    scale  = []
    for index, x in enumerate(vars):
        if len(x) != 3: continue
        name, m, s = x
        if name != colnames[index]:
            error("The order of variables in %s does not match\n\tthat in %s"\
                  % (datafile, varfile))
        offset.append(atof(m))
        scale.append(atof(s))
    return (colnames, data, offset, scale)
#------------------------------------------------------------------------------
# MAIN Program
#------------------------------------------------------------------------------
def main():

    #---------------------------------------------------
    # Decode command line using getopt module
    #---------------------------------------------------
    try:
        options, inputs = getopt(sys.argv[1:], SHORTOPTIONS)
    except GetoptError, m:
        print
        print m
        usage()

    # Make sure we have a network name

    if len(inputs) == 0: usage()

    # Name of BNN

    bnnname = inputs[0]
    trainfile = "%s.dat" % bnnname

    # Set defaults, then parse input line

    binaryModel = True

    varfile = ''
    count   = COUNT
    hidden  = HIDDEN
    iterations = ITERATIONS

    for option, value in options:
        if option == "-h":
            usage()

        elif option == "-m":
            binaryModel = value == "b"

        elif option == "-N":
            count = atoi(value)

        elif option == "-H":
            hidden = map(atoi, split(value, ','))

        elif option == "-I":
            iterations = atoi(value)

    #---------------------------------------------------
    # Check that input files exist
    #---------------------------------------------------
    trainvarfile = nameonly(trainfile)+".var"
    if not os.path.exists(trainfile): error("Can't find %s" % trainfile)
    if not os.path.exists(trainvarfile): error("Can't find %s" % trainvarfile)

    #---------------------------------------------------
    # Mix signal and background events
    #---------------------------------------------------
    colnames, data, offset, scale = readData(trainfile,
                                             trainvarfile,
                                             count)
    ndata = len(data)
    #---------------------------------------------------
    # Use all variables in input files except "target",
    # which should be the last column
    #---------------------------------------------------
    var = colnames[:-1]
    nvar = len(var)

    print "\nInput variables for BNN %s" % bnnname
    varlist = ""
    for i, name in enumerate(var):
        print "\t%s" % name
        index  = i+1
        varlist += "#\t%d\t%s\n" % (index, name) 
    varlist = varlist[:-1]

    #---------------------------------------------------
    # Write <BNN>.sh file
    #---------------------------------------------------

    names = {}
    names['name'] = bnnname
    names['varlist'] = varlist
    names['datafile'] = trainfile
    names['time'] = ctime(time())
    names['count']= ndata+1
    names['I'] = nvar
    fmt = ' %d' * len(hidden)
    names['H'] = fmt % tuple(hidden)
    names['Hprior'] = ' 0.05:0.5 - 0.05:0.5 -' * (len(hidden)-1)
    names['iter'] = iterations
    if binaryModel:
        names['modeltype'] = "binary"
        names['datatype']  = "1 2"
    else:
        names['modeltype'] = "real 0.05:0.5"
        names['datatype']  = "1"

    record = template % names

    shfile = "%(name)s.sh" % names
    print "Write %s" % shfile
    open(shfile, "w").write(record)
#------------------------------------------------------------------------------
main()
