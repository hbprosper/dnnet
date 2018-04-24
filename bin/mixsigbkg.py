#!/usr/bin/env python
#------------------------------------------------------------------------------
# File: mixsigbkg.py
# Description: Mix signal and background files and normalize the data
# Created: 06-Dec-2005 Harrison B. Prosper
# Updated: 20-Oct-2006 HBP & Daekwang Kau
#          02-Apr-2008 HBP Adapt for Serban
#          22-Apr-2018 HBP handle csv files
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
   mixsigbkg.py [options] net-name

   options:
         -h   print this
         -s   signal text file [sig.dat]
         -b   background text file [bkg.dat]
         -v   variables text file [use all variables in input text file
                                   but skip event weight etc.]
         -N   number of events/file [5000]

         The output files will be:
         
                            <filename>.[dat|csv]
                            <filename>.var
'''
SHORTOPTIONS = 'hs:b:v:N:'
COUNT = 5000
SKIPVARS = ['','eventweight',
            'f_weight', 'weight','entry','target','eventcode', 'process']

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
# Decode command line using getopt module
#------------------------------------------------------------------------------
def decodeCommandLine():
    try:
        options, inputs = getopt(sys.argv[1:], SHORTOPTIONS)
    except GetoptError, m:
        print
        print m
        usage()

    # Need an output file name
    
    if len(inputs) == 0: usage()

    outfile = inputs[0]
    
    # Set defaults, then parse input line

    sigfile = 'sig.dat'
    bkgfile = 'bkg.dat'
    varfile = ''
    count   = COUNT
    
    for option, value in options:
        if option == "-h":
            usage()

        elif option == "-s":
            sigfile = value
            
        elif option == "-b":
            bkgfile = value
            
        elif option == "-v":
            varfile = value
            
        elif option == "-N":
            count = atoi(value)

    #---------------------------------------------------
    # Check that input files exist
    #---------------------------------------------------
    if not os.path.exists(sigfile): error("Can't find %s" % sigfile)
    if not os.path.exists(bkgfile): error("Can't find %s" % bkgfile)
    return (sigfile, bkgfile, varfile, count, outfile)
#------------------------------------------------------------------------------
def main():

    sigfile, bkgfile, varfile, count, outfile = decodeCommandLine()

    names = {}
    names['name'] = nameonly(outfile)

    # check extension. if csv
    ext = split(sigfile, '.')[-1] 
    is_csv = ext == 'csv'
    if is_csv:
        delim = ','
    else:
        delim = ' '
        
    names['ext'] = ext
    names['delim'] = delim
    
    # Read signal file and add Target column

    print 
    print "signal file:     %s" % sigfile,
    sigrec = map(lambda x: rstrip(x), open(sigfile).readlines())
    
    header = sigrec[0] + delim + 'target'
    sigrec = sigrec[1:]
    print "\t=> %d events" % len(sigrec)
    sigrec = map(lambda x: x + delim + '1', sigrec)

    # Read background file and add Target column

    print "background file: %s" % bkgfile,
    bkgrec = map(lambda x: rstrip(x), open(bkgfile).readlines())
    bkgrec = bkgrec[1:]
    print "\t=> %d events" % len(bkgrec)
    bkgrec = map(lambda x: x + delim + '0', bkgrec)

    # Make sure we don't ask for more than the number of
    # events in each file
    
    count = min(count, min(len(sigrec), len(bkgrec)))

    # Concatenate signal + background records and shuffle them
    records = sigrec[:count] + bkgrec[:count]
    print "shuffle %d signal and background events" % len(records)
    shuffle(records)
    records = map(lambda x: x + '\n', records)
    
    #---------------------------------------------------
    # create column name to index map
    #---------------------------------------------------
    colnames = [strip(x) for x in split(header, delim)]
    colmap = {}
    for index in range(len(colnames)):
        name = colnames[index]
        colmap[name] = index

    #---------------------------------------------------
    # Get list of possible input variables
    #---------------------------------------------------
    if varfile == '':
        # Use all variables in input files
        var = colnames
    else:
        # Use variables given in variables file
        if not os.path.exists(varfile): error("Can't find %s" % varfile)
        var = map(strip, open(varfile).readlines())

    # Skip any blank lines and non-physics data variables
    var  = filter(lambda x: not lower(x) in SKIPVARS, var)
    nvar = len(var)
    
    #---------------------------------------------------
    # Convert data to float
    #---------------------------------------------------
    data = map(lambda row:
               map(atof, split(row, delim)), records)

    #---------------------------------------------------
    # Compute mean and sigma for each variable
    #---------------------------------------------------
    mean   = nvar * [0.0]
    sigma  = nvar * [0.0]
    count = 0
    for row in data:
        for index in range(nvar):
            name = var[index]
            x = row[colmap[name]]
            mean[index] += x
            sigma[index] += x * x

    #---------------------------------------------------
    # Write means and sigmas to variables file
    #---------------------------------------------------
    varfile = "%(name)s.var" % names
    print "\nwrite", varfile
    out = open(varfile, "w")
    ndata = len(data)
    for index in range(nvar):
        mean[index]  = mean[index] / ndata
        sigma[index] = sigma[index] / ndata
        sigma[index] = sqrt(sigma[index] - mean[index]**2)
        if sigma[index] == 0:
            error("variable %s has zero variance" % var[index])
            
        record = "%-24s\t%10.3e\t%10.3e" % \
                 (var[index], mean[index], sigma[index])
        print "\t%s" % record
        out.write('%s\n' % record)
    out.close()
    
    #---------------------------------------------------
    # Normalize the data
    #---------------------------------------------------
    outfilename = "%(name)s.%(ext)s" % names
    print "\nwrite", outfilename
    out = open(outfilename, "w")
    fmt = "%s" + delim
    fmt = fmt * nvar
    
    header = fmt % tuple(var) + 'target\n'    
    out.write(header)
    recs = nvar * [0]
    for ii, row in enumerate(data):
        if ii % 2000 == 0: print ii
        for index in range(nvar):
            name = var[index]
            k = colmap[name]
            x = (row[k] - mean[index]) / sigma[index]
            recs[index] = '%e' % x
        record = joinfields(recs, delim) + delim + '%d\n' % row[colmap['target']]
        out.write(record)
    out.close()
#------------------------------------------------------------------------------
main()
