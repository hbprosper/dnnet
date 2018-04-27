#!/usr/bin/env python
#------------------------------------------------------------------------------
# File: unweight.py
# Description: Create unweighted events from weighted ones.
# Created: 19-Dec-2005 Harrison B. Prosper
#$Id: unweight.py,v 1.17 2011/05/18 08:51:48 prosper Exp $
#------------------------------------------------------------------------------
import os, sys
from string import *
from random import *
#------------------------------------------------------------------------------
# Binary search - Slightly modified version of
# http://personal.denison.edu/~havill/102/python/search.py
#------------------------------------------------------------------------------
def binsearch(L, item):
    first = 0
    last = len(L) - 1   

    found = False
    while (first <= last) and not found:
        mid = (first + last) / 2
        if item <= L[mid]: # was <
            last = mid;       
        elif item > L[mid]:
            first = mid + 1
##         else:
##             found = True
            
        if first >= last: # fixed 12 May 2011 HBP (> changed to >=)
            mid = first
            found = True

        if found: return mid
    return -1
#------------------------------------------------------------------------------
argv = sys.argv[1:]
argc = len(argv)
if argc < 3:
    sys.exit("Usage:\n\tpython unweight.py input-filename output-filename count")

inpfile = argv[0]
outfile = argv[1]
count   = atoi(argv[2])

if not os.path.exists(inpfile):
    sys.exit("Can't find %s" % inpfile)

#------------------------------------------------------------------------------
# Read input file
#------------------------------------------------------------------------------
print "read input file: %s" % inpfile
inprec = map(lambda x: rstrip(x), open(inpfile).readlines())
header = inprec[0]
inprec = inprec[1:]

#------------------------------------------------------------------------------
# Determine which field is the weight
#------------------------------------------------------------------------------
hdr = split(header)
which=-1
for index in xrange(len(hdr)):
    h = lower(hdr[index])
    if find(h, "weight") > -1:
        which = index
        break;

if which < 0:
    sys.exit("\tweight field not found")

#------------------------------------------------------------------------------
# Sum weights
#------------------------------------------------------------------------------
wcdf = len(inprec)*[0]
t = split(inprec[0])
w = atof(t[which])
wcdf[0] = w
for i in xrange(1,len(inprec)):
    t = split(inprec[i])
    w = atof(t[which])
    wcdf[i] = wcdf[i-1] + w

sumw = wcdf[-1]
print "\tweight sum: %e" % sumw

#------------------------------------------------------------------------------
# Select events according to weight
#------------------------------------------------------------------------------
records = count * [0]
for i in xrange(count):
    w = uniform(0, sumw)
    k = binsearch(wcdf, w)
    if k < 0:
        sys.exit("**error** Not found %e" % w)
    t = split(inprec[k])
    t[which] = "1.0"
    records[i] = joinfields(t,' ')

#------------------------------------------------------------------------------
# Write out shuffled records
#------------------------------------------------------------------------------
shuffle(records)
records = [header] + records
recs = [joinfields(x, ' ')+'\n' for x in map(split, records)]
open(outfile,"w").writelines(recs)


