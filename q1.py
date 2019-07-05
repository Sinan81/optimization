#!/usr/bin/env python

# Sinan81 , Jan 2018
# This code does an exact calculation of an optimization problem with a simple objective function.

import numpy as np
import argparse
import multiprocessing
import sys
# import pdb # for debugging
# from copy import deepcopy # for copying by value

inputfilename='input.txt'
outputfilename='output.txt'
isParallel=False

def input2array():
    "Read and cast raw input into proper arrays"
    "This function requires the global variables: inputfilename outputfilename"

    global ia
    global ja
    global q

    try:
        # read the input file of interest
        with open(inputfilename, "r") as f:
            # read in array
            a = np.loadtxt(f)

        # get indices on the first column
        ia = array2indexes(a,0)

        # get indices on the second column
        ja = array2indexes(a,1)

        # get "q": the weights of links between nodes "i" and "j"
        q = a[:,2]

        # print a few values to see what we're dealing with
        print(a[0:3,:])
        print('-------')
        return ia,ja,q
    except IOError:
        print('!! IOError !!')
        print('Input file "'+inputfilename+'" doesnt exist, please try again. \n\
                Exiting ...')
        raise SystemExit
    except ValueError:
        print('!! ValueError !!')
        print('It seems something is not right with the input file. \n'
              'Check it is all numbers in the input file, and \n'
              'also check every line has 3 numbers in it. \nExiting ...')
        raise SystemExit

def array2indexes(a,ncol):
    # extract the column and convert to int
    ja = a[:,ncol]
    ja = ja.astype(int)

    return ja


def H(q,xi,xj):
    # calculate H=sum_i sum_j h_ij
    # where H is the objective function
    return np.sum(q*xi*xj)


def write_results(lowest_H_calcs):
    # write results to a file in the format
    #   value x_i1 x_i2 ...
    # where x_i are all *unique* node indicies indicated in the input file
    # and sorted in increasing order

    np.savetxt(outputfilename, lowest_H_calcs[:,1:], fmt="%2.3g")
    print('\n# Calculations are written to: '+outputfilename+'\n')


def initialize_lowest_H_calcs(nconfigs,nbasis):
    # nconfigs: number of possible xi configurations
    # nbasis: size of the set xi (i.e. iju.size, unique integers in the first
    #   two column of the input file)

    # keep a certain fraction of lowest obj-function results
    nkeep = 10 if nconfigs > 10 else nconfigs
    lowest_H_calcs = np.zeros((nkeep,nbasis+2))
    # initialize H to inf so that the column will be populated
    #   by the calculations right away
    lowest_H_calcs[:,1] = np.inf

    return lowest_H_calcs


def get_config(k,nbasis):
    # get kth configuration
    config = np.binary_repr(k,width=nbasis)
    # convert to integer array
    config = np.array(list(config), dtype=int)

    return config


def set_up_x(ia,iju,config):
    # set xi (decision variables) as determined by configuration

    # initialize xi to zero
    xi = np.zeros((ia.size))

    # pdb.set_trace()

    # set ones
    for m in iju[config==1]:
        xi[ia==m] = 1

    return xi

def calc_H_exact_parallel():
    # calc H exactly, i.e. go through all possible configurations of xi,xj
    # here the calculation of H for every configuration is divided into chunks
    # which should yield better performance on machines with multiple cpus

    global iju

    # import input data into proper arrays
    [ia,ja,q] = input2array()

    # get all unique node indexes: iju
    #   note that size of iju determines nbasis, the size of xi
    iju = np.unique(np.concatenate((ia,ja)))

    # number of configurations go like 2^{all unique x_i variables}
    nconfigs = get_nconfigs(iju.size)

    # initalize the array where we save results
    lowest_H_calcs = initialize_lowest_H_calcs(nconfigs,iju.size)

    p = multiprocessing.Pool(multiprocessing.cpu_count())
    stride=pow(10,6)
    # memory footprint of results vector is 8 bytes x stride
    # stride,  memory footprint
    # 10^6      8MB
    # 10^7      80MB
    if nconfigs > stride: # Do not bother with chunks if system is small
        # get number of chunks
        nstride=nconfigs/stride

        # initialize arrays where we will keep lowest 10 configurations
        result=np.zeros((10))
        kresult=np.ones((10))
        for chunk in range(0,nstride+1):
            print("%d th chunk, %3.3f percent complete"\
                    % (chunk,100.*float(chunk)/float(nstride)))
            ibegin = chunk*stride
            iend = (chunk+1)*stride
            if iend > nconfigs:
                iend = nconfigs
            # parallel calculation via p.map
            kchunk = range(ibegin,iend)
            result_chunk = p.map(kernel,kchunk)
            result_chunk = np.array(result_chunk)

            # combine overall and current results
            stage = np.concatenate([result, result_chunk])
            kstage = np.concatenate([kresult, kchunk])

            # pdb.set_trace()

            # sort it all and keep the lowest 10
            sortind = stage.argsort()
            stage = stage[sortind]
            kstage = kstage[sortind]

            # extract lowest 10
            result = stage[0:10]
            kresult = kstage[0:10]

    else:
        result = p.map(kernel,range(0,nconfigs))
        result = np.array(result)
        sortind = result.argsort()
        result = result[sortind]
        ka = np.array(range(0,nconfigs))
        kresult = ka[sortind]
        result = result[0:10]
        kresult = kresult[0:10]

    for i in range(0,10 if nconfigs > 10 else 8):
        # config indixes to configs
        k=int(kresult[i])
        lowest_H_calcs[i,2:] = get_config(k,iju.size)

    lowest_H_calcs[:,0] = kresult
    lowest_H_calcs[:,1] = result

    print("The lowest H values and corresponding configurations:")
    print(lowest_H_calcs)

    write_results(lowest_H_calcs)
    return result


def kernel(k):
    config = get_config(k,iju.size)
    xi  = set_up_x(ia,iju,config)
    xj  = set_up_x(ja,iju,config)

    return H(q,xi,xj)


def get_nconfigs(ijusize):
    # check if maximum integer supported by the system will be exceeded:
    if ijusize > np.log2(sys.maxint):
        print("it seems nconfigs will be bigger the maximum integer supported "+
            "by the system")
        print("Make sure the system has less than %d decision variables"\
                % np.log2(sys.maxint) )
        print("Exiting ...")
        raise SystemExit

    # number of configurations go like 2^{all unique x_i variables}
    # TODO: consider using np.float_power instead
    return  np.power(2,ijusize)


def calc_H_exact():
    # calc H exactly, i.e. go through all possible configurations of xi,xj

    # import input data into proper arrays
    [ia,ja,q] = input2array()

    # get all unique node indexes (i.e. decision variables): iju
    #   note that size of iju determines nbasis, the size of xi
    iju = np.unique(np.concatenate((ia,ja)))

    # number of configurations go like 2^{number of decision variables xi}
    nconfigs = get_nconfigs(iju.size)

    # initalize the array where we save results
    lowest_H_calcs = initialize_lowest_H_calcs(nconfigs,iju.size)

    # calc H(q,xi,xj) for each possible configuration of xi,xj
    for k in range(0,nconfigs):
        # get kth config
        config = get_config(k,iju.size)

        xi = set_up_x(ia,iju,config)
        xj = set_up_x(ja,iju,config)

        value = H(q,xi,xj)

        if value < np.max(lowest_H_calcs[:,1]):
            nrow = np.argmax(lowest_H_calcs[:,1])
            lowest_H_calcs[nrow,2:] = config
            lowest_H_calcs[nrow,0]  = k
            lowest_H_calcs[nrow,1]  = value

    # sort by the value of the objective-function
    sortind = lowest_H_calcs[:,1].argsort()
    lowest_H_calcs = lowest_H_calcs[sortind]

    print("The lowest H values and corresponding configurations:")
    print(lowest_H_calcs)

    write_results(lowest_H_calcs)

    return lowest_H_calcs


def parse_arguments():
    global inputfilename
    global outputfilename
    global isParallel

    parser = argparse.ArgumentParser(\
            description="A script solving a simple small size optimization problem.")
    parser.add_argument('inputfile', type=str, \
            help='input file, in human readable text format, where one set of \
            x_i x_j q_ij values  are indicated in each line')
    parser.add_argument('--output','-o', dest='outputfile', type=str, \
            help='output file name', default='output.txt')

    parser.add_argument('--parallel','-p', action='store_true', \
            help='do calculation in parallel on a shared memory machine with\
            multiple CPU')
    args = parser.parse_args()

    inputfilename = args.inputfile
    outputfilename = args.outputfile
    isParallel = args.parallel
    print('Input file: "'+inputfilename+'"')
    print('Output file: "'+outputfilename+'"')

# =========================================================================== #
def main():

    parse_arguments()

    if isParallel:
        calc_H_exact_parallel()
    else:
        calc_H_exact()

if __name__== '__main__':
    main()
