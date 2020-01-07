from random import random, sample
import copy
import math
import numpy
from logger import logger
from sklearn.metrics.pairwise import cosine_distances as cosine
debug = True  # for logger

numberofplaylists = 1000
numberoftracks = 10000
minimumnumberoftracksperplaylist = 50
maximumnumberoftracksperplaylist = 200
dimensionality = 1000
denseness = 10


# simulated data
playlists = []
for i in range(numberofplaylists):
    playlistlength = minimumnumberoftracksperplaylist + \
                 int(random() * (maximumnumberoftracksperplaylist - minimumnumberoftracksperplaylist))
    oneplaylist = sample(range(numberoftracks), playlistlength)
    playlists.append(oneplaylist)


# tested this as an alternative to my own simplistic interpretation but this was no faster.
def newrandomvectornumpy(vectordimensionality, nonzeroelements):
    vec = numpy.zeros(vectordimensionality)
    nonzeros = sample(range(vectordimensionality), nonzeroelements)
    negatives = nonzeroelements // 2
    for nz in nonzeros[:negatives]:
        vec[nz] = 1
    for nz in nonzeros[negatives:]:
        vec[nz] = -1
    return vec


#  Very simple implementation of index vectors
#  Some standard libraries for sparse vectors are available
#  in Python but appear to have prohibitive overhead of implementation.
def newrandomvector(dimensionality, denseness):
    """
    Generates a sparse vector in the form of a dict with denseness entries composed of
    integer keys and 1 or -1 values, equal number of each.
    :param dimensionality: int
    :param denseness: int
    :return: dict int->int
    """
    vec = {}
    if denseness % 2 != 0:
        denseness += 1
    if denseness > 0:  # no need to be careful about this, right? and k % 2 == 0):
        nonzeros = sample(range(dimensionality), denseness)
        negatives = denseness // 2
        for ix in nonzeros[:negatives]:
            vec[ix] = 1
        for ix in nonzeros[negatives:]:
            vec[ix] = -1
    return vec


def sparseadd(onevec, othervec, weight=1.0):
    """
    Adds two sparse vectors represented as dicts with numerical values, optionally weighting othervec with the
    weight parameter which defaults to 1.0 (equal weighting).
    :param onevec: dict with numerical values
    :param othervec: dict with numerical values
    :param weight: float
    :return: dict with float values
    """
    result = copy.copy(onevec)
    for ll in onevec:
        result[ll] = onevec[ll]
    for kk in othervec:
        if kk in result:
            result[kk] = result[kk] + othervec[kk] * float(weight)
        else:
            result[kk] = othervec[kk] * float(weight)
    return result


def sparsecosine(xvec, yvec):
    """
    Calculates cosine between two sparse vectors. (Can be replaced with sklearn.dist.cosine,
    but that seems to be more costly in processing time.)
    :param xvec: dict with numerical values
    :param yvec: dict with numerical values
    :return: float
    """
    x2 = 0
    y2 = 0
    xy = 0
    for ix in xvec:
        x2 += xvec[ix] * xvec[ix]
    for jx in yvec:
        y2 += yvec[jx] * yvec[jx]
        if jx in xvec:
            xy += xvec[jx] * yvec[jx]
    if x2 * y2 == 0:
        cos = 0
    else:
        cos = xy / (math.sqrt(x2) * math.sqrt(y2))
    return cos


def npyfy(vec):
    npvec = numpy.zeros(dimensionality)
    for ix in vec:
        npvec[ix] = vec[ix]
    return npvec


if __name__ == '__main__':

    # for every playlist
    #   generate sparse random index vector for that playlist
    #   for every song in that playlist
    #       if not seen, initiate empty context vector
    #       add the playlist index vector to the context vector
    testsampleproportion = 500
    logger("Start building space without NumPy.", debug)
    tracks = {}
    for oneplaylist in playlists:
        indexvector = newrandomvector(dimensionality, denseness)
        for track in oneplaylist:
            if track not in tracks:
                contextvector = newrandomvector(dimensionality, denseness)
            else:
                contextvector = tracks[track]
            contextvector = sparseadd(contextvector, indexvector)
            tracks[track] = contextvector
    logger("Done building space of {} items without NumPy.".format(len(tracks)), debug)
    samplesize = len(tracks) // testsampleproportion
    testsample = sample(tracks.keys(), samplesize)
    for t1 in testsample:
        for t2 in testsample:
            c1 = sparsecosine(tracks[t1], tracks[t2])
    logger("Done testing distances for {} items against {} items without NumPy.".format(len(testsample), len(tracks)),
           debug)
    logger("Start building space with NumPy.", debug)
    tracks = {}
    for oneplaylist in playlists:
        indexvector = newrandomvectornumpy(dimensionality, denseness)
        for track in oneplaylist:
            if track not in tracks:
                contextvector = newrandomvectornumpy(dimensionality, denseness)
            else:
                contextvector = tracks[track]
            contextvector = contextvector + indexvector
            tracks[track] = contextvector
    logger("Done building space of {} items without NumPy.".format(len(tracks)), debug)
# samplesize = len(tracks) // testsampleproportion
# testsample = sample(tracks.keys(), samplesize)
# for t1 in testsample:
#     for t2 in testsample:
#         c = cosine(tracks[t1], tracks[t2])
# logger("Done testing distances for {} items against {} items with NumPy.".format(len(testsample), len(tracks)),
#        debug)
