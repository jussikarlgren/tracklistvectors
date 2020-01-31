from random import random, sample
import copy
import math
import numpy
from logger import logger
from sklearn.metrics.pairwise import cosine_distances as cosine
import csv

debug = True  # for logger
monitor = True
error = True


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

    datafile = "/Users/jik/data/playlists/500k_playlist_items.csv"
    testsampleproportion = 1000

    dimensionality = 1000
    denseness = 10
    minimumlenghtofplaylistthreshold = 10  # minimum length of playlist to care
    minimumfrequencyoftrack = 5  # minimum number of playlists a track appears in
    # read CSV file & load into list
    # track_uri,name,popularity_normalized,genres,playlist_uri
    playlisturiposition = 4
    trackuriposition = 0
    tracknameposition = 1
    logger(f"""Reading {datafile}""", debug)
    i = 0
    with open(datafile, 'r') as infile:
        reader = csv.reader(infile, delimiter=",")
        playlists = list(reader)
    logger(f"""Finished reading {len(playlists)} items. Now pruning to playlists of length >= {minimumlenghtofplaylistthreshold}""", debug)
    prev = "dummy"
    nn = 0
    thisrun = []
    prunedlistoftracks = []
    trackfrequency = {}
    for item in playlists:
        if not item[playlisturiposition] == prev:
            if nn > minimumlenghtofplaylistthreshold:
                prunedlistoftracks += thisrun
            thisrun = []
            nn = 0
        nn += 1
        prev = item[playlisturiposition]
        thisrun.append(item)
        if item[trackuriposition] in trackfrequency:
            trackfrequency[item[trackuriposition]] += 1
        else:
            trackfrequency[item[trackuriposition]] = 1
    logger(f"""Number of items down from {len(playlists)} to {len(prunedlistoftracks)} after pruning""", monitor)

    logger("Start building space without NumPy.", debug)
    tracks = {}
    playlistindexvectors = {}
    for oneitem in prunedlistoftracks:
        logger(oneitem, debug)
        track = oneitem[trackuriposition]
        if trackfrequency[track] >= 5:
            playlist = oneitem[playlisturiposition]
            if playlist not in playlistindexvectors:
                playlistindexvectors[playlist] = newrandomvector(dimensionality, denseness)
            indexvector = playlistindexvectors[playlist]
            if track not in tracks:
                contextvector = {}
            else:
                contextvector = tracks[track]
            contextvector = sparseadd(contextvector, indexvector)
            tracks[track] = contextvector
    logger("Done building space of {} items without NumPy.".format(len(tracks)), debug)
    samplesize = len(tracks) // testsampleproportion
    testsample = sample(tracks.keys(), samplesize)
    sparsecosines = []
    for t1 in testsample:
        for t2 in testsample:
            c1 = sparsecosine(tracks[t1], tracks[t2])
            sparsecosines.append([t1, t2, c1])
    logger(f"""Done testing distances for {samplesize} items without NumPy.""", debug)
    with open("/Users/jik/data/tmp/sparsecosines.csv", "w") as cosineoutfile:
        cosinewriter = csv.writer(cosineoutfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for c in sparsecosines:
            cosinewriter.writerow(c)
    logger("Start building space with NumPy.", debug)
    tracks = {}
    playlistindexvectors = {}
    for oneitem in prunedlistoftracks:
        track = oneitem[trackuriposition]
        playlist = oneitem[playlisturiposition]
        if trackfrequency[track] >= 5:
            if playlist not in playlistindexvectors:
                playlistindexvectors[playlist] = newrandomvectornumpy(dimensionality, denseness)
            indexvector = playlistindexvectors[playlist]
            if track not in tracks:
                contextvector = numpy.zeros(dimensionality)
            else:
                contextvector = tracks[track]
            contextvector = contextvector + indexvector
            tracks[track] = contextvector
    logger("Done building space of {} items with NumPy.".format(len(tracks)), debug)
    samplesize = len(tracks) // testsampleproportion
    testsample = sample(tracks.keys(), samplesize)
    numpycosines = []
    for t1 in testsample:
        for t2 in testsample:
            c = cosine([tracks[t1], tracks[t2]])
            numpycosines.append([t1, t2, c])
    logger(f"""Done testing distances for {samplesize} items with NumPy.""", debug)
    with open("/Users/jik/data/tmp/numpycosines.csv", "w") as numpyoutfile:
        cosinewriter = csv.writer(numpyoutfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for c in numpycosines:
            cosinewriter.writerow(c)

