#!/usr/bin/env python

## source: https://github.com/davidbau/net-scope/blob/565fc753f11bb4096ed8cc21811dcbb80fa4f800/ade20k.py

import glob
import os
import re
from collections import namedtuple

import numpy

from scipy.io import loadmat
from scipy.misc import imread, imsave
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom

from settings import DATA_DIRECTORY

ADE_ROOT = DATA_DIRECTORY
ADE_VER = 'ADE20K_2016_07_26'


def decodeClassMask(im):
    '''Decodes pixel-level object/part class and instance data from
    the given image, previously encoded into RGB channels.'''
    # Classes are a combination of RG channels (dividing R by 10)
    return (im[:, :, 0] // 10) * 256 + im[:, :, 1]


def decodeInstanceMask(im):
    # Instance number is scattered, so we renumber them
    (orig, instances) = numpy.unique(im[:, :, 2], return_inverse=True)
    return instances.reshape(classes.shape)


def encodeClassMask(im, offset=0):
    result = numpy.zeros(im.shape + (3,), dtype=numpy.uint8)
    if offset:
        support = im > offset
        mapped = (im + support) * offset
    else:
        mapped = im
    result[:, :, 1] = mapped % 256
    result[:, :, 0] = (mapped // 256) * 10
    return result


class Dataset:
    def __init__(self, directory=None, version=None):
        # type: (object, object) -> object
        # Default to value of ADE20_ROOT env variable
        if directory is None:
            directory = os.environ['ADE20K_ROOT']
        directory = os.path.expanduser(directory)
        # Default to the latest version present in the directory
        if version is None:
            contents = os.listdir(directory)
            if not list(c for c in contents if re.match('^index.*mat$', c)):
                version = sorted(c for c in contents if os.path.isdir(
                    os.path.join(directory, c)))[-1]
            else:
                version = ''
        self.root = directory
        self.version = version

        mat = loadmat(self.expand(self.version, 'index*.mat'), squeeze_me=True)
        index = mat['index']
        Ade20kIndex = namedtuple('Ade20kIndex', index.dtype.names)
        for name in index.dtype.names:
            setattr(self, name, index[name][()])
        self.index = Ade20kIndex(
            **{name: index[name][()] for name in index.dtype.names})
        self.raw_mat = mat

    def expand(self, *path):
        '''Expands a filename and directories with the ADE dataset'''
        result = os.path.join(self.root, *path)
        if '*' in result or '?' in result:
            globbed = glob.glob(result)
            if len(globbed):
                return globbed[0]
        return result

    def filename(self, n):
        '''Returns the filename for the nth dataset image.'''
        filename = self.index.filename[n]
        folder = self.index.folder[n]
        return self.expand(folder, filename)

    def short_filename(self, n):
        '''Returns the filename for the nth dataset image, without folder.'''
        return self.index.filename[n]

    def size(self):
        '''Returns the number of images in this dataset.'''
        return len(self.index.filename)

    def num_object_types(self):
        return len(self.index.objectnames)

    def seg_filename(self, n):
        '''Returns the segmentation filename for the nth dataset image.'''
        return re.sub(r'\.jpg$', '_seg.png', self.filename(n))

    def part_filenames(self, n):
        '''Returns all the subpart images for the nth dataset image.'''
        filename = self.filename(n)
        level = 1
        result = []
        while True:
            probe = re.sub(r'\.jpg$', '_parts_%d.png' % level, filename)
            if not os.path.isfile(probe):
                break
            result.append(probe)
            level += 1
        return result

    def part_levels(self):
        return max([len(self.part_filenames(n)) for n in range(self.size())])

    def image(self, n):
        '''Returns the nth dataset image as a numpy array.'''
        return imread(self.filename(n))

    def segmentation(self, n, include_instances=False):
        '''Returns the nth dataset segmentation as a numpy array,
        where each entry at a pixel is an object class value.

        If include_instances is set, returns a pair where the second
        array labels each instance with a unique number.'''
        data = imread(self.seg_filename(n))
        if include_instances:
            return (decodeClassMask(data), decodeInstanceMask(data))
        else:
            return decodeClassMask(data)

    def parts(self, n, include_instances=False):
        '''Returns an list of part segmentations for the nth dataset item,
        with one array for each level available.  If included_instances is
        set, the list contains pairs of numpy arrays (c, i) where i
        represents instances.'''
        result = []
        for fn in self.part_filenames(n):
            data = imread(fn)
            if include_instances:
                result.append((decodeClassMask(data), decodeInstanceMask(data)))
            else:
                result.append(decodeClassMask(data))
        return result

    def full_segmentation(self, n, include_instances=False):
        '''Returns a single tensor with all levels of segmentations included
        in the channels, one channel per level.  If include_instances is
        requested, a parallel tensor with instance labels is returned in
        a tuple.'''
        full = [self.segmentation(n, include_instances)
                ] + self.parts(n, include_instances)
        if include_instances:
            return tuple(numpy.concatenate(tuple(m[numpy.newaxis] for m in d)
                                           for d in zip(full)))
        return numpy.concatenate(tuple(m[numpy.newaxis] for m in full))

    def object_name(self, c):
        '''Returns a short English name for the object class c.'''
        # Off by one due to use of 1-based indexing in matlab.
        if c == 0:
            return '-'
        result = self.index.objectnames[c - 1]
        return re.split(',\s*', result, 1)[0]

    def object_count(self, c):
        '''Returns a count of the object over the whole dataset.'''
        # Off by one due to use of 1-based indexing in matlab.
        return self.index.objectcounts[c - 1]

    def object_presence(self, c):
        '''Returns a per-dataset-item count of the object.'''
        # Off by one due to use of 1-based indexing in matlab.
        return self.index.objectPresence[c - 1]