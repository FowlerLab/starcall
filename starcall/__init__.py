"""
# STARCall: STitching Alignment and Read Calling for in-situ sequencing experiments

This is the a python package that holds code for the processing of in situ sequencing
experiments. It was developed with a SnakeMake pipeline at <https://github.com/FowlerLab/starcall-workflow.git>,
however it can also be used on its own to process image based sequencing experiments.

## Installation

STARCall can be installed by cloning the repository, then installing with pip.

	git clone https://github.com/FowlerLab/starcall
	cd starcall
	pip3 install ./

## Usage

A full snakemake workflow using this package is available at <https://github.com/FowlerLab/starcall-workflow.git>

The main improvements make in this package are filters and methods for the
detection of the amplicon colonies, which present as small dots in sequencing
images. This is encapsulated in the starcall.dotdetection.detect_dots() method,
which takes a sequence of images in the form of a numpy array, shape (num_cycles, num_channels, width, height).
Background is filtered out and bright areas are detected, and their sequences are read out. The result
is a pandas dataframe of reads, with the value in each channel for each cycle.

Starcall also provides a custom pandas accessor to interact with these read tables, called 'reads'. A dataframe
with the columns below can use the accessor:
    'position_x', 'position_y': The pixel location of the sequencing read
    'values_cycle00_G', 'values_cycle00_T' ... 'values_cycle11_A', 'values_cycle11_C':
        The sequencing values of the read, from the filtered sequencing images.
        Values from every cycle and each channel are stored.

The accessor provides custom properties listed below:
    'positions': a numpy array of shape (num_reads, 2). The positions of the reads,
    'values': a numpy array of shape (num_reads, num_cycles, num_channels). 
        The sequencing values for all reads
    'sequences': a numpy array of strings of shape (num_reads,). If there is a column
        called 'sequence' in the dataframe this is just the contents of that column.
        Otherwise, the sequences are generated from the sequencing values, by selecting
        the maximum channel in each cycle to build up a sequence.

Changes to positions and values will both propagate back to the underlying dataframe, so
you can do something like:
    table.reads.positions *= 2
    table.reads.values /= np.linalg.norm(table.reads.values, axis=2)[:,:,None]

Full reference documentation is available at <https://fowlerlab.github.io/starcall-docs/starcall.html>
"""
