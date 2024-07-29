#!/bin/bash
# Here, we run motif embedding learning and generation
# Adapted from design_motifscaffolding.sh (RFdiffusion) by Kevin Michalewicz (AstraZeneca)

../scripts/run_new_inversion.py inference.output_prefix=example_outputs/design_motifscaffolding inference.input_pdb=input_pdbs/6exz.pdb 'contigmap.contigs=[30-50/A557-572/30-50]' inference.num_designs=30 inference.inversion_train=False
