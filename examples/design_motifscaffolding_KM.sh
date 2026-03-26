#!/bin/bash
# Here, we run motif embedding learning and generation
# Adapted from design_motifscaffolding.sh (RFdiffusion) by Kevin Michalewicz (Imperial/UBA)

../scripts/run_new_inversion.py inference.output_prefix=example_outputs/design_motifscaffolding inference.input_pdb=input_pdbs/1bcf.pdb 'contigmap.contigs=[17-17/A18-25/21-21/A47-54/37-37/A92-99/23-23/A123-130/28-28]' inference.num_designs=1000 inference.inversion_train=False



