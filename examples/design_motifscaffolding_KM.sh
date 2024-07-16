#!/bin/bash
# Here, we're running one of the motif-scaffolding benchmark examples
# Specifically, we're scaffolding site 5 from RSV-F protein
# We specify the output path and input pdb (the RSV-F protein)
# We specify the protein we want to build, with the contig input:
#   - 10-40 residues (randomly sampled)
#   - residues 163-181 (inclusive) on the A chain of the input
#   - 10-40 residues (randomly sampled)
# We generate 10 designs

chmod 755 ../scripts/run_inversion.py 
../scripts/run_new_inversion.py inference.output_prefix=example_outputs/design_motifscaffolding inference.input_pdb=input_pdbs/8cds.pdb 'contigmap.contigs=[5-30/A10-25/5-30]' inference.num_designs=10 
# ../scripts/run_inversion.py inference.output_prefix=example_outputs/design_motifscaffolding inference.input_pdb=input_pdbs/8cds.pdb 'contigmap.contigs=[10-10/A101-116/30-50]' inference.num_designs=10
