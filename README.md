# CRISPR-Cas9 off-target prediction engine

## Problem Statement

### General Task

The CRISPR-Cas system is an RNA-guided nuclease commonly used to introduce a
single-stranded break (ssDNA nick) or double-stranded break (dsDNA break). The
guide RNA is typically 20 nucleotides (nts) long, and the genomic locus must
often be adjacent to a PAM (Protospacer Adjacent Motif) site for DNA cleavage to
occur.

The RNA sequence and PAM site determine the specificity of the nuclease.
However, many Cas and Cas-like nucleases exhibit off-target activity, meaning
they can occasionally bind to unintended sites resembling the target sequence.
When the CRISPR-Cas system localizes to such unintended sites and performs its
function, this is referred to as an off-target effect.

In in vitro methods, if off-target editing occurs in protein-coding regions of a
gene, it can lead to significant unintended consequences. Thus, predicting the
likelihood of off-target effects is critical.

This problem cannot be solved solely by algorithmic sequence alignment methods,
as off-target effects depend not only on nucleotide complementarity between the
RNA and DNA but also on more complex mechanisms.

Machine learning (ML) approaches are effective because, when trained on large
datasets of experimentally observed off-target effects, models learn to capture
hidden factors influencing targeting.

### Technical Task

Input: Two sequences, sgRNA and targetDNA, represented as strings of nucleotides
(A, T, G, C) approximately 20 characters in length (variable size). Output: A
binary prediction (0 or 1) indicating the presence (1) or absence (0) of an
off-target effect.

## Data and experiments

Methods performance on Change-seq dataset

## Project progress

30.03 Added project description. Added metrics for performance at Change-seq
dataset for all four methods so far

23.03 Added R-Crispr and Deep-CNN methods

17.03 Added Crispr DipOff method code

15.03 Added bidirectional LSTM with BERT embeddings method code

## References

This code is based on methods presented in following papers:

1. [Lin J, Wong KC. Off-target predictions in CRISPR-Cas9 gene editing using deep learning. Bioinformatics. 2018 Sep 1;34(17):i656-i663. doi: 10.1093/bioinformatics/bty554. PMID: 30423072; PMCID: PMC6129261. ](https://pmc.ncbi.nlm.nih.gov/articles/PMC6129261/)
2. [Orhan Sari, Ziying Liu, Youlian Pan, Xiaojian Shao, Predicting CRISPR-Cas9 off-target effects in human primary cells using bidirectional LSTM with BERT embedding, Bioinformatics Advances, Volume 5, Issue 1, 2025, vbae184, https://doi.org/10.1093/bioadv/vbae184](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbae184/7934878)
3. [Md Toufi kuzzaman, Md Abul Hassan Samee, M Sohel Rahman, CRISPR-DIPOFF: an interpretable deep learning approach for CRISPR Cas-9 off-target prediction, Briefi ngs in Bioinformatics, Volume 25, Issue 2, March 2024, bbad530, https://doi.org/10.1093/bib/bbad530](https://academic.oup.com/bib/article/25/2/bbad530/7588687)
4. [Niu R, Peng J, Zhang Z, Shang X. R-CRISPR: A Deep Learning Network to Predict Off-Target Activities with Mismatch, Insertion and Deletion in CRISPR-Cas9 System. Genes (Basel). 2021 Nov 25;12(12):1878. doi: 10.3390/genes12121878. PMID: 34946828; PMCID: PMC8702036.](https://pmc.ncbi.nlm.nih.gov/articles/PMC8702036/)
