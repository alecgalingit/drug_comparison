# Drug comparison

This repository contains a number of scripts used to compute statistics regarding the use of experimental drugs including Nitrous Oxide (NO) and Lysergic Acid Diethylamide (LSD) to treat mental health conditions. The code within this repo is summarized in the following bullet points:
   * subj_energy.py is used to compute control energy statistics as well as to compute t tests between different treatment groups.
   * dominance.py is used to determine the extent to which different receptors in the brain influence control energy changes.
   * displacement.py is used to analyze how movement during the brain scanning process may bias results.
   * receptor_histograms.py is used to visualize receptor densities.

## Background

Different drugs interact with a wide variety of different receptors in the brain, with varying degrees of affinity and efficacy. Spatial (regional) patterns of a drug’s impact on brain dynamics may map on to receptor profiles. (see https://doi.org/10.1126/sciadv.adf8332). 

Classic psychedelics bind to a number of receptors, however the serotonin (5-HT) 2a receptor subtype has been identified as the primary site responsible for their subjective, therapeutic, and neural effects. When studying their impacts on human blood-oxygenation level dependent (BOLD) fMRI data, there are a few confounds that are difficult to control for: 1) systematic differences in attentional awareness, alertness, etc between drug and non-drug conditions and 2) impacts of vasoconstriction on the BOLD signal. Comparing the fMRI effects of classic psychedelics (ie 2a agonists) to other drugs may help resolve what effects, if any, are due to systematic changes in these two confounds, what are drug-specific, and what are general to drug-induced altered states. 

## The data:

* 16 participants sober vs nitrous oxide (see: https://pubmed.ncbi.nlm.nih.gov/37031827; dataset 1).
* 15 participants sober vs LSD (see Carhart-harris 2016; Singleton 2022)

## The approach

* get regional differences between each condition.
    * t-tstats or group-level deltas
    * these can be used for relating each drug’s brain effect to neurotransmitter data
* compare the differences across drugs
    * for each subject get their delta (placebo - drug) control energy
    * z-score each subject’s delta
    * unpaired t-test (at each region) between the two datasets using these subject-level deltas





