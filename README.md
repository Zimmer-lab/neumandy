# NeuManDy - Neuronal Manifold Dynamics


This is code for the masters thesis of Liana Akobian, which can be accessed here:  https://doi.org/10.34726/hss.2025.120462

## Code Overview

There are three sets of functions in this repository:
1. zimmer_helper_functions - Conversion of Zimmer lab data formats to a standardized .h5 format
2. alignment - Preprocessing of multiple datasets to remove individual dataset differences
3. modeling - Modeling of low dimensional neuronal manifold using dynamical systems

## Project Abstract

Understanding how neurons interact with each other to produce behavior is a key challenge in neuroscience. The dynamics of these interacting neurons define the computations that underlie the processing of sensory information, decision making, and the generation of motor output. Recent advances in dynamical system modeling have formalized observed neural activity as the temporal evolution of states within a neural state space governed by dynamical laws. Significant progress has been achieved by assuming these laws to be of autonomous nature, meaning that neural states evolve deterministically. However, such models may not provide sufficient biological interpretability as they fail to capture unpredictable external forces. Here, we propose a controlled decomposed linear dynamical system (cdLDS), an extension of the autonomous dynamical system model dLDS, by incorporating inputs that control the system. We apply cdLDS to a neural manifold, a low-dimensional dynamical structure, from 23 C. elegans individuals and show that it successfully disentangles intrinsic neural dynamics from control signals, offering insight into perturbations of neural dynamics. This framework provides a foundation for identifying the neural correlates of control signals and for understanding the impact of control mechanisms on neural dynamics.

