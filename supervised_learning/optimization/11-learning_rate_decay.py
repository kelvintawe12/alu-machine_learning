#!/usr/bin/env python3
"""
This module provides a function to update the learning.
"""

def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy.

    Parameters
    ----------
    alpha : float
        The original learning rate.
    
    Returns
    -------
    float
        The updated value for alpha.
    """
    return alpha / (1 + decay_rate * (global_step // decay_step))
