# Losses for the Consciousness Prior Model.
import tensorflow as tf


def prediction_loss(conscious_predictions, actual_conscious_state):
    """Prediction loss on the conscious predictions versus that """
    # TODO(liamfedus):  Do we really want the objective as initially outlined
    # in the paper?  The model predicts the values of particular conscious
    # elements but then compares against potentially 'unconscious' realized
    # values.  The predictability mechanism feels very important, where perhaps
    # unconscious elements should be maintaining a predictive model of theiri
    # own future states, but as initially described, this doesn't seem sensible.
    #
    # Two losses:  
    #   1. Unconscious elements each maintain predictive model of their
    #      respective representations.
    #   2. Conscious state predicts actual *realized* conscious states. 
    pass


def mutual_information_objective():
    pass


def importance_loss(conscious_states):
    """Importance loss encourages the weighting to each conscious element to be
    equal. 

    Args:
        conscious_states:  Weighted gating to each conscious element,
            [batch_size, consciousness_dim].

    Returns:
        Importance loss summed across the batch.
    """
    # TODO(liamfedus):  For this to make sense, we'll need to batch across 
    # simultaneous observational or control environments.
    return tf.reduce_sum(conscious_states, 0)


