# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 15:25:44 2021

@author: tgent
"""

def update_results(results, threshold=0.8):
    """
    This function keeps the detection within an image whose scores are above a threshold and whose class is 'person'.
    Parameters
    ----------
    results: dictionnary
        The output of the predict function of the Mask-RCNN method. Contains the boxes.
        rois: array_like
            Regions of interest: coordinate of the box around an object.
        class_ids: array_like
            Id of the objects according to the MS COCO labels.
        scores: array_like
            Confidence score to belong to the associated id class.
        masks: array_like
            Binary masks of each object.
    threshold: float
        Detector confidence score above which the object is kept.
    Return
    ------
    persons: dictionnary
        The updated results dictionnary. If no persons in the frame, returns an empty dictionnary.
    """

    persons = {'rois': None, 'class_ids': None, 'scores': None, 'masks': None}
    for i, class_id in enumerate(results['class_ids']):

        if (class_id) == 1 & (results['scores'][i] >= threshold):
            # If the box contains a person which more than "threshold" confidence, then the object is kept.
            if persons['rois'] is None :
                # For the first object to be kept
                persons['rois'] = np.array([results['rois'][i]], dtype=np.int32)
                persons['class_ids'] = np.array([results['class_ids'][i]], dtype=np.int32)
                persons['scores'] = np.array([results['scores'][i]], dtype=np.float32)
                persons['masks'] = np.expand_dims(np.array(results['masks'][:,:,i]), axis=2)
            else :
                # For the next objects
                persons['rois'] = np.concatenate((persons['rois'], [results['rois'][i]]))
                persons['class_ids'] = np.concatenate((persons['class_ids'], [results['class_ids'][i]]))
                persons['scores'] = np.concatenate((persons['scores'], [results['scores'][i]]))
                persons['masks'] = np.concatenate((persons['masks'], np.expand_dims(results['masks'][:,:,i], axis=2)), axis=2)

    if persons['rois'] is None:
        persons = {'rois': np.array([], dtype=np.int32), 'class_ids': np.array([], dtype=np.int32), 'scores': np.array([], dtype=np.float32), 'masks': np.array([])}
    return persons