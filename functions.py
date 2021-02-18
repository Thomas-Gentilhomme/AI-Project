# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 15:25:44 2021

@author: Thomas GENTILHOMME
"""

import numpy as np
import cv2
from skimage.measure import find_contours

###########################
#    Keep individuals     #
###########################

def update_results(results, threshold):
    """
    This function keeps the detections within an image whose score is above a 
    threshold and whose class is 'person'.
    ----------
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
    ------
    Return
    ------
    persons: dictionnary
        The updated results dictionnary. If no persons in the frame, returns an 
        empty dictionnary.
    """
    
    rois = results['rois']
    class_ids = results['class_ids']
    scores = results['scores']
    masks = results['masks']

    available_individuals_indices = []
    for idx in range(len(class_ids)):
        if (class_ids[idx]) == 1 & (scores[idx] >= threshold):
            available_individuals_indices.append(idx)
  
    rois = rois[available_individuals_indices]
    class_ids = class_ids[available_individuals_indices]
    scores = scores[available_individuals_indices]
    masks = masks[:,:,available_individuals_indices]

    output = {'rois': rois, 'class_ids': class_ids, 'scores': scores, 'masks': masks}

    if output['rois'] is None:
        # Format where no detections meet both conditions.
        output = {'rois': np.array([], dtype=np.int32), 
                  'class_ids': np.array([], dtype=np.int32), 
                  'scores': np.array([], dtype=np.float32), 
                  'masks': np.array([])
                  }

    return output



###########################
#    Convert ROI shape    #
###########################
    
def convert_roi_shape(rois):
    '''
    Convert Regions of Interest format to DeepSORT format.
    ---------
    Parameter
    ---------
    rois: array_like
        List of boxes (regions of interest).
    ------
    Return
    ------
    converted_rois: array_like
        List of boxes with DeepSORT format.
    '''
    converted_rois = []
    for roi in rois:
        y1, x1, y2, x2 = roi
        box = [x1, y1, x2 - x1, y2 - y1]
        converted_rois.append(box)
    return converted_rois



###########################
#      Visualization      #
###########################
    
def draw_masks_and_boxes_without_tracking(image, rois, masks, scores, ids, 
                                          colors, show_masks=True, show_rois=True,
                                          show_captions=True, show_masks_contour=True, 
                                          mask_intensity=0.6, roi_thickness=1, 
                                          mask_thickness=1):
    """
    Function that draws the detected boxes, masks and captions of a frame and 
    returns it.
    ----------
    Parameters
    ----------
    image: array_like
        Frame on which we draw the boxes, masks and captions.
    rois: array_like
        Regions of interest: coordinates of the detected box around an object.     
    masks: array_like
        Binary masks of each object.
    ids: array_like
        Unique ids to identify an individual through the tracking process.
    colors: array_like
        Color palette to differentiale individuals.
    show_masks: boolean
        Allow to masks or not.
    show_masks_contour: boolean
        Allow to masks contour or not.
    show_rois: boolean
        Allow to show boxes or not.
    show_rois_track: boolean
        Allow to show predicted boxes or not.
    show_captions: boolean
        Allow to show captions or not.
    mask_intensity:  float
        Intensity of the color of the mask.
    roi_thickness: int
        Thickness of the predicted boxes.
    mask_thickness: int
        Thickness of the masks coutour.
    ------
    Return
    ------
    masked_image: array_like
        The frame with the drawn masks, boxes, ids and scores.
    """
    
    N = rois.shape[0]

    frame_height, frame_width = image.shape[:2]

    masked_image = image.astype(np.uint8).copy()
    #masked_image = cv2.UMat(masked_image).get()

    for i in range(N):
        # Loop over all instances.
        color = colors[i]  

        # Bounding boxes
        y1, x1, y2, x2 = rois[i]
        if show_rois:
            masked_image = cv2.rectangle(masked_image,(x1,y2),(x2,y1),color,roi_thickness)

        # Caption
        if show_captions:
            score = scores[i] 
            caption = " ID: {} score: {:.3f}".format(ids[i], score)
            
            # Add a background to the caption
            cv2.rectangle(masked_image, (x1,y1+10), (x2,y1), color, -1)

            # cv2.LINE_AA give anti-aliased lines
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img = masked_image, text = caption, org = (x1,y1+8), 
                        fontFace = font, fontScale = 0.3, color = (0,0,0), 
                        thickness = 1, lineType = cv2.LINE_AA
                        ) 
        
        # Mask
        mask = masks[:, :, i]
        if show_masks:
            for c in range(3):
                # Attenuate original area and add attenuate color over it.
                masked_image[:, :, c] = np.where(mask == 1, 
                                                 masked_image[:, :, c] * \
                                                 (1 - mask_intensity) + \
                                                 mask_intensity * color[c] * 255,
                                                 masked_image[:, :, c]
                                                 )

            # Mask Contour
            if show_masks_contour:
                '''
                Using a padding to apply the skimage.measure.find_coutours method and
                keep edges.
                '''
                padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
                padded_mask[1:-1, 1:-1] = mask
                contours = find_contours(padded_mask, 0.5)
                '''
                Remove the padding and flip the coordinates.
                '''
                flip_contours = [np.fliplr(verts) - 1 for verts in contours]
                for contour in flip_contours:
                  '''
                  In case mask parts are separated from each other.
                  This is the case in some treaky detection.
                  '''
                  cv2.polylines(masked_image, pts = [np.int32(contour)], 
                                isClosed = True, color = color, thickness = mask_thickness
                                )
            
    return masked_image

def draw_masks_and_boxes_with_tracking(image, dect_rois, pred_rois, masks, scores, 
                                       ids, colors, show_masks=True, show_rois=True, 
                                       show_rois_track=True, show_captions=True,
                                       show_masks_contour=True, mask_intensity=0.6, 
                                       roi_thickness=1, mask_thickness=1):
    """
    Function called during tracking that draws the detected and predicted boxes, 
    masks and captions of a frame and returns it.
    ----------
    Parameters
    ----------
    image: array_like
        Frame on which we draw the boxes, masks and captions.
    dect_rois: array_like
        Regions of interest: coordinates of the detected box around an object.
    pred_rois: array_like
        Coordinates of the predicted box around an object by the Kalman filers.     
    masks: array_like
        Binary masks of each object.
    ids: array_like
        Unique ids to identify an individual through the tracking process.
    colors: array_like
        Color palette to differentiale individuals.
    show_masks: boolean
        Allow to masks or not.
    show_masks_contour: boolean
        Allow to masks contour or not.
    show_rois: boolean
        Allow to show boxes or not.
    show_captions: boolean
        Allow to show captions or not.
    mask_intensity:  float
        Intensity of the color of the mask.
    roi_thickness: int
        Thickness of the predicted boxes.
    mask_thickness: int
        Thickness of the masks coutour.
    ------
    Return
    ------
    masked_image: array_like
        The frame with the drawn masks, boxes, ids and scores.
    """
    
    N = dect_rois.shape[0]

    frame_height, frame_width = image.shape[:2]

    masked_image = image.astype(np.uint8).copy()
    #masked_image = cv2.UMat(masked_image).get()

    for i in range(N):
        # Loop over all instances.
        color = colors[i]  

        # Bounding boxes
        y1, x1, y2, x2 = pred_rois[i]
        if show_rois_track and (pred_rois[i] != [0. 0. 0. 0.]):
            masked_image = cv2.rectangle(masked_image,(x1,y2),(x2,y1),color,roi_thickness)
        y1_dect, x1_dect, y2_dect, x2_dect = dect_rois[i]
        if show_rois:
            masked_image = cv2.rectangle(masked_image,(x1_dect,y2_dect),(x2_dect,y1_dect),
                                         (255,255,255),1
                                         )

        # Caption
        if show_captions and (pred_rois[i] != [0. 0. 0. 0.]):
            score = scores[i] 
            caption = " ID: {} score: {:.3f}".format(ids[i], score)
            
            # Add a background to the caption
            cv2.rectangle(masked_image, (x1,y1+10), (x2,y1), color, -1)

            # cv2.LINE_AA give anti-aliased lines
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img = masked_image, text = caption, org = (x1,y1+8), 
                        fontFace = font, fontScale = 0.3, color = (0,0,0), 
                        thickness = 1, lineType = cv2.LINE_AA
                        ) 
        
        # Mask
        mask = masks[:, :, i]
        if show_masks and len(pred_rois[i])!=0:
            for c in range(3):
                # Attenuate original area and add attenuate color over it.
                masked_image[:, :, c] = np.where(mask == 1, 
                                                 masked_image[:, :, c] * \
                                                 (1 - mask_intensity) + \
                                                 mask_intensity * color[c] * 255,
                                                 masked_image[:, :, c]
                                                 )

            # Mask Contour
            if show_masks_contour:
                '''
                Using a padding to apply the skimage.measure.find_coutours method and
                keep edges.
                '''
                padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
                padded_mask[1:-1, 1:-1] = mask
                contours = find_contours(padded_mask, 0.5)
                '''
                Remove the padding and flip the coordinates.
                '''
                flip_contours = [np.fliplr(verts) - 1 for verts in contours]
                for contour in flip_contours:
                  '''
                  In case mask parts are separated from each other.
                  This is the case in some treaky detection.
                  '''
                  cv2.polylines(masked_image, pts = [np.int32(contour)], 
                                isClosed = True, color = color, thickness = mask_thickness
                                )
            
    return masked_image
