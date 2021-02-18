# -*- coding: utf-8 -*-

### Multi-targets multi-cameras tracking

# Useful imports
import warnings
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import clear_output
from time import time
import cv2
import functions

# Preparation for the Mask RCNN model

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x
warnings.filterwarnings('ignore') 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mask_rcnn.mrcnn import utils
import mask_rcnn.mrcnn.model as modellib
from mask_rcnn.mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "masks_rcnn/samples/coco/"))  # To find local version
import coco


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "mask_rcnn/logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn/mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Load trained dataset configuration
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()


# Useful imports and parameters initialization

# Commented out IPython magic to ensure Python compatibility.
# %cd deep_sort

# Imports

from deep_sort.detection import Detection
from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

# Model classes

## Individuals
class Individuals():
    '''
    Class used to store the caracteristics of each detected individuals on a frame.
    ----------
    Parameters
    ----------
    rois: array-like
        Coordinates of the bounding boxes (rois = Regions of Interest).
    sores: array-like
        Confidence scores to belong to the associated class.
    masks: array-like
        Binary masks of the detected individuals on the associated frame.
    class_ids: array-like
        Class id of the detection according to MS COCO nomenclature (always '1' in our case).
    track_ids: array-like
        Unique ids to identify an individual through the tracking process.
    ids_color: array_like
        List of all available colors to link to an unique individual.
    img: array-like
        The frame associated to these individuals.
    ----------
    Attributes
    ----------
    rois: array-like
        Coordinates of the bounding boxes (rois = Regions of Interest).
    scores: array-like
        Confidence scores to belong to the associated class.
    masks: array-like
        Binary masks of the detected individuals on the associated frame.
    class_ids: array-like
        Class id of the detection according to MS COCO nomenclature (always '1' in our case).
    track_ids: array-like
        Unique ids to identify an individual through the tracking process.
    ids_color: array_like
        List of all available colors to link to an unique individual.
    img: array-like
        The frame associated to these individuals.
    '''

    def __init__(self):
        self.rois = []
        self.scores = []
        self.masks = []
        self.class_ids = [] 
        self.track_ids = []
        self.ids_color = []
        self.img = []

## MaskRCNN
class MaskRCNN():
    """
    Create an instance of a MaskRCNN object based on the keras implementation by 
    Matterport: https://github.com/matterport/Mask_RCNN.
    Mask RCNN in an end-to-end detection algorithm. Given an image, it returns a 
    list of object ids and the corresponding boxes, masks and scores.
    This model has been pre-trained on the MS COCO data set.
    ----------
    Parameters
    ----------
    model: MaskRCNN
        The MaskRCNN model.
    threshold: float
        The confidence under which we consider the detection as incorect.
    pretrained_weight_path: string
        Path to used MS COCO pretrained weights for the Mask RCNN deep layers.
    ----------
    Attributes
    ----------
    model: MaskRCNN
        The MaskRCNN model.
    threshold: float
        The confidence under which we consider the detection as incorect.
    individuals: Individuals
        Object that contains all the caracteristics of the detected individuals.
    pretrained_weight_path: string
        Path to used MS COCO pretrained weights for the Mask RCNN deep layers.
    """
    def __init__(self, model, threshold, pretrained_weight_path):
        self.model = model
        self.threshold = threshold
        self.individuals = Individuals()
        self.pretrained_weight_path = pretrained_weight_path

        # Load weights trained on MS-COCO
        self.model.load_weights(self.pretrained_weight_path, by_name=True)
      
    def predict(self, frame, verb=0):
        """
        ----------
        Parameters
        ----------
        frame: array_like
            The image on which the MaskRCNN detector is apply.
        ------
        Return
        ------
        output: dictionnary
            It contains four lists: one for the objects boxe coordinates, one for the 
            objects id, one for the objects score and one for the objects binary mask.
        """

        output = self.model.detect([frame], verbose=verb)[0]
        
        # Keep all objects that are a person with a confidence score above a threshold
        output = functions.update_results(output, self.threshold)

        #TO TO : case where nothing is detected ???

        self.individuals.rois = output['rois']
        self.individuals.masks = output['masks']
        self.individuals.class_ids = output['class_ids']
        self.individuals.scores = output['scores']
        self.individuals.img = frame

        return self.individuals

## DeepSORT
class DeepSORT():
    """
    Create an instance of a DeepSORT object based on the implementation by Nicolai 
    Wojke: https://github.com/nwojke/deep_sort.
    The multi-cams adapatation has been inspired by: 
    https://github.com/LeonLok/Multi-Camera-Live-Object-Tracking.
    The idea of this class is to perform a tracking of the detected individuals.
    DeepSORT is a tracking algorithm. It holds a 'track' object for each detected
    individual. Newly created tracks are classified as 'tentative' until enough 
    evidence has been collected. Then, the track state is changed to 'confirmed'. 
    Tracks that are no longer alive are classified as 'deleted' to mark them for 
    removal from the set of active tracks.

    In the initialization, we simply create the tracker object based on cosine metric.
    ----------
    Parameters
    ----------
    colors: array_like
        List of all available colors to link to an unique ID.
    model_filename: string
        DeepSORT encoder weights.
    max_cosine_distance: float
        The matching threshold. Samples with larger distance are considered an invalid 
        match. It is used to associate the detections with the existing tracks.
        Associations with cost larger than this value are disregarded.
    nms_max_overlap: float
        Threshold used in NMS: Non-Max-Suppression.
    nn_budget: 

    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    ----------
    Attributes
    ----------
    colors: array_like
        List of all available colors to link to an unique ID.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    nms_max_overlap: float
        Threshold used in NMS: Non-Max-Suppression.
    metric: nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    tracker: Tracker
        This is the multi-target tracker of the Deep Sort Algorithm. It initializes 
        a tracking list and a Kalman filter.
    encoder: ImageEncoder
        ...
    """
    def __init__(self, colors, model_filename, max_cosine_distance=0.2, nn_budget=None, 
               max_age=30, nms_max_overlap=1.0, n_init=3):

        self.colors = colors
        self.max_age = max_age
        self.n_init = n_init
        self.nms_max_overlap = nms_max_overlap 
        self.metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
            )
        self.tracker = Tracker(self.metric,self.max_age)
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)


    def perform_track(self,frame,individuals):
        '''
        The tracking method. It makes the link between our videos, the outputs of the 
        detector and the used DeepSORT architecture. It allows each identical detection 
        to have a unique ID through the videos.
        ----------
        Parameters
        ----------
        frame: array_like
            The current frame from which we track the individuals.
        output: Individuals
            The output of the detector from which we only kept the individuals with more 
            than a certain threshold confidence score.
        ------
        Return
        ------
        output: Individuals
            The updated output after performing the tracking method. 
        '''
        frame_height, frame_width, _ = frame.shape

        # Generate detections objects from the output of the Mask RCNN Detector.
        '''
        Detection class represents a bounding box detection in a single image.
        To do so, we need to compute the features associated with each box.
        '''
        detections=[]
        # Change bboxes format y1,x1,y2,x2 to match bbox format of DeepSORT x,y,w,h.
        converted_rois = functions.convert_roi_shape(individuals.rois)
        features = self.encoder(frame,converted_rois)
        for roi, score, feature in zip(converted_rois,individuals.scores,features):
            detections.append(Detection(roi,score,feature)) 
    
        # Run non-maxima suppression. 
            # TO DO : try without NMS (maybe unuseful as already in mask rcnn)
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        '''
        tracker.predict() propagates track state distributions one time step forward.

        It is called once every time step (before 'update') and for each track of the 
        tracks list, it propagates the state distribution to the current time step using 
        a Kalman filter prediction step. It also increments each track age.
        '''
        self.tracker.predict()
        '''
        tracker.update(detections) performs the actual measurement update and track 
        management.

        Is starts by runing a cascade matching between all detections and tracks on
        the last 'max_age' time steps. For that, the function splits the track set 
        into 'confirmed' and 'unconfirmed' tracks ('unconfirmed' means either 'tentative' 
        or 'deleted').

        Then it associates the 'confirmed' tracks using appearance features. For that,
        the function computes a cost matrix based on the chosen metric between all 
        tracks and all detection and performs a minimal cost matching using the 
        sklearn.utils.linear_assignment function.

        Then it associates remaining tracks together with 'unconfirmed' tracks using 
        IOU matching (= Computer intersection over union Matching).

        Finally, it updates the track set and the distance metric. For that, it updates
        the matched tracks (by updating the Kalman Filters mean and covariance, by
        adding the detection features to the associated track features and by changing
        'Tentative' state to 'Confirmed' if necessary), it marks as "missed" the tracks 
        that hasn't been associated with any match at the current time step (and deletes 
        them after 'max_age' missed steps) and it initiates a track for the detections 
        that hasn't been associated with any track.
        '''
        self.tracker.update(detections)


        # Update individuals ids and colors to be identify.
        total_track = len(self.tracker.tracks)
        individuals.track_ids = np.zeros((total_track))
        individuals.ids_color = np.zeros((total_track,3))
        C = len(self.colors)
        for i, track in enumerate(self.tracker.tracks):
            individuals.track_ids[i] = track.track_id
            individuals.ids_color[i] = self.colors[i%C]

        return individuals

## Track
class Track():
    def __init__(self, detector, tracker, VideoFile, OutputVideo, fps, 
                 show_masks, show_masks_contour):
        self.detector = detector
        self.tracker = tracker
        self.VideoFile = VideoFile
        self.OutputVideo = OutputVideo
        self.fps = fps
        self.show_masks = show_masks
        self.show_masks_contour = show_masks_contour

    def tracking(self):
        '''
        This function performs the detection and the tracking through a single video file.
        ----------
        Parameters
        ----------
        detector: MaskRCNN objet
            The used detector object (MaskRCNN).
        tracker: DeepSORT object
            The used tracker object (DeepSORT).
        VideoFile: string
            Path to the used video file.
        OutputVideo: string
            Path to the output video file.
        fps: int
            Output Video frequency
        -------
        Returns
        -------
        global_times, detecting_times, tracking_times, drawing_times : lists
            Lists containing the time period of each interesting process.
        '''
        cap = cv2.VideoCapture(self.VideoFile)

        if (cap.isOpened()== False): 
            print("Error opening video stream or file")

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        out = cv2.VideoWriter(
            self.OutputVideo,
            cv2.VideoWriter_fourcc('M','J','P','G'),
            self.fps,
            (frame_width,frame_height)
            )

        # Take initial time to measure time performance and put it in a list
        time0 = time()
        global_times = [time0]
        detecting_times = []
        tracking_times = []
        drawing_times = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0

        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if ret == True:
                #### Operations to perform over each frame ####
                
                # Apply detector to the frame
                time1 = time()
                individuals = self.detector.predict(frame)
                time2 = time()
                detecting_times.append(time2-time1)
                # Apply tracker to the frame
                time1 = time()
                individuals = self.tracker.perform_track(frame,individuals)
                time2 = time()
                tracking_times.append(time2-time1)

                # Create an output image by drawing the corresponding masks over the 
                # detected persons of the frame
                time1 = time()   
                masked_frame = functions.draw_masks_and_boxes(frame, individuals.rois, 
                                                    individuals.masks, 
                                                    individuals.scores, 
                                                    individuals.track_ids, 
                                                    individuals.ids_color,
                                                    show_masks=self.show_masks,
                                                    show_masks_contour=self.show_masks_contour,
                                                    roi_thickness = 2,
                                                    mask_thickness = 1
                                                    )
                time2 = time()
                drawing_times.append(time2-time1)
                # Display the resulting frame
                #cv2.imshow('masked_frame', masked_frame)
                #cv2_imshow(masked_frame) # For Google Collab

                # Write the resulting frame in a new video file
                out.write(masked_frame)
                
                count += 1
                if (count % 10) == 0:
                    clear_output(wait=True)
                    print('Progression: {:.2%}'.format(count/total_frames))
            else:
                clear_output(wait=True)
                print('Progression: {:.2%}'.format(count/total_frames))   
                break      
            global_times.append(time())

        # When everything done, release the capture
        out.release()
        cap.release()

        return global_times, detecting_times, tracking_times, drawing_times


"""### **Tests for performance analysis**"""

# Parameters
VideoFile = 'tests/6p-c1.avi'
OutputVideo = 'tests/tracked_6p-c1.avi'
fps = 25
show_masks = False
show_masks_contour = False

# Set of available colors
colors = [(0,255,255),(0,255,0),(255,0,0),(0,0,255),(255,0,255),(255,255,0),
          (128,128,0),(0,128,128),(128,0,0),(128,0,128)]

detection_threshold = 0.98
model_filename = 'deep_sort/mars-small128.pb'       
max_age = 60
n_init = 25

# Create Detector Class
detectorMaskRCNN = MaskRCNN(
      modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config),
      threshold=detection_threshold,
      pretrained_weight_path=COCO_MODEL_PATH
      )

# Create DeepSORT Class
trackerDeepSORT = DeepSORT(
     colors, 
     model_filename,
     max_cosine_distance=0.2, 
     nn_budget=None, 
     max_age=max_age, 
     n_init=n_init
      )

# Crete Track Class
track = Track(detectorMaskRCNN, 
                           trackerDeepSORT, 
                           VideoFile, 
                           OutputVideo,
                           fps,
                           show_masks,
                           show_masks_contour
                           )

# Perform Tracking
global_times, detecting_times, tracking_times, drawing_times = track.tracking()

# Display time results
total_detecting_time = np.sum(detecting_times)
total_tracking_time = np.sum(tracking_times)
total_drawing_time = np.sum(drawing_times)
total_time = (global_times[-1] - global_times[0])
precessing_times = np.array(global_times[1:]) - np.array(global_times[:-1])

print('The average detecting time is {:.5f} milli-seconds and represents {:.2%} of total time.'.format(
                            np.mean(detecting_times)*1000,(total_detecting_time/total_time)))

print('The average tracking time is {:.5f} milli-seconds and represents {:.2%} of total time.'.format(
                            np.mean(tracking_times)*1000,(total_tracking_time/total_time)))

print('The average drawing time is {:.5f} milli-seconds and represents {:.2%} of total time.'.format(
                            np.mean(drawing_times)*1000,(total_drawing_time/total_time)))

print('The average global processing time is {:.5} milli-seconds.'.format(
                            np.mean(precessing_times[1:])*1000))

print('On average, the MaskRCNN+DeepSORT algorithm processes {:.4} FPS.'.format(
                            1/np.mean(precessing_times[1:])))