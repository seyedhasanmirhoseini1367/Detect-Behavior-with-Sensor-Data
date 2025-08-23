
Overview

Can you use movement, temperature, and proximity sensor data to differentiate between body-focused repetitive behaviors (BFRBs), like hair pulling, from non-BFRB everyday gestures, like adjusting glasses? The goal of this competition is to develop a predictive model that distinguishes BFRB-like and non-BFRB-like activity using data from a variety of sensors collected via a wrist-worn device. Successfully disentangling these behaviors will improve the design and accuracy of wearable BFRB-detection devices, which are relevant to a wide range of mental illnesses, ultimately strengthening the tools available to support their treatment.

<img width="298" height="556" alt="image" src="https://github.com/user-attachments/assets/a096eec3-736e-43ba-8715-740367240c41" />


Description
Body-focused repetitive behaviors (BFRBs), such as hair pulling, skin picking, and nail biting, are self-directed habits involving repetitive actions that, when frequent or intense, can cause physical harm and psychosocial challenges. These behaviors are commonly seen in anxiety disorders and obsessive-compulsive disorder (OCD), thus representing key indicators of mental health challenges.

To investigate BFRBs, the Child Mind Institute has developed a wrist-worn device, Helios, designed to detect these behaviors. While many commercially available devices contain Inertial Measurement Units (IMUs) to measure rotation and motion, the Helios watch integrates additional sensors, including 5 thermopiles (for detecting body heat) and 5 time-of-flight sensors (for detecting proximity). See the figure to the right for the placement of these sensors on the Helios device.

We conducted a research study to test the added value of these additional sensors for detecting BFRB-like movements. In the study, participants performed series of repeated gestures while wearing the Helios device:

1. They began a transition from “rest” position and moved their hand to the appropriate location (Transition);

2. They followed this with a short pause wherein they did nothing (Pause); and

3. Finally they performed a gesture from either the BFRB-like or non-BFRB-like category of movements (Gesture; see Table below).

Each participant performed 18 unique gestures (8 BFRB-like gestures and 10 non-BFRB-like gestures) in at least 1 of 4 different body-positions (sitting, sitting leaning forward with their non-dominant arm resting on their leg, lying on their back, and lying on their side). These gestures are detailed in the table below, along with a video of the gesture.

BFRB-Like Gesture (Target Gesture)	
Above ear - Pull hair	 
Forehead - Pull hairline	
Forehead - Scratch	
Eyebrow - Pull hair	
Eyelash - Pull hair	
Neck - Pinch skin	
Neck - Scratch	
Cheek - Pinch skin


Non-BFRB-Like Gesture (Non-Target Gesture)
Drink from bottle/cup	
Glasses on/off	
Pull air toward your face	
Pinch knee/leg skin	
Scratch knee/leg skin	
Write name on leg	 
Text on phone	
Feel around in tray and pull out an object	
Write name in air	
Wave hello	


This competition challenges you to develop a predictive model capable of distinguishing (1) BFRB-like gestures from non-BFRB-like gestures and (2) the specific type of BFRB-like gesture. Critically, when your model is evaluated, half of the test set will include only data from the IMU, while the other half will include all of the sensors on the Helios device (IMU, thermopiles, and time-of-flight sensors).

Your solutions will have direct real-world impact, as the insights gained will inform design decisions about sensor selection — specifically whether the added expense and complexity of thermopile and time-of-flight sensors is justified by significant improvements in BFRB detection accuracy compared to an IMU alone. By helping us determine the added value of these thermopiles and time-of-flight sensors, your work will guide the development of better tools for detection and treatment of BFRBs.

Relevant articles:
Garey, J. (2025). What Is Excoriation, or Skin-Picking? Child Mind Institute. https://childmind.org/article/excoriation-or-skin-picking/

Martinelli, K. (2025). What is Trichotillomania? Child Mind Institute. https://childmind.org/article/what-is-trichotillomania/

Evaluation
The evaluation metric for this contest is a version of macro F1 that equally weights two components:

Binary F1 on whether the gesture is one of the target or non-target types.
Macro F1 on gesture, where all non-target sequences are collapsed into a single non_target class
The final score is the average of the binary F1 and the macro F1 scores.

If your submission includes a gesture value not found in the train set your submission will trigger an error.

Submission File
You must submit to this competition using the provided evaluation API, which ensures that models perform inference on a single sequence at a time. For each sequence_id in the test set, you must predict the corresponding gesture.
