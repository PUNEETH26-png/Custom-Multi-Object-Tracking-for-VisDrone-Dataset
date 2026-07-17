## Table of Contents

- [Demo](#-demo)
- [Project Objective](#project-objective)
- [Dataset](#dataset)
- [Detection Model](#detection-model)
- [Tracking Methodology](#tracking-methodology)
- [Tracking Pipeline](#tracking-pipeline)
- [Where the Tracker Works Well](#where-the-tracker-works-well)
- [Current Limitations](#current-limitations)
- [Possible Improvements](#possible-improvements)
- [Technologies Used](#technologies-used)## 🎥 Demo

## Project Structure

```text
.
├── dataset/
├── tracking_Demo/
├── main.py
├── README.md
├── requirements.txt
├── tracking.py
├── utils.py
└── visualize.py
```
## Installation

```bash
git clone https://github.com/PUNEETH26-png/Custom-Multi-Object-Tracking-for-VisDrone-Dataset
cd Custom-Multi-Object-Tracking-for-VisDrone-Dataset

pip install -r requirements.txt
```
### Multi-Object Tracking

![Tracking Demo](tracking_Demo/tracking_demo.gif)

# Custom Multi-Object Tracking for UAV Video Streams

A custom multi-object tracking pipeline for UAV video sequences using YOLOv8 for object detection and class-aware object association across consecutive frames.

The project focuses on understanding and implementing the core components of multi-object tracking instead of directly using existing tracking frameworks such as SORT, DeepSORT, or ByteTrack.

## Project Objective

The objective of this project is to detect multiple objects in UAV video frames and maintain a persistent tracking ID for each object across consecutive frames.

The pipeline handles:

- Object detection
- Class-wise object association
- One-to-one detection matching
- Persistent track IDs
- New object entry
- Temporary missed detections
- Expired track removal

## Dataset

The project uses the **VisDrone-VID dataset**, which contains video sequences captured using drone-mounted cameras.

The dataset includes challenging aerial scenarios such as:

- Small objects
- Moving cameras
- Dense scenes
- Partial occlusion
- Multiple objects of the same class
- Scale variation


## Detection Model

**YOLOv8 Nano** is used for object detection.

For each detected object, the model provides:

- Bounding box
- Predicted class
- Confidence score

Each prediction is converted into the following internal representation:

```text
(bounding_box, centre, class_index)
```

The centre of a bounding box is calculated as:

```text
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
```

The processed detections are passed to the custom tracking algorithm.

## Tracking Methodology

### 1. Initial Track Assignment

Objects detected in the first frame are assigned unique tracking IDs.

For every tracked object, the following information is maintained:

```text
centre
track_id
bounding_box
class_label
TTL
```

### 2. Class-Wise Object Association

Detections are grouped according to their predicted object class.

For example, cars are matched only with cars and persons are matched only with persons.

This prevents spatially close objects belonging to different classes from being incorrectly associated.

### 3. Cost Matrix Construction

For every object class, Euclidean distance is calculated between previous tracked object centres and current detected object centres.

The Euclidean distance is calculated as:

```text
distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)
```

The calculated distances are stored in a cost matrix.

Conceptually:

```text
                 Current Detections

                 D1      D2      D3

Previous T1      d11     d12     d13
Tracks   T2      d21     d22     d23
         T3      d31     d32     d33
```

Each value represents the centre distance between a previous track and a current detection.

### 4. One-to-One Object Assignment

SciPy's `linear_sum_assignment` function is used to perform Hungarian-based one-to-one assignment.

The assignment minimizes the total centre distance between previous tracks and current detections.

Unlike independent nearest-neighbour matching, one previous track cannot be assigned to multiple current detections.

Matched objects preserve their existing tracking IDs.

### 5. New Object Handling

Current detections that are not selected during object assignment are considered new objects.

Each new object receives a unique tracking ID and is added to the tracking state.

### 6. Missing Object Handling

Every tracked object maintains a Time-To-Live (`TTL`) value.

When an object is successfully matched:

```text
TTL = MAX_TTL
```

When an object is not matched:

```text
TTL = TTL - 1
```

When the TTL reaches zero:

```text
TTL = 0
```

the object is removed from the tracking state.

This prevents tracks from being immediately deleted because of temporary missed detections.

## Tracking Pipeline

```text
UAV Video Frame
       |
       v
YOLOv8 Detection
       |
       v
Bounding Box Centre Calculation
       |
       v
Group Detections by Class
       |
       v
Build Distance Cost Matrix
       |
       v
Hungarian-Based Assignment
       |
       v
Update Existing Tracks
       |
       +------> Add New Objects
       |
       +------> Update Missing Object TTL
       |
       v
Remove Expired Tracks
       |
       v
Visualize Tracking IDs
```

## Where the Tracker Works Well

The custom tracker performs well when:

- Object movement between consecutive frames is small or moderate.
- The video has a sufficiently high frame rate.
- Objects belong to different classes.
- Detection failures occur only for a small number of frames.
- New objects enter the video frame.
- Objects leave the frame gradually.

Class-wise object association prevents incorrect matches between different object categories.

The TTL-based track management mechanism also helps maintain object tracks during short detection failures.

## Current Limitations

The current tracking algorithm may fail when:

- Objects move a large distance between consecutive frames.
- Multiple objects of the same class move close to each other.
- Objects cross paths.
- Significant UAV camera movement occurs.
- An object remains occluded for a long duration.
- Multiple visually similar objects are present in the same region.

The current association method mainly relies on object class and bounding box centre distance.

The tracker does not currently use motion prediction or visual appearance information.

## Possible Improvements

Future improvements include:

- Motion prediction using a Kalman Filter
- Maximum-distance gating for object association
- IoU-based object matching
- Combining centre distance and bounding box overlap
- Appearance feature extraction
- Object re-identification
- Camera motion compensation
- Improved track state management


## Technologies Used

- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- NumPy
- SciPy

## Author

**M. Puneeth Raj Kumar**