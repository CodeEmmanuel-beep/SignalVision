# LISA Traffic Light Dataset Exploration

Exploration and preprocessing analysis of the **LISA Traffic Light Dataset** for a traffic-light object detection system.

The goal of this stage was to understand the dataset structure, annotation format, class distribution, temporal characteristics, and potential issues before converting the annotations into a YOLO-compatible format.

## Dataset

The LISA Traffic Light Dataset contains traffic-light annotations collected from video sequences across different environments, including daytime and nighttime scenes.

The dataset is organized into groups such as:

* `dayTrain`
* `nightTrain`
* `daySequence1`
* `daySequence2`
* `nightSequence1`
* `nightSequence2`

Annotations are provided as CSV files containing bounding-box coordinates and traffic-light states.

## Exploration

The exploration focused on:

* Understanding the dataset directory structure.
* Inspecting annotation files and their columns.
* Identifying the available traffic-light classes.
* Measuring class and frame distributions.
* Checking how many traffic lights appear in individual frames.
* Investigating the relationship between annotations and video clips.
* Identifying potential temporal leakage during dataset splitting.
* Validating image-to-annotation mapping.
* Converting the original annotations into a simplified three-class representation.
* Converting bounding boxes into YOLO format.
* Visually validating the generated bounding boxes.

## Original Classes

The LISA annotations contain seven traffic-light states:

| Original Class | Annotation Count |
| -------------- | ---------------: |
| `go`           |           46,723 |
| `stop`         |           44,318 |
| `stopLeft`     |           12,734 |
| `warning`      |            2,669 |
| `goLeft`       |            2,476 |
| `warningLeft`  |              350 |
| `goForward`    |              205 |

The dataset contains significant class imbalance, particularly for yellow traffic lights.

## Target Classes

For the object detection task, the original classes are consolidated into three traffic-light colors:

| Target ID | Color  | Original Classes            |
| --------: | ------ | --------------------------- |
|       `0` | Red    | `stop`, `stopLeft`          |
|       `1` | Yellow | `warning`, `warningLeft`    |
|       `2` | Green  | `go`, `goLeft`, `goForward` |

After consolidation:

| Target Class | Annotation Count |
| ------------ | ---------------: |
| Red          |           57,052 |
| Yellow       |            3,019 |
| Green        |           49,404 |

Yellow represents a relatively small portion of the annotations, making it an important class to monitor during model training and evaluation.

## Multiple Objects Per Frame

The dataset contains many frames with multiple traffic lights.

Among the annotated frames:

| Number of Classes Present | Frames |
| ------------------------: | -----: |
|                         1 | 22,220 |
|                         2 | 13,404 |
|                         3 |    641 |

There are therefore many multi-object detection examples rather than simply one traffic light per image.

Across the dataset, there are:

* **36,265 annotated frames**
* **109,475 traffic-light annotations**
* Approximately **3 annotations per annotated frame**

The YOLO label format therefore contains multiple lines for images containing multiple traffic-light objects.

## Temporal Structure

Because LISA is derived from video sequences, consecutive frames can be highly similar.

A random image-level train/validation/test split could therefore place nearly identical frames from the same sequence into different splits, resulting in temporal leakage and overly optimistic evaluation.

The dataset should therefore be split at the **clip/sequence level**, ensuring that frames from the same video sequence do not appear across different splits.

## Annotation Validation

The original LISA bounding boxes use:

* Upper-left X/Y coordinates
* Lower-right X/Y coordinates

These were converted to the normalized YOLO format:

```text
class_id x_center y_center width height
```

with all coordinates normalized to the image dimensions.

The conversion produced:

```text
Missing annotations: 0
Invalid annotations: 0
Written annotations: 109,475
Label files: 36,265
```

This confirms that all processed annotations successfully mapped to images and produced valid YOLO labels.

## Visual Validation

Generated YOLO annotations were loaded back onto the original images and visualized with their bounding boxes.

The visual inspection checks that:

* Bounding boxes are positioned over the correct traffic lights.
* Bounding-box dimensions correspond to the annotated objects.
* Multiple objects in the same frame are correctly represented.
* Class labels correspond to the intended traffic-light colors.
* Normalization and coordinate conversion did not introduce systematic offsets.

## Key Findings

The exploration established several important characteristics of the dataset:

1. **The dataset is suitable for object detection.**
   It contains a large number of annotated traffic-light objects across thousands of frames.

2. **The original seven classes can be consolidated into three color-based classes.**
   This simplifies the detection problem to red, yellow, and green.

3. **Yellow is significantly underrepresented.**
   It accounts for only a small fraction of the total annotations and will require particular attention during model evaluation.

4. **Multiple traffic lights frequently occur in the same frame.**
   The model therefore needs to perform genuine multi-object detection.

5. **The data is temporally correlated.**
   Train/validation/test splitting must be performed by clip or sequence rather than by individual frame.

6. **The annotation conversion was successful.**
   All processed annotations mapped to images without missing or invalid bounding boxes.

7. **Visual inspection confirmed the generated bounding boxes are usable for the next stage of the pipeline.**

## Next Step

With dataset exploration and annotation validation complete, the project can proceed to the modeling pipeline:

```text
Dataset Exploration
        ↓
Clip-Aware Train/Validation/Test Split
        ↓
YOLO Dataset Structure
        ↓
Baseline YOLO Training
        ↓
Per-Class Evaluation
        ↓
Error Analysis
        ↓
Targeted Improvements
```

The baseline model will provide the first objective measurement of detection performance, with particular attention given to **yellow traffic-light detection**.
