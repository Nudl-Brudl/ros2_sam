# ROS SAM

This package is what the name suggests: Meta's `segment-anything` wrapped in a ROS node.

## Installation

Installation is easy: 
 1. Start by cloning this package into your ROS environment. 
 2. Download the (checkpoints)[https://github.com/facebookresearch/segment-anything#model-checkpoints] for the desired SAM models to the `models` directory in this package.
 3. Install SAM by running `pip install git+https://github.com/facebookresearch/segment-anything.git`.

## Using ROS SAM standalone

Run the SAM ROS node using `rosrun`:

```bash
rosrun ros_sam sam_node.py
```

The node currently offers a single service `ros_sam/segment` which can be called to segment an image. Check `rossrv show ros_sam/Segmentation` for request and response specifications.

You can test SAM by starting the node and then running `rosrun ros_sam sam_test.py`.

## ROS SAM Services

`ros_sam` offers a single service `segment` of the type `ros_sam/Segmentation.srv`. The service definition is
```
sensor_msgs/Image        image            # Image to segment
geometry_msgs/Point[]    query_points     # Points to start segmentation from
int32[]                  query_labels     # Mark points as positive or negative samples
std_msgs/Int32MultiArray boxes            # Boxes can only be positive samples
bool                     multimask        # Generate multiple masks
bool                     logits           # Send back logits

---

sensor_msgs/Image[]   masks            # Masks generated for the query
float32[]             scores           # Scores for the masks
sensor_msgs/Image[]   logits           # Logit activations of the masks
```

The service request takes input image, input point prompts, corresponding labels and the box prompt. The service response contains the segmentation masks, confidence scores and the logit activations of the masks.

To learn more about the types and use of different queries, please refer to the [original SAM tutorial](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb)

The service calls are wrapped up conveniently in the ROS SAM client

## Using ROS SAM client

Alternatively, one can use the ROS SAM client instead of the service calls.

Initialize the client with the service name of the SAM segmentation service
```python
from ros_sam import SAMClient
sam_client = SAMClient('ros_sam')
```

Call the segment method with the input image, input prompt points and corresponding labels. This returns 3 segmentation masks for the object and their corresponding confidence scores
```python
img = cv2.imread('path/to/image.png')
points = np.array([[100, 100], [200, 200], [300, 300]])
labels = [1, 1, 0]
masks, scores = sam_client.segment(img, points, labels)
```

Additional utilities for visualizing segmentation masks and input prompts
```python
from ros_sam import show_mask, show_points
show_mask(masks[0], plt.gca())
show_points(points, np.asarray(labels), plt.gca())
```
<img src="https://github.com/ARoefer/ros_sam/blob/devel/harsha/figures/segmentation-example.png" width=50% height=50%>

## Using with RQT click interface

To use the GUI install the following in your ROS workspace:

[rqt_image_view_seg](https://github.com/ipab-slmc/rqt_image_view_seg)

Run the launch file:

`roslaunch ros_sam gui_test.launch`

Check the terminal and wait until the SAM model has finished loading.

There will be two windows loaded. One will have the header `rqt_image_view_seg__ImageView` and the other `rqt_image_view__ImageView`, note the lack of `_seg`. The first window is where you should click, so select the topic of the camera you want to view from the drop down. In the second window you should select `/rqt_image_segmentation/masked_image`. This is where the segmented image will be displayed.

<img src="https://github.com/ARoefer/ros_sam/blob/devel/russell/figures/interface-example.png" width=50% height=50%>
