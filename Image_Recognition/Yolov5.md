```
# to see all the options, see detect.py
python detect.py --conf-thres 0.01 --iou-thres 0.45 --source test_images --save-crop --project Results --agnostic-nms
```
- `iou_thres` (default 0.45) is NMS (Non Maximum Suppression) IOU (Intersetion over Union) threshold.  
`IOU(Box1, Box2) = Intersection_Size(Box1, Box2) / Union_Size(Box1, Box2)`  
NMS Alg ([source](https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/#:~:text=Non%20Maximum%20Suppression%20(NMS)%20is,arrive%20at%20the%20desired%20results.)):
  - Step 1: Select the prediction S with highest confidence score and remove it from P and add it to the final prediction list K. (K is empty initially).
  - Step 2: Now compare this prediction S with all the predictions present in P. Calculate the IoU of this prediction S with every other predictions in P. If the IoU is greater than the threshold thresh_iou for any prediction T present in P, remove prediction T from P.
  - Step 3: If there are still predictions left in P, then go to Step 1 again, else return the list keep containing the filtered predictions.
