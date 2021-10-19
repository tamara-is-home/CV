To see how each method with each descriptor works call utils.py
Affine transformation example may be seen by running affine_transform.py
To run the main code part call main.py

MODS itself is implemented in mods.py and it works as follows:
we have 3 detectors - OBR, MSER, BRISK. 
we perform iterations trying to match two images.
On each iteration (except of the first one) we do some affine transformation of the images, to change the view angle and rotate them
Then we pass those images thru mathcing pipeline: detect --> describe --> match --> ransac
On the first and second iteration we use OBR, then MSER for 3 to 5 iter, then BRISK (in original MODS papers there was HessAff but had to think of change to that guy)
each time we compute amount of good matches using RANSAC.
we stop to iterate if good matches rate is good enough for us of if max allowed iterations is exceeded.

In main.py we do this procedure with every image from EVD and WxBS datasets, return the best found H (Homography) matrix and see how it goes if we wrap out image with that H.
We plot best matches to see how we perform
Also we can compare H with the given ground-truth H by wrapping image with true H and by calculating EUC distance between True H and H we found.
