# RSI-LS-VHR-2-DATASE
 
To advance performance evaluation research in remote sensing object detection, we built the Remote Sensing Imagery of Large-Scale-VHR-2 categories (RSI LS-VHR-2) dataset, which is much larger than most existing datasets in this field. Table Ⅰ lists the details of the dataset for two categories, aircraft and ship. 

![image](https://github.com/Damon2019/RSI-LS-VHR-2-DATASET/blob/master/ship.png)

### TABLE Ⅰ. DESCRIPTION OF THE RSI LS-VHR-2 DATASET

Label	   |  Name	  |  Total instances	   | Complete instances	  |  Fragmentary instances	  |  Scene class	  |  Images	   |  Image width	  |  Sub-images
 :-----:  | :-----:  |  :-----:   |  :-----:   |  :-----:   |  :-----:    |  :-----:    |  :-----:   |  :-----:  
1 	 |   aircraft	   |   103917	  |   85975    | 	17942	   |   203	   |   2858	    |    6000-15000	    |   62129
2	   |   ship	   |     68436	       |   54386	  |   14050	   |   30	    |   397	         |    5000-18000	   |   53860

As shown in Table Ⅰ, the RSI LS-VHR-2 dataset has four notable characteristics: 
1) Rich image variability: this dataset is collected from different sensors and platforms and includes 203 airports and 30 harbors.
2) Large scale: the width and height of each original image varies from 5000 to 18,000 pixels and contains objects exhibiting a wide variety of scales, orientations, and shapes.
3) Abundant instances: the dataset consists of 172,353 positive samples (103,917 aircrafts and 68,436 ships) obtained from 3255 large-scale remote sensing images distributed in 115,989 sub-images cropped from the original large-scale images.
4) Multiple target difference: an additional 31,992 fragmented instances were added to the dataset for data augmentation to test the capacity of trained models to detect incomplete targets.
All the original large-scale images were cropped with a non-overlapping sliding window to generate sub-images. To facilitate feature extraction, the sub-image size is a uniform 600×600 pixels.

### TABLE Ⅱ.  DETAILS OF THE TEST IMAGES

Label	  |  Scale(pixels)	  |   Images	  |  Instances	|  Sub-images
 :-----:    |  :-----:    |  :-----:    |  :-----:   | :-----:
aircraft  |8000 x 8000	   |   5	   |    272    |   980
ship   |8000 x 8000	   |   5	   |    225    |   980


## version：RSI-LS-VHR-2-DATASE - v1.0
It contains all the images  for training and verification!

We will continue to improve it in the future.
