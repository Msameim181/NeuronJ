# NeuronJ

This is an implementation of NeuronJ work with python. NeuronJ is a plug-in for ImageJ that allows you to create and edit neurons masks.

- ImageJ: https://imagej.nih.gov/ij/
- NeuronJ: https://imagescience.org/meijering/software/neuronj/
----
## Why We Made It

Since we are not a neuroscientist, we wanted to make a tool that would be useful for our research, would be easy to use, and would be easy to modify. But the main reason we made it was if you want to process a lot of data and you want to create a mask for each image and save them, it could take a lot of your time with NeuronJ and ImageJ.

You may say ImageJ can be fast with scripts, But it isn't. It will not help you to do everything. So due to the lack of time, we made this.

## How to use
### Step 1: prepare the data
For this step, your Raw data folder should contain the following files:

For each sample:
- sample_name.tif
- sample_name.ndf


**Notes:**

- The name of the image and mask (trace) sample should be the same as each other.
- '.ndf' file must be in the same format as NeuronJ format.
- This folder is your <u>Raw data</u> folder.

### Step 2: Where to save the data
Consider where you want to save your data. Don't worry, we do all the work for you and creat folders for you. Just tell us where.

The folling folders will be created:
- Output folder you specified
- Output folder for images as <u>images</u>
- Output folder for masks as <u>masks</u>
- Output folder for image and masks together as <u>image_mask</u>

**Notes:**
- image_mask: the mask will be on the image for show case.


### Step 3: Run the script
You just need to replace the path of your Raw data folder and the output folder in the script. Our default path is:
- Raw data folder: "./data/rawData/"
- Output folder: "./data/dataSet/"

```
NeuronJ(data_addr:str, 
        output_dir:str,
        image_dir_name:str = 'images', 
        mask_dir_name:str = 'masks', 
        image_ext:str = '.tif', 
        mask_ext:str = '.tif',
        colorize:bool = True)
```


## Features

- Save results as every image format. Default is TIFF (.tif).
- Show the mask on the image.
- Colorize the mask. It can be black and white or colorful.
- Easy to modify the script as you need.
- Perfect documentation. Easy to undrestand. Easy to use.
- Quickly and good performance.
- Functional with good representation.


## Support
Please feel free to contact us if you have any questions. If you have any problem or idea, please leave us an issue on [Github](https://github.com/Msameim181/NeuronJ/issues).

If you enjoyed it, I appreciate it if you bless us with your star. Thank you for your support.


## To Do
- Add some samples (.ndf files, images and masks) for represent the work.
