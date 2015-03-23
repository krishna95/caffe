* The mat files are the label files. 
* Depending upon the number of samples in the train.txt and test.txt, I have divided the labels manually. This could be modified at a later stage.
* The text files contains the location of images. In my case it was stored in Downloads Section.
* This text file can be made using
**find `location_images` -type f -exec echo {} \; > train.txt**
* If your path for the images is different then you need to create this file again.
* To run the code 
```bash
$ python feature_extraction.py
```

