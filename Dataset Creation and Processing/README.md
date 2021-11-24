# Dataset Creation and Processing

The creation of the datasets underlying this GAN were curated using a three step process.

1. Raw images were scraped by keyword using the Selenium package and the Chrome webdriver from one of two websites, either Google images or Tattoodo
2. A subset of each image type (Skull, Bird, Etc.) was uploaded to roboflow and manually labeled.
3. An object detection network was trained on the manually labeled images, and deployed to identify and crop out the relevant object from the remainder of the images.

## Image Scraping

The first step to correctly run the `Image Scraping.ipynb` notebook is to download the appropriate Chromedriver for your version of google Chrome, then insert the path to that driver in the notebook. The notebook downloads the images from each given search term to a folder representing the overall class.

```{python}
for j in ['scorpion', 'orca', 'skull']:
        searchterms = [j, j + ' tattoo sketch', j + ' tattoo design', j + ' tattoo drawing', j + ' tattoo flash', j + ' tattoo outline', 
                   j + ' tattoo line art', j + ' aesthetic line art tattoos', j + ' pencil art tattoo drawing', 
                   j + ' tattoo artwork',j + ' tattoo art', j + ' illustration tattoo', j + ' tattoo', j + ' idea',
                   j + ' tattoo sketch drawing ideas', 'black and white '+ j + ' drawing', 
                   j + ' sketch', j + ' drawing']
```

In the above structure `j` is the class, and the name of the directory into which the images will be saved.

As of right now, the main function is set up to download images from either google image search or Tattoodo. The primary difference between these two is as follows:

- Google requires you to scroll to the bottom of the page, and click a button to display more images. It also allows you to scrape all the images you have loaded at once.

- Tattoodo, conversely, is an endlessly scrolling page. There is no button clicking required. However, it will only allow you to scrape images it currently has loaded, which means you need to get the appropriate urls after each scroll (then filter out duplicates)

If you wish to extend this process to work with other websites, look at the two bullets above and figure out which kind of image site you are dealing with, the modify the function based on which is the most similar.

## Roboflow

[Roboflow]<https://roboflow.com/> is a website that allows for fast manual labeling of objects. Approximately 150-200 objects from the Image Scraping phase, divided into 80% train and 20% test, were uploaded to the roboflow platform and manually labeled for each of the image classes. Roboflow does require you to create an account, but you can do a lot with a free account and there are augmentation and image processing tools available as well.

If Roboflow will **not** be used, ensure that your images and annotations are in the Yolov5 format.

## Object Detection and Cropping

The `Tattoo Identification.ipynb` assumes the following:

1. You have the images uploaded on to google drive in an appropriately named folder.
2. You are running the notebook on google colab.

To use the notebook, copy and paste the link from your Roboflow labeled data. Then change the references to the appropriate folders on your google drive.

The notebook has three steps:

1. It trains a Yolov5m object identification model on the labeled roboflow data.
2. It uses that trained model to identify objects of the appropriate class in the images provided, the crops them out and lands them locally
3. It copies the local cropped images back to your google drive.