# Snackathon Image Classification

## Project
DeliciousAI (https://deliciousai.com/) is a company that uses computer vision for inventory management. As such, one important problem they face is accurate classification of product images. 

In this project, we have 99 classes (UPC labels for 99 different beverages). As training data, we have images of beverages in vending machines. These images can be very zoomed in, partially obscured, etc.

## Our Solution

Our code loads the data, resizes it, and trains a classifier by tuning a pre-trained ResNet152.
