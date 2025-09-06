#imageProcessor.py
from ultralytics import YOLO
import pandas as pd
import cv2
import os
import numpy as np
from tqdm import tqdm

##Configuration
CSV_PATH = "vehicleImage.csv"