#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import csv
import cv2

if __name__ == "__main__":
  url_list = sys.argv[1]
  r = open(url_list, "r")
  csv_reader = csv.reader(r)
  for urls in csv_reader:
    filename = urls[0].split('/')[-1]
    img = cv2.imread('images/' + filename)
    cv2.imwrite('images/convert/' + filename + ".jpg", img)
