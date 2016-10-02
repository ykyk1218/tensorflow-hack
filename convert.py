#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import csv
from PIL import Image

if __name__ == "__main__":
  url_list = sys.argv[1]
  r = open(url_list, "r")
  csv_reader = csv.reader(r)
  for urls in csv_reader:
    filename = urls[0].split('/')[-1]
    print(filename)
    if os.path.isfile('images/' + filename):
      try:
        img = Image.open('images/' + filename, 'r')
        img.save('images/convert/' + filename + ".jpg", 'JPEG', quality=100, optimize=True)
      except:
        print(filename + "でエラー")
