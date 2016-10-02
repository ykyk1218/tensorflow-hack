#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import csv

if __name__ == "__main__":
  url_list = sys.argv[1]
  type_num = sys.argv[2]
  r = open(url_list, "r")
  csv_reader = csv.reader(r)
  t = open("tensorflow_image.csv", "a")

  for urls in csv_reader:
    filename = urls[0].split('/')[-1]
    t.write(filename + "," + type_num + "\n")
  t.close()


