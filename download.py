#!/usr/bin/env python
# -*- coding: utf-8 -*-

import urllib.request, urllib.parse
import sys
import os
import csv

def download(url, path):
  print('ダウンロード開始: ' + url)
  try:
    downloa_data = urllib.request.urlopen(url)
    filename = url.split('/')[-1]

    save_data = open(path + "/" + filename, 'wb')
    save_data.write(downloa_data.read())
    save_data.close()

    downloa_data.close()
  except:
    print('error: ' + url)

if __name__ == "__main__":
  url_list = sys.argv[1]
  save_path = sys.argv[2]
  r = open(url_list, "r")
  csv_reader = csv.reader(r)
  for i in csv_reader:
    download(i[0], save_path)
