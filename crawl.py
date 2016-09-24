#!/usr/bin/env python
# -*- coding: utf-8 -*-

import urllib.request, urllib.parse
import sys
import os
import lxml.html
import base64
import json
import re

def build_url(skip=0):
  query ={}
  url = "https://api.datamarket.azure.com/Bing/Search/v1/Image"
  query["Query"]="'ネコ'"
  query["Market"]="'ja-JP'"
  query["$format"]="json"
  if skip != 0:
    query["$skip"]=skip
  params = urllib.parse.urlencode(query)
  req_url = url+'?'+params

  return req_url

def get_image_url(req_url, headers):
  try:
    req = urllib.request.Request(req_url, headers=headers)
    json_str = urllib.request.urlopen(req).read().decode("utf-8")
    json_data = json.loads(json_str)
    
    #URLをファイルに書き込む
    f = open("url_list.txt", "a")
    for ret in json_data["d"]["results"]:
      f.write(ret["MediaUrl"] + "\n")
    f.close()

    if "__next" in json_data["d"]:
      print(json_data["d"]["__next"])
      # json_data["d"]["__next"]の中に次のページのURLが入っているけど、そのまま使うとxmlで返却されてしまう
      # そのため$format=jsonを付け加える必要がある
      next_url = json_data["d"]["__next"]

      pattern = "\$skip=(\d*)"
      matchOB = re.search(pattern, next_url)
      if matchOB:
        skip_number = matchOB.group().split("=", 1)
        next_url = build_url(skip_number[1])

        get_image_url(next_url, headers)
      
  except urllib.error.HTTPError as e:
    print(e.read())

if __name__ == "__main__":
  req_url = build_url()

  user = os.environ.get("MS_ACCOUNT_ID") 
  password = os.environ.get("MS_ACCOUNT_PRAYMARY_KEY") 

  encoded = str(base64.b64encode(bytes(user + ':' + password ,'utf8')),'utf-8')

  headers = {
    "Authorization":"Basic %s" % encoded
  }

  get_image_url(req_url, headers)
