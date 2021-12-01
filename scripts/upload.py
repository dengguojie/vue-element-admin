#!/usr/bin/python
# -*- coding:utf-8 -*-
# Copyright 2019 Huawei Technologies Co.,Ltd.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License.  You may obtain a copy of the
# License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations under the License.



from __future__ import print_function
from obs import ObsClient, PutObjectHeader
import sys
import os
AK = 'PRVDTCWGFZ3PSGFRVCIX'
SK = 'tA3WyyoWzQhWq26M015C7cgzU1GUOXj7wBzvIyo7'
server = 'https://obs.cn-north-4.myhuaweicloud.com/'

bucketName = 'ops-transfer'
lib_file = sys.argv[1]
username = sys.argv[2]
libs = ["liboptiling.so", "libopsproto.so"]
for l in libs:
    if l in lib_file:
        lib = l
# Constructs a obs client instance with your account for accessing OBS
obsClient = ObsClient(access_key_id=AK, secret_access_key=SK, server=server)
try:
     
    headers = PutObjectHeader() 
    headers.contentType = 'text/plain' 
    obj_name = "{}/{}".format(username,lib)
    upload_file  = "{}/{}".format(sys.argv[3], lib_file)
    resp = obsClient.putFile(bucketName, obj_name, upload_file) 
          
    if resp.status >= 300: 
        print('errorCode:', resp.errorCode) 
        print('errorMessage:', resp.errorMessage)
        
except:
    import traceback
    print(traceback.format_exc())
    
try:
    resp = obsClient.listObjects(bucketName) 
     
    if resp.status < 300: 
        file_list=[content.key for content in resp.body.contents]
        if obj_name in file_list:
            print("https://ops-transfer.obs.cn-north-4.myhuaweicloud.com/{}".format(obj_name))
        else:
            print("upload fail")
    else: 
        print('errorCode:', resp.errorCode) 
        print('errorMessage:', resp.errorMessage)
except:
    import traceback
    print(traceback.format_exc())



