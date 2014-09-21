# -*- coding: utf-8 -*-
#!/usr/bin/env python
#
#__author__= 'ihciah@gmail.com'
#__author__= 'BaiduID-ihciah'
#__author__= 'http://www.ihcblog.com'
import os,sys,urllib2

def download(i):
    s=urllib2.urlopen('http://xk.fudan.edu.cn/xk/image.do').read()
    f=file('pic/'+str(i)+'.jpg','wb')
    f.write(s)
    f.close()

for i in range(eval(sys.argv[1]),500):
    download(i)
    print i
