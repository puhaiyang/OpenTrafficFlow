# OpenTrafficFlow
Make traffic faster and safer.



## 数据集合格式
### CCPD-2019数据集

它的图片内容在ccpd_开头的目录下，详细文件格式如下：

> ll CCPD_Datasets/CCPD/puhaiyang___CCPD2019/CCPD2019/ccpd_base/
> -rw-rw-r-- 1 1000 1000  80339  2月  6  2019 '0212703544062-86_91-244&503_460&597-465&574_259&593_258&514_464&495-0_0_16_25_25_29_13-140-72.jpg'

(ultralytics-env) [root@xg-ragflow-node1 OpenTrafficFlow]# ll CCPD_Datasets/CCPD/puhaiyang___CCPD2019/CCPD2019/
总用量 50748
drwxrwxr-x 2 1000 1000 30953472  2月  4 16:36 ccpd_base
drwxrwxr-x 2 1000 1000  2822144  2月  4 16:16 ccpd_blur
drwxrwxr-x 2 1000 1000  6844416  2月  4 16:14 ccpd_challenge
drwxrwxr-x 2 1000 1000  1388544  2月  4 16:17 ccpd_db
drwxrwxr-x 2 1000 1000  2920448  2月  4 16:10 ccpd_fn
drwxrwxr-x 2 1000 1000    69632  2月  4 16:17 ccpd_np
drwxrwxr-x 2 1000 1000  1376256  2月  4 16:18 ccpd_rotate
drwxrwxr-x 2 1000 1000  4194304  2月  4 16:39 ccpd_tilt
drwxrwxr-x 2 1000 1000  1359872  2月  4 16:15 ccpd_weather
-rw-rw-r-- 1 1000 1000     1061  8月 25  2018 LICENSE
-rw-rw-r-- 1 1000 1000     4022  8月 25  2018 README.md
drwxrwxr-x 2 1000 1000     4096  2月  4 16:17 splits


### CCPD-2020数据集
(ultralytics-env) [root@xg-ragflow-node1 OpenTrafficFlow]# ll CCPD_Datasets/CCPD2020/puhaiyang___CCPD2020/CCPD2020/ccpd_green/
总用量 1864
drwxr-xr-x 2 root root 811008  2月  4 15:55 test
drwxr-xr-x 2 root root 929792  2月  4 15:55 train
drwxr-xr-x 2 root root 167936  2月  4 15:55 val

格式为：
(ultralytics-env) [root@xg-ragflow-node1 OpenTrafficFlow]# head -n 10| ll CCPD_Datasets/CCPD2020/puhaiyang___CCPD2020/CCPD2020/ccpd_green/train/ 
总用量 450032
-rw-r--r-- 1 root root  92934  2月  4 15:55 '00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg'
-rw-r--r-- 1 root root 133527  2月  4 15:55 '00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg'


### CCPD_BlueGreenYellow数据集

(ultralytics-env) [root@xg-ragflow-node1 OpenTrafficFlow]# ll CCPD_Datasets/CCPD_BlueGreenYellow/puhaiyang___ccpdblueyellowgreen/ccpd_blue_yellow_green | head -n 10
总用量 2122368
-rw-r--r-- 1 root root  412719  2月  4 17:06 0-0_0-0&342_719&610-714&610_0&585_15&342_719&367-29_16_2_2_30_31_28-0-0.jpg.png
-rw-r--r-- 1 root root 1233169  2月  4 17:06 0-0_0-0&355_641&631-641&631_54&501_0&355_565&485-11_14_2_2_31_27_25-0-0.jpg.png
-rw-r--r-- 1 root root 1300288  2月  4 17:06 0-0_0-0&355_641&631-641&631_54&501_0&355_565&485-11_2_32_14_13_24_26-0-0.jpg.png
-rw-r--r-- 1 root root  848302  2月  4 17:06 0-0_0-0&368_647&599-636&574_0&599_0&393_647&368-23_0_7_33_26_30_19-0-0.jpg.png
-rw-r--r-- 1 root root  781570  2月  4 17:06 0-0_0-0&368_719&666-719&666_0&537_0&368_719&497-19_12_14_18_32_31_25-0-0.jpg.png
-rw-r--r-- 1 root root  657086  2月  4 17:06 0-0_0-0&368_719&666-719&666_0&537_0&368_719&497-3_18_26_26_8_21_26-0-0.jpg.png
-rw-r--r-- 1 root root  437494  2月  4 17:06 0-0_0-0&370_676&614-667&614_0&592_0&370_676&392-19_20_32_6_24_26_31-0-0.jpg.png
-rw-r--r-- 1 root root  506120  2月  4 17:06 0-0_0-0&370_719&644-719&597_0&644_0&417_719&370-15_16_28_26_25_12_17-0-0.jpg.png
-rw-r--r-- 1 root root  671386  2月  4 17:06 0-0_0-0&370_719&644-719&597_0&644_0&417_719&370-23_6_33_26_27_5_5-0-0.jpg.png
(ultralytics-env) [root@xg-ragflow-node1 OpenTrafficFlow]# 
