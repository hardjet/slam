
# openvslam深入解析系列博客相关代码

博客地址:http://hardjet.cn

openvslam地址：https://github.com/xdspacelab/openvslam


## 文件说明

```yaml

|--3rd                             使用到的第三方库，与openvslam一致
|--cmake                      程序依赖库的cmake
|--config_files            测试程序使用的一些配置文件
|--data                           测试程序使用的数据
|----match                    特征点匹配示例用到的数据
|------imgs_sample  测试程序使用到的一些图片
|------indemend         indemind相机数据
|------realsense           realsense D435相机数据
|--match                       特征点匹配示例
|----test_match          特征点匹配测试
|----orb_compare     不同的相机orb特征点匹配测试

```

## 示例

### 特征点匹配
博客链接：[第二篇 特征点匹配以及openvslam中的相关实现详解](https://www.cnblogs.com/hardjet/p/11448272.html)

运行测试特征点匹配测试：

```shell
./test_match -i /home/anson/work/vslam/vslam/data/match/imgs_sample \
-c /home/anson/work/vslam/vslam/config_files/kitti/KITTI_mono_00-02.yaml \
--debug
```

运行不同的相机orb特征点匹配测试：

```shell
./orb_compare
```

