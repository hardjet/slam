
# openvslam深入解析系列博客相关代码

博客地址:https://hardjet.cn

openvslam地址：https://github.com/xdspacelab/openvslam


## 文件说明

>
|--3rd              使用到的第三方库，与openvslam一致
|--cmake            程序依赖库的cmake
|--config_files     测试程序使用的一些配置文件
|--imgs_sample      测试程序使用到的一些图片
|--match            特征点匹配示例
>

## 示例

### 特征点匹配
博客链接：[第二篇 特征点匹配以及openvslam中的相关实现详解](https://www.cnblogs.com/hardjet/p/11448272.html)

> orb_vocab.dbow2为词库文件，具体见openvslam。实际上这个示例并没有用到这个文件，偷懒没有改程序～

运行测试程序示例：

```shell
./test_match -v /home/anson/sdb3/dataset/slam/orb_vocab/orb_vocab.dbow2 \
-i /home/anson/work/vslam/vslam/imgs_sample \
-c /home/anson/work/vslam/vslam/config_files/kitti/KITTI_mono_00-02.yaml \
--debug
```

