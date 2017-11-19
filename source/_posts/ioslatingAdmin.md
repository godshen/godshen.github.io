---
title: hexo前后端分离并存git仓库 
date: 2017-11-19 15:57:54
tags: 
- git
- branch
- hexo管理
cotagries:
- 建站
---

#缘由
由于hexo框架在执行了generate以及deploy之后, 上传到远端仓库的仅仅是渲染出来的html, 而可运行的工程文件夹还保存在本地无法灵活增删改

#选择
- 有大神选择使用APS在线搞了一套git的交互界面, 通过hook等技术, 同步远端和本地, 使我们无论是在PC端, 还是在云端都可以自由编辑; 但是其局限性也很明显--壕, APS不是每个人都有条件搞的;
- 另外一波大神提供了这个思路, 就是在页面的仓库上面, 独立于master分支, 再新建一个branch, 把源代码push到这里, 自由checkout, 也是爽的飞起来的, 当然啦, 其本质和再建一个仓库, 一个存html, 一个存.md, 是一样的;

#操作
1. 把原来的gitcline下来
2. 打开git bash, 或者习惯用交互界面的也无所谓
3. 新建一个分支branch
4. 从master分支 git checkout 切换到这个source分支
5. 先commit到远端, 省去后续麻烦
5. git rm 删除掉所有关于页面的
6. 把执行hexo init时候生成的那个文件夹, 除去.deploy_git这个文件夹, 统统复制粘贴过来
7. git add 所有的内容, 第一次可能很慢
8. git commit
9. git push
10. done!