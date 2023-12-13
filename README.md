# LangChainAgent

一些有用的链接

[LangChain Chat](https://chat.langchain.com)：API/文档问题查询问答

[LangChain中文社区](https://www.langchain.cn)：近期不怎么活跃

[Huggingface MTEB榜单：向量模型选型](https://huggingface.co/spaces/mteb/leaderboard)



## TODOLIST
- [ ] 提升向量索引的正确性：找得对
  - [x] 构建测试数据集v1: 10条, 简单任务，标题召回
  - [x] 向量索引自动化测试脚本: 20%，query按序出现在原文
  - 建索引：规范文档、分词器
    - [x] 向量模型选型：text2vec-base-chinese -> gte_base_zh(当前C-MTEB top2): 20%->60%
    - [ ] 规范文档：当前按页读pdf，数据语义也不连续，考虑优化
    - [ ] 转向量之前：让GPT理解这段文本的内容，提出一些可能跟这段文本有关的问题，然后将这批问题以注释的形式放在文档内。这样子用户提问时，向量搜索出来的准确度可以提高非常多
  - 索引方式：向量距离计算
    - [ ] 子index索引：将query拆分多个子查询，分别召回 
    - MMR算法：Maximal Marginal Relevance(最大边际相关性)，牺牲部分准确性，控制推荐算法多样性
  - RetrievalQA研究
- [ ] 并发和异步
- [ ] 出处



## Notes

### 文本分词器
**知识点**
- 分词：不同的分词器使用不同的分词词表。词表大小影响分词的token数量，从而影响模型推理性能。
- 大语言模型的输入有最大长度有限制，所以要对长文本分块chunk
  - chunk_size: 控制每个chunk的最大长度
  - chunk_overlap: 相邻chunk之间的重叠token数量，保证语义的连惯性

例子1: 理解chunk_size和separator的关系

> 以如下文本为例，以chunk_size = 8，separator = ‘,’，chunk_overlap=0 为参数：
> 
> 输入: "阿斯顿发阿萨德，大沙发靠阿道夫，达到，而且，发二，切尔奇奇情况， 安三胖放进去为大发放安抚， 阿阿道夫，爱欧吃矛，五六千"
> 
> 输出: ['阿斯顿发阿萨德', '大沙发靠阿道夫', '达到，而且，发二', '切尔奇奇情况', '安三胖放进去为大发放安抚', '阿阿道夫', '爱欧吃矛，五六千']
> 
> 1、将文本以中文符逗号进行分割。
> 
> 2、分割后的文本块，与前面的chunk合并，当前chunk文本长度小于8，则可以继续进行合并（合并时分割符也计算长度），比如“达到”,“而且”,“发二”，单个文本的长度小于8，合并之后也不大于8，所以会将这三个合并到一起当做一个文本。
> 
> 3、如果分割之后的文本长度大于8，比如"安三胖放进去为大发放安抚"，这种大于8的长度的文本因为无法按照逗号分割，所以也只能保留下来作为一个文本。这种情况分词器会打日志报错打醒。

例子2: 理解chunk_overlap
```
>>> textspliter = CharacterTextSplitter(separator="，", chunk_size=10, chunk_overlap=1)
>>> textspliter.split_text(s)
Created a chunk of size 13, which is longer than the specified 10
['阿斯顿发阿萨德', '大沙发靠阿道夫，达到', '而且，发二', '切尔奇奇情况', '安三胖放进去为大发放安抚', '阿阿道夫，爱欧吃矛', '五六千']
```

**分词器**
CharacterTextSplitter

RecursiveCharacterTextSplitter（重叠滑窗分句法）

NLTKTextSplitter

SpacyTextSplitter



## 第一个LangChain例子

**原始pdf文档内容**

<img width="529" alt="image" src="https://github.com/l-i-p-f/LangChainAgent/assets/18002343/b35d031e-d27a-40f1-a920-03719791752f">

**使用组件**
- PyPDF
- CharacterTextSplitter
  - separator: '。\n', chunk_size: 800, chunk_overlap: 200
- Embedding model
  - shibing624/text2vec-base-chinese
- FAISS
- Openai-GPT3.5

**输出**

"A:"后面为模型回答部分，效果还不错。主要取决于索引检索的准确性。

```
Q: 终端连不上Wi-Fi信号
Searched:
=========
0. ----结束
2.1.2 终端连不上 Wi-Fi信号
现象描述
使用Leader AP 配置Wi-Fi网络后，部分终端在连接 Wi-Fi上网时，搜不到 Wi-Fi信号；
或者搜到 Wi-Fi信号后，无法连接成功。
可能原因
如果终端上搜不到 Wi-Fi信号，可能是如下原因。
●开启了 Wi-Fi名称（ SSID）隐藏功能。
● Wi-Fi 名称配置错误。
● Wi-Fi 信号强度配置过小。
如果终端上可以搜到 Wi-Fi信号，但连不上，可能是如下原因。
● Wi-Fi 密码输入错误。
●配置了终端黑白名单功能。
● DHCP 地址池 IP地址耗尽。
操作步骤
●手机APP方式
a.关闭Wi-Fi名称（ SSID）隐藏功能，终端重新搜索 Wi-Fi信号。
# 进入Leader AP 的管理界面，点击“ Wi-Fi设置”，取消勾选“隐藏该网络
不被发现”。WLAN
维护宝典（分销） 2 故障处理： Leader AP 类问题
文档版本  01 (2023-11-16) 版权所有  © 华为技术有限公司 4 b.修改Wi-Fi名称，终端重新搜索 Wi-Fi信号。
# 进入Leader AP 的管理界面，点击“ Wi-Fi设置”，修改 Wi-Fi名称。WLAN
维护宝典（分销） 2 故障处理： Leader AP 类问题
文档版本  01 (2023-11-16) 版权所有  © 华为技术有限公司 5 c.修改Wi-Fi的信号强度，终端重新搜索 Wi-Fi信号

1. # 使用有线终端接入网络测速，查看测速结果。如果测速较低，则需联系运
营商检查网络，或者升级出口带宽。
●Web网管方式
a.将终端接入 5G频段的 Wi-Fi信号。
# 在手机的 WLAN设置中，可以查看已连接的 Wi-Fi信息，包括建链速率（连
接速度），信号频段（频率）， Wi-Fi标准（ WLAN能力）。不同型号的手
机，显示信息有所差异。WLAN
维护宝典（分销） 2 故障处理： Leader AP 类问题
文档版本  01 (2023-11-16) 版权所有  © 华为技术有限公司 17 # 登录Web网管，进入“监控  > 概览”，在用户版块点击测速低的终端，在
下面的详情里查看当前连接的 Wi-Fi信号的频段， Wi-Fi标准，协商速率等信
息。WLAN
维护宝典（分销） 2 故障处理： Leader AP 类问题
文档版本  01 (2023-11-16) 版权所有  © 华为技术有限公司 18 # 如果接入 Wi-Fi信号的是 2.4G，则重新接入 5G频段的 Wi-Fi信号。
i.在终端的 WLAN设置中，忘记已连接的 WLAN网络，重新搜索和连接，
再检查频段是否为 5G。
ii.如果频段仍为 2.4G，则需要先确认下终端型号是否支持 5G频段。建议更
换支持 5G频段的手机进行测速。
iii.如果手机支持 5G频段，但仍接入 2.4G频段的信号，则可以修改 5G频段的
Wi-Fi名称，在 WLAN设置中搜索和连接 5G频段的 Wi-Fi信号。
# 登录Web网管，进入“配置  > 无线网络配置”。
# 单击“新建”按钮，配置新的 SSID，选中 5G射频。WLAN
维护宝典（分销） 2 故障处理： Leader AP 类问题
文档版本  01 (2023-11-16) 版权所有  © 华为技术有限公司 19 b.将5G频宽修改成 160 MHz

2. # 在手机的 WLAN设置中，可以查看已连接的 Wi-Fi信息，包括建链速率（连
接速度），信号频段（频率）， Wi-Fi标准（ WLAN能力）。不同型号的手
机，显示信息有所差异。
# 打开APP，首页上方可以看到当前连接的 Wi-Fi信号的频段， Wi-Fi标准，协
商速率等信息。
# 如果接入 Wi-Fi信号的是 2.4G，则重新接入 5G频段的 Wi-Fi信号。
i.在终端的 WLAN设置中，忘记已连接的 WLAN网络，重新搜索和连接，
再检查频段是否为 5G。
ii.如果频段仍为 2.4G，则需要先确认下终端型号是否支持 5G频段。建议更
换支持 5G频段的手机进行测速。WLAN
维护宝典（分销） 2 故障处理： Leader AP 类问题
文档版本  01 (2023-11-16) 版权所有  © 华为技术有限公司 14 iii.如果手机支持 5G频段，但仍接入 2.4G频段的信号，则可以修改 5G频段的
Wi-Fi名称，在 WLAN设置中搜索和连接 5G频段的 Wi-Fi信号。
从APP进入Leader AP 的管理界面，点击“ Wi-Fi设置”，取消勾选“ 5G
优选”，修改“ 5G Wi-Fi ”的“ Wi-Fi名称”。
b.将5G频宽修改成 160 MHz 。
# 点击“本地管理网络  > 选择网络卡片  > 查看更多  > 极速漫游”，建议组网
的AP数量小于 8个时，开启极速漫游功能。WLAN
维护宝典（分销） 2 故障处理： Leader AP 类问题
文档版本  01 (2023-11-16) 版权所有  © 华为技术有限公司 15 c.检查网线的物理协商速率是否为千兆及以上

3. 步骤5部分终端的网卡驱动版本较低，无法搜索到 Wi-Fi6信号，可以参考 关于Intel 部分无线
网卡无法接入 11ax协议AP问题的预警 进行排查和处理。
----结束
5.3 终端上线失败
步骤1执行命令 display station online-fail-record sta-mac  sta-mac，查看用户上线失败
原因

=========
A:
可能原因包括：
1. Wi-Fi名称（SSID）隐藏功能开启。
2. Wi-Fi名称配置错误。
3. Wi-Fi信号强度配置过小。
4. Wi-Fi密码输入错误。
5. 配置了终端黑白名单功能。
6. DHCP地址池IP地址耗尽。

操作步骤：
1. 使用手机APP方式：
   a. 关闭Wi-Fi名称（SSID）隐藏功能，在Leader AP的管理界面中点击“Wi-Fi设置”，取消勾选“隐藏该网络不被发现”选项。
   b. 修改Wi-Fi名称，在Leader AP的管理界面中点击“Wi-Fi设置”，修改Wi-Fi名称。
   c. 修改Wi-Fi的信号强度，在Leader AP的管理界面中点击“Wi-Fi设置”，修改Wi-Fi信号强度。

2. 使用有线终端接入网络测速，查看测速结果。如果测速较低，则需要联系运营商检查网络，或者升级出口带宽。

3. 使用Web网管方式：
   a. 将终端接入5G频段的Wi-Fi信号，在手机的WLAN设置中查看已连接的Wi-Fi信息，将终端连接到5G频段的Wi-Fi信号。
   b. 登录Web网管，进入“监控 > 概览”，在用户版块点击测速低的终端，在详情中查看当前连接的Wi-Fi信号的频段、Wi-Fi标准、协商速率等信息。
   c. 如果接入Wi-Fi信号的是2.4G频段，则需要重新接入5G频段的Wi-Fi信号。在终端的WLAN设置中，忘记已连接的WLAN网络，重新搜索和连接，并检查频段是否为5G。

4. 将5G频宽修改为160 MHz，在手机的WLAN设置中查看已连接的Wi-Fi信息，将5G频宽修改为160 MHz。

5. 检查网线的物理协商速率是否为千兆及以上。

如果终端上线失败，请执行命令display station online-fail-record sta-mac sta-mac，查看用户上线失败原因。
Finished.
```
