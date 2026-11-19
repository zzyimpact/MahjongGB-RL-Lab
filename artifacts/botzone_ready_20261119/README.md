# Botzone 上传包：2026-11-19 部署兼容修订

本目录仅重新打包已有模型，**没有训练、调参或运行对战评估**。所有导出参数与对应训练 checkpoint 逐 tensor 完全相同。不要继续使用 round3/round4 旧代码搭配本目录的新文件名。

## 先按这四步提交

1. 在头像菜单的“管理存储空间”中，上传下表需要的 `.pt`。**保留完整文件名**，直接放用户存储根目录。不要在根目录再建一个 `data` 文件夹；程序中的 `data/` 已经指向用户存储。
2. 每个模型创建一个独立 Bot，选国标麻将 `Chinese-Standard-Mahjong`。使用136张、无花、各家独立牌墙的复式规则；不要拿这些权重测试144张含花版本。
3. 语言/运行环境选 **Python 3.6**，源码上传对应的 **`botzone_*.zip`**。ZIP根目录已有 `__main__.py`，无需解压、加外层目录或逐个复制模块。模型权重不在ZIP中。
4. **SimpleIO / 简单交互不勾选，使用JSON。** 本次入口也兼容简单交互，但建议七个Bot统一按JSON设置。

长时运行由代码每轮输出 `>>>BOTZONE_REQUEST_KEEP_RUNNING<<<` 自动请求。如果你的提交页面另有允许/启用长时运行的选项，勾选；如果没有该选项，不需要寻找额外开关。每个Bot启动时会核对其专用权重SHA。

源码公开/共享等选项与运行无关，默认可不勾选。真实登录后的页面没有被访问，不能虚构其他选项名称；如果出现未列出的必填配置，请提供其文字或截图再对应。

## 可同时提交的模型

| 建议Bot名 | 源码上传文件 | 用户存储权重 | 用途 |
|---|---|---|---|
| RL-Lab-C-v2 | [botzone_C_v2.zip](C_v2/botzone_C_v2.zip) | [mahjong_C_v2.pt](C_v2/mahjong_C_v2.pt) | 历史C，基础特征ResNet |
| RL-Lab-D-v2 | [botzone_D_v2.zip](D_v2/botzone_D_v2.zip) | [mahjong_D_v2.pt](D_v2/mahjong_D_v2.pt) | 历史D，增强特征ResNet |
| RL-Lab-D-e06 | [botzone_D_e06.zip](D_e06/botzone_D_e06.zip) | [mahjong_D_e06.pt](D_e06/mahjong_D_e06.pt) | 第四轮重训D，更快的兼容候选 |
| RL-Lab-V3-BC | [botzone_V3_BC.zip](V3_BC/botzone_V3_BC.zip) | [mahjong_V3_BC.pt](V3_BC/mahjong_V3_BC.pt) | 第四轮主推荐，纯BC |
| RL-Lab-V3-BC-seed3 | [botzone_V3_BC_seed3.zip](V3_BC_seed3/botzone_V3_BC_seed3.zip) | [mahjong_V3_BC_seed3.pt](V3_BC_seed3/mahjong_V3_BC_seed3.pt) | 历史最终池点估计较高的探索性BC备选 |
| RL-Lab-V3-PPO | [botzone_V3_PPO.zip](V3_PPO/botzone_V3_PPO.zip) | [mahjong_V3_PPO.pt](V3_PPO/mahjong_V3_PPO.pt) | 短程PPO开发最佳，768局更新快照 |
| RL-Lab-V3-PPO-long | [botzone_V3_PPO_long.zip](V3_PPO_long/botzone_V3_PPO_long.zip) | [mahjong_V3_PPO_long.pt](V3_PPO_long/mahjong_V3_PPO_long.pt) | 3072局PPO最后权重，真实RL策略 |

各目录另有同名 `.py` 单文件版本，和ZIP执行相同代码；二者任选其一，不要同时上传两份入口。现在可以多文件打包，优先用ZIP。

**七份权重文件名互不相同，可以同时放在一个账户存储中。** 不要再把它们都改名成 `mahjong_model.pt`。旧Bot仍可保留旧文件；新Bot不读取旧共享文件名。

若只先提交三份，建议 `V3_BC`、`V3_BC_seed3`、`V3_PPO_long`；C/D供核对旧版本问题，`D_e06`是较快备选。推荐来源均为第四轮既有结果，本次没有进行新的实力排名。

## 本次实际修订

- 增加JSON交互，兼容正常历史输入、长时运行的当前request与冷重启重放。
- 修复简单交互首次输入末尾的data/globaldata未明确消费的问题。
- 每个模型使用独立权重文件名；启动时严格核验SHA、架构和全部参数。
- 官方参考仓库的Python扩展只注册算番函数，没有V3依赖的五种向听API。现在代码包自带MIT许可的纯Python兼容实现：原生API齐全时用原生，否则使用内置实现。没有改动V3特征公式或删减通道。
- 兼容实现按PyMahjongGB 1.4语义处理五种向听和有效进张，覆盖密集同花、七对、十三幺、组合龙等测试。1004个固定/合成手牌、2020次函数比对通过；额外比较候选特征张量与模拟旧算番库下的实际模型输出。
- ZIP和单文件均限制线程数为1，不在Botzone现场安装依赖或下载文件。

## 已检查与未检查

已检查：七份参数逐tensor身份；代码/权重体积；Python3.6语法；每个包的JSON、SimpleIO、冷重启、单文件/ZIP输出一致性；模拟只有官方算番接口的环境。每个版本的三个协议请求包含一个真实非强制模型决策。没有运行完整对局或训练。

限制仍为：模型≤268435456字节；ZIP、解压代码资源总量与单文件分别<4000000字节。真实字节数、SHA及每包检查结果见 [DELIVERY_MANIFEST.json](DELIVERY_MANIFEST.json) 和各版本 `submission_check.json`、`runtime_check.json`。

**未登录Botzone，不知道你账户上此前实际上传了哪些文件、选择了哪些选项，不能确定过去C/D报错的唯一原因。** 没有宣称线上通过；没有实际运行平台的Python3.6/PyTorch1.8环境。纯Python向听兼容实现会增加V3计算成本，线上时间/内存是否满足需要你上传后观察。

平台至少需要 `numpy`、`torch` 和官方 `MahjongGB.MahjongFanCalculator`。Wiki列出的Python3.6对应PyTorch1.8.0；不要选择使用旧PyTorch0.4的其他Python环境。无需再要求五种原生向听API。Docker在本机没有运行，因此没有伪称完成Linux平台镜像验证。

## 报错时看什么

| 错误 | 核对事项 |
|---|---|
| `Upload the matching weight file ...` | 用户存储是否存在报错中指定的完整文件名；是否误放在额外data子目录 |
| `SHA-256 mismatch` | 代码和权重是否来自同一行；是否误上传训练checkpoint或另一模型 |
| JSON/int解析错误 | 是否还在使用旧代码；本次建议SimpleIO不勾选 |
| 没有GroupNorm/GELU、加载torch失败 | 是否选了Python3.6及其PyTorch运行环境 |
| 花牌/BUHUA不支持 | 是否进入了144张含花规则；本项目只支持136张复式 |
| 超时/内存超限 | 保留出错回合、耗时、内存与stderr；可先用D_e06核对平台环境，不要修改V3特征凑运行 |

首次响应的debug含模型ID、权重路径、SHA和向听backend，可辅助确认实际加载版本。长时运行后debug是否保留遵循平台规则。排查时请提供失败对局链接、出错回合的“玩家输入”、错误信息以及提交选项截图；不需要提供密码或令牌。

## 依据与复现

已读取：

- [Bot官方文档](https://wiki.botzone.org.cn/index.php?title=Bot)：多文件ZIP、用户存储、Python3.6/PyTorch1.8、JSON/SimpleIO、长时运行。
- [国标麻将官方文档](https://wiki.botzone.org.cn/index.php?title=Chinese-Standard-Mahjong)：牌与动作协议、136张复式规则。
- [官方参考仓库](https://github.com/ailab-pku/Chinese-Standard-Mahjong)，实际读取commit `5e818212d833a74f0ed1f4e6fb24c5b533a48ef9` 的README、C++样例、裁判和Python算番接口。仓库clone网络失败后通过GitHub API读取成功；无需手动粘贴已读页面。该仓库提供样例和算番器，不含本项目神经网络权重。
- 向听兼容语义参考PyMahjongGB1.4.0源码包中的 `mahjong-algorithm/shanten.cpp`。源码包SHA-256：`90d07327910e34890c0790f42d3b3b2926c2cb6b62d8e4122f491ccd8edc242e`；上游MIT声明保留在兼容模块中。

在仓库根目录使用现有环境，导出到不存在的新目录：

```sh
.venv/bin/python -m mahjong.prepare_botzone_delivery --output artifacts/new_botzone_delivery
.venv/bin/python -m pytest mahjong/tests/test_botzone_delivery.py mahjong/tests/test_submission_v3.py -q
```

这两条命令只导出和检查已有checkpoint，不启动训练或对战评估。
