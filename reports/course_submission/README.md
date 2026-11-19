# 课程最终报告

正式正文：[Markdown](FINAL_COURSE_REPORT.md) · [5 页 A4 PDF](FINAL_COURSE_REPORT.pdf)。提交前填写 Markdown 中的 `【姓名】` 和 `【学号】`，然后重新生成 PDF；未代填任何个人信息。

最终 Botzone 策略为用户确认已提交并成功运行的 **V3 短程 PPO**，对应 `artifacts/round4/PPO_legacy_best/checkpoint.pt`，不是 BC 或 `PPO_long`。整次 PPO 实验共 1024 局；选中的 iteration 48 权重累计经历 768 局、422 次优化器更新，包含前 128 局 Critic 预热。

`SOURCE_CHECK.json` 保存核对过的 checkpoint/导出 SHA、逐参数一致性结论、原始日志计数、结果表数据和来源文件 SHA。`BUILD_CHECK.json` 保存 PDF 页数、正文汉字数和文件身份；最终人工视觉核验记录见 `REVIEW.md`。这些附属文件不必与正文一并上传教学网。

## 重建报告

在仓库根目录使用具备 PyTorch / Matplotlib 的项目环境生成来源核验和图像：

```sh
.venv/bin/python reports/course_submission/build_report.py --check-sources --assets
```

使用安装了 ReportLab、Pillow、pypdf 的 Python 排版同一份 Markdown：

```sh
python reports/course_submission/build_report.py --pdf
```

本次使用项目 Python 3.10.22 / Matplotlib 3.10.9 生成图表，Codex 配套 Python 中的 ReportLab 4.4.9 / pypdf 6.21.4 生成 PDF。脚本默认嵌入 macOS 宋体及黑体；其他系统可用 `--font`、`--bold-font` 指定有中文覆盖的 TrueType 字体，并设置相应 `--font-index`、`--bold-font-index`（普通 TTF 使用 0）。图像和公式从源码结构、已有训练日志与 Markdown 生成，未新增训练或对战。

最后使用 `pdftoppm` 渲染并逐页检查中文、公式、图表与分页。修改正文后必须重新运行 `--assets` 和 `--pdf`，再次核对；旧的 `REVIEW.md` 不能作为修改后 PDF 的视觉验证。
