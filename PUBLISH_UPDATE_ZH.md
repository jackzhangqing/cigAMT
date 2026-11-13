# 本次数据、代码与网站更新

## 1. 已经完成的 Hugging Face 发布

数据地址：<https://huggingface.co/datasets/ZhangQing123/cigAMT>

- 两条数据分支、独立噪声库、任务配置及加载重放代码，约 29.4 GB。
- Dataset Viewer：32 张实际波形预览，以及天电、参考信号、噪声三个完整索引。
- [DATA_INVENTORY.md](DATA_INVENTORY.md) 包含数据来源、划分、数量、噪声类别和读取方式。
- 数据许可证：CC BY-SA 4.0。原始数据与实验代码没有改动。

## 2. 提交本地代码和网站更新

在当前 `website/cigAMT` 仓库目录的 PowerShell 中执行：

```powershell
git status --short
git add .gitignore README.md DATA_INVENTORY.md RELEASE_CHECKS.md PUBLISH_UPDATE_ZH.md code docs tools/download_data.py tools/download_weights.py tools/publish_weights.ps1
git diff --cached --stat
git commit -m "Release cigAMT benchmark code and update data access"
git push origin main
```

无需再次 `git init` 或 `git remote add`。网站配置、正文和使用指南已加入 Hugging Face 与 GitHub 资源入口。推送后现有 Pages 工作流会部署 `docs/`。

已有的 `DEPLOYMENT_ZH.txt` 修改未被本次操作覆盖，也未放入上述暂存命令。如需一并提交，请先查看该文件的差异，再单独 `git add DEPLOYMENT_ZH.txt`。

## 3. 模型权重要另行上传附件

8 个已核对的 ZIP 位于 **`D:\cigAMT_release\github_release_assets`**，合计约 1.03 GB（十进制）。共 128 个推理资产：48 个天电资产和 80 个 AMT 检查点。原始权重未改动，发布副本去除了优化器恢复状态和机器路径。

**普通 `git push` 只上传代码、架构、配置和说明，不会上传这些 ZIP。**

两种方式任选其一：

1. GitHub 仓库进入 **Releases → Draft a new release**，新建标签 `weights-v1`，上传上述目录里的 8 个 ZIP。标题和正文可参考 [WEIGHTS_RELEASE_NOTES.md](code/WEIGHTS_RELEASE_NOTES.md)，然后点击 Publish release。
2. 若已安装 GitHub CLI，在 PowerShell 中执行：

```powershell
gh auth login
gh api user --jq .login
powershell -ExecutionPolicy Bypass -File .\tools\publish_weights.ps1 -AssetDirectory "D:\cigAMT_release\github_release_assets"
```

登录账号应为 `jackzhangqing`。脚本先校验每个 ZIP 的大小与 SHA-256，随后创建 Release 并附加文件；不会修改原始实验目录。已经创建同名 Release 时，先检查网页上的附件状态，避免重复创建或盲目覆盖。

附件名称需要保持不变，下载脚本通过这些名称和 `weights-v1` 标签定位文件。校验清单在 [pretrained_assets.json](code/pretrained_assets.json)，使用说明在 [WEIGHTS.md](code/WEIGHTS.md)。不要把 ZIP、`.pt` 或原始数据强制加入普通 Git 历史。

## 4. 推送后的检查

- 网站 Resources 中的 Hugging Face 和 GitHub 链接可以打开。
- GitHub `code/sferic` 与 `code/amt` 均可见，包含训练、评估、架构及阻抗入口。
- Releases 显示 8 个 ZIP 附件后，用户才能用权重下载脚本下载。
- 数据卡上的 Dataset Viewer 可切换四个配置；完整索引与代表性图片预览分别展示。

本次没有替你执行 Git commit、push 或发布 GitHub Release。
