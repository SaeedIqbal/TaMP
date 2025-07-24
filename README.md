# 时序感知多原型医学自适应 (TaMP)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

该仓库包含**时序感知多原型医学自适应 (TaMP)** 的官方实现，这是一个用于**医学影像中的时序感知持续域迁移学习 (TaCDSL-Med)** 的新颖框架。TaMP通过整合时序动态、内存效率和实时自适应，解决了现有持续学习方法在动态临床环境中的关键局限性。

---

## 📚 **基础论文**

本工作基于以下基础研究：

> **基于多原型建模的无监督持续域迁移学习 (MPM)**  
> [@inproceedings{sun2025unsupervised,
	title={Unsupervised Continual Domain Shift Learning with Multi-Prototype Modeling},
	author={Sun, Haopeng and Zhang, Yingwei and Xu, Lumin and Jin, Sheng and Luo, Ping and Qian, Chen and Liu, Wentao and Chen, Yiqiang},
	booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
	pages={10131--10141},
	year={2025}
}]  
> 该论文提出了多原型学习 (MPL) 框架和双层图增强器 (BiGE)，用于无监督持续域迁移学习 (UCDSL)，在PACS和DomainNet等基准测试上展示了最先进的性能。

尽管MPM通过原型保存建立了一个强大的域自适应范式，但它假设域之间相互独立，并且缺乏显式的时序建模，这在医学影像的背景下是一个显著的缺点。

---

## 🔍 **基础论文的研究空白与不足**

MPM框架虽然有效，但在应用于真实世界的医学影像时存在几个关键局限性：

1.  **时序无关性**：MPM将每次域迁移视为孤立事件，忽略了临床环境中数据的**时序结构**（例如，患者的纵向扫描、医院的顺序部署）。这导致了次优的自适应和伪标签的不一致。
2.  **静态图传播**：双层图增强器 (BiGE) 使用固定步长的静态图传播，计算成本高昂，且无法根据域迁移的幅度进行自适应。
3.  **低效的内存管理**：MPM为每个域存储一个原型，导致**原型爆炸**和长期部署中的高内存开销，这对于资源受限的临床系统来说是不切实际的。
4.  **缺乏伪标签稳定性**：该框架没有显式地在预测中强制执行时序一致性，使其容易受到扫描仪伪影或染色变化等非临床因素引起的**预测振荡**的影响。

---

## ✨ **我们提出的解决方案：TaMP-Med**

为了解决这些不足，我们提出了**TaMP-Med**，这是一个综合框架，通过四个为医学影像独特挑战设计的关键组件扩展了MPM：

| 组件 | 目的 | 核心创新 |
| :--- | :--- | :--- |
| **时序原型适配器 (TPA)** | 建模域演化 | 使用LSTM和数据集特定的建模（例如，BRaTS的黎曼几何，Camelyon16的最优传输）融合当前和历史原型。 |
| **动态原型管理器 (DPM)** | 高效管理内存 | 使用KL散度和谱聚类来剪枝过时和合并相似的原型，确保次线性增长和高内存效率 (ME)。 |
| **时序一致性精炼 (TCR)** | 确保稳定的预测 | 应用卡尔曼滤波和时序卷积网络 (TCN) 随时间平滑伪标签，防止误差累积。 |
| **自适应图传播网络 (AGPN)** | 实现实时推理 | 使用切比雪夫近似和动态步长控制进行快速、可扩展的标签传播，性能优于BiGE。 |

TaMP-Med通过最小化**自适应差距**建立了一个新的误差界，并集成了高级数学原理（微分几何、最优传输、随机过程）以实现模态特定的自适应。

---

## 📁 **数据集**

TaMP-Med在以下医学影像数据集上进行了评估，每个数据集代表一个独特的域迁移挑战：

### 训练与持续自适应数据集

*   **NIH Chest X-ray**  
    一个包含112,120张前位胸片的大规模数据集，涵盖14种常见胸部疾病标签。其数据因不同医院、扫描仪（便携式 vs. 固定式）和患者群体而产生迁移。  
    **参考文献**: Wang, X. 等. "ChestX-ray8: 医院规模的胸部X光数据库和常见胸部疾病弱监督分类与定位的基准." *CVPR*, 2017.  
    [https://arxiv.org/abs/1705.02315](https://arxiv.org/abs/1705.02315)

*   **BRaTS (脑肿瘤分割)**  
    包含多机构、多模态MRI扫描（T1, T1c, T2, FLAIR）用于胶质瘤。主要挑战是扫描仪间的异质性和纵向变化。  
    **参考文献**: Menze, B.H. 等. "多模态脑肿瘤图像分割基准 (BRATS)." *IEEE TMI*, 2014.  
    [https://ieeexplore.ieee.org/document/6975210](https://ieeexplore.ieee.org/document/6975210)

*   **Camelyon16**  
    提供淋巴结切片的全切片图像 (WSI)，用于乳腺癌转移检测。它是组织病理学中染色变化和批次效应的基准。  
    **参考文献**: Bejnordi, B.E. 等. "数字病理学中淋巴结转移检测的深度学习算法的诊断评估." *JAMA*, 2017.  
    [https://jamanetwork.com/journals/jama/fullarticle/2643711](https://jamanetwork.com/journals/jama/fullarticle/2643711)

*   **PANDA (前列腺癌分级评估)**  
    一个用于从活检组织进行前列腺癌Gleason分级的大规模数据集。其特点是组织制备和病理学家评分的显著实验室间变异性。  
    **参考文献**: Bulten, W. 等. "活检中前列腺癌诊断和分级的人工智能：一项基于人群的评估." *The Lancet Digital Health*, 2022.  
    [https://www.thelancet.com/journals/landig/article/PIIS2589-7500(22)00013-3/fulltext](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(22)00013-3/fulltext)

### 未见域验证数据集

*   **ISIC-2019**  
    一个用于皮肤病变分析的数据集，用于评估从放射学/组织病理学到皮肤病学的零样本泛化能力。  
    **参考文献**: Tschandl, P. 等. "HAM10000数据集：一个大型多源常见色素性皮肤病变的皮肤镜图像集合." *Scientific Data*, 2018.  
    [https://www.nature.com/articles/sdata2018161](https://www.nature.com/articles/sdata2018161)

*   **BreakHis**  
    包含不同放大倍数（40x, 100x, 200x, 400x）的乳腺组织病理学图像，用于测试跨空间尺度的泛化能力。  
    **参考文献**: Spanhol, F.A. 等. "乳腺癌组织病理学图像分类的数据集." *IEEE TMI*, 2016.  
    [https://ieeexplore.ieee.org/document/7460968](https://ieeexplore.ieee.org/document/7460968)

---

## 📂 **仓库结构**

```
TaMP/
├── tpa.py              # 时序原型适配器 (TPA) 的实现
├── dpm.py              # 动态原型管理器 (DPM) 的实现
├── tcr.py              # 时序一致性精炼 (TCR) 的实现
├── agpn.py             # 自适应图传播网络 (AGPN) 的实现
├── dataLoader.py       # 所有数据集的统一数据加载和预处理
├── train.py            # TaMP-Med框架的主训练脚本
├── evaluate.py         # 模型评估和指标计算脚本
├── config/
│   └── defaults.yaml   # 默认超参数和配置
├── results/
│   └── ...             # 用于保存结果、日志和可视化文件的目录
└── README.md           # 本文件
```

---

## 🚀 **快速开始**

### 安装

```bash
git clone https://github.com/your-username/TaMP.git
cd TaMP
pip install -r requirements.txt
```

### 数据准备

1.  下载数据集并根据 `dataLoader.py` 中的结构将其组织在 `/home/phd/datasets/` 目录下。
2.  确保数据集文件夹的命名正确（例如，`NIH_Chest_Xray`, `BRaTS`）。

### 训练

在特定数据集上训练TaMP-Med：

```bash
python train.py --dataset NIH --backbone resnet50 --alpha 0.6 --beta 0.8 --lambda 0.7 --epochs 100
```

### 评估

评估模型性能并生成分析图表：

```bash
python evaluate.py --model_path ./checkpoints/TaMP_NIH.pth --dataset NIH --output_dir ./results/NIH_eval
```

---

## 📈 **结果**

我们大量的实验证明，**TaMP-Med在所有数据集上始终优于MPM和其他SOTA方法**。关键结果包括：

*   **更高的DAA和DGA**：卓越的自适应和泛化能力。
*   **更低的遗忘 (FM)**：在过往域上的性能得以保留。
*   **更高的时序一致性 (TCS)**：稳定、可信的预测。
*   **更好的内存效率 (ME)**：原型占用空间减少高达40%。

有关详细结果和可视化，请参阅我们的完整论文。

---

## 📄 **引用**

如果您在研究中使用了此代码或框架，请引用我们的论文：

```bibtex
@inproceedings{your_paper_2024,
  title={Temporal-Aware Multi-Prototype Medical Adaptation for Continual Domain Shift Learning},
  author={Your Name and Co-authors},
  booktitle={Proceedings of the IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year={2024}
}
```

---

## 🤝 **许可证**

该项目采用MIT许可证发布 - 详情请见 [LICENSE](LICENSE) 文件。
