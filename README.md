# ct2us — CT 驱动的仿真超声数据生成（CT→US 伪配对数据集）

面向**肝脏手术导航系统的 CT–US 配准**任务。输入仅为术前 CT 及其多器官分割
（模式来自 Medical Segmentation Decathlon：Task03 肝脏+肿瘤、Task08 肝血管+肿瘤），
输出为 **{CT, US_synthetic, T_CT→US}** 伪配对训练数据集。

---

## 1. 与原始方案相比的改进点

原始方案提出两条路线（快速纹理渲染 + k-Wave 物理仿真），本实现遵循该总体框架，
并在工程与物理表达上做了如下改进：

| 方面 | 原始方案 | 本实现（改进） |
| --- | --- | --- |
| 渲染几何 | 未明确 | 采用**探头扫描平面射线重采样（raycasting）**：探头位姿 → 平面像素物理坐标 → 仿射逆变换 → 体素三重线性插值。几何与外观完全解耦。 |
| 探头模型 | 仅提及位置/姿态/深度/FOV | 支持**线阵 + 凸阵**两种真实几何；给出从肝脏表面网格随机采样位姿的完整流程，并保证平面与肝脏有足够重叠。 |
| 组织分类 | 多器官分割 | 融合 Task03/Task08 标签，并用 CT HU 阈值补充骨骼/肋骨/软组织等分割未覆盖的组织，形成统一组织图（tissue map）。 |
| 散射建模 | 空间相关散斑 | 散射场 = 组织基频 + 组织内连续三维散射起伏；散斑采用**各向异性相关的高斯–瑞利包络**在扫描空间内生成，轴向/横向相关长度对应探头 PSF。 |
| 伪影 | 声影/后方增强/边缘增强/噪声 | 声影（强反射体与高衰减沿束线累积）、后方增强（低衰减腔后）、边缘声影（阴影图横向梯度）、混响（多次回波）、深度相关噪声。所有伪影为**扫描空间的 2D 后处理算子**，可独立开关/调参。 |
| 变形 | 提及呼吸/压迫 | 以**探头局部坐标位移场**实现：呼吸运动沿束轴调制 + 探头近场压迫压缩，作为采样坐标的形变（warping）。 |
| 物理校准 | 比较 6 项指标后反向调整 | 实现 `calibrate` 模块：自动计算 6 项指标（Speckle 统计、深度衰减、组织对比、边界回声、血管外观、声影），并通过轻量网格搜索自动调整渲染参数。 |
| k-Wave | 独立仿真 | 提供 `physic_sim`：纯 Python 的**解析物理代理**（PSF ✳ 反射场 + 逐束线衰减）保证可离线运行；并提供可选的 MATLAB k-Wave 适配器完成全物理仿真，二者输出统一 RF/B-mode 接口。 |
| 配对数据 | 保存位姿与变换 | 除位姿外保存 **T_CT→US（4×4）及其逆矩阵、被重采样的 CT 切片（resliced CT）**——后者作为配准的“完美对齐参考”，可直接用于监督训练。 |

## 2. 总体框架

```
                     CT影像 (+ 多器官分割: Task03 肝脏/肿瘤 + Task08 血管/肿瘤)
                                  │
                                  ▼
                    io_utils: 加载 / 仿射对齐 / 砍包围盒
                                  │
                                  ▼
                    anatomy: 标签融合 → 组织分类图 + 声学参数表 (c,ρ,α,S)
                                  │
              ┌───────────────────┴───────────────────┐
              │                                       │
           pose: 探头位姿采样                       physic_sim: 声学参数场
        (肝脏表面 → 面/束轴/横轴)                    c(x), ρ(x), α(x), S(x)
              │                                       │  (解析物理代理 / k-Wave)
              ▼                                       ▼
        geometry: 扫描平面 raycasting              RF 信号 → 包络 → B-mode
        (线阵/凸阵, 深度/FOV/分辨率)                     │
              │                                       │
              ▼                                       ▼
         speckle + render + artifacts     ┌───────────┘
              │                           calibrate: 6 项指标对比与参数校准
              ▼                           ▼
        B-mode 超声影像                    校准后渲染参数反馈
              │
              ▼
      dataset: 导出 {CT, US, T_CT→US, 重采样CT切片, 参数}
              │
              ▼
      CT–US 伪配对训练数据集 → 配准网络训练 (train/baseline_registration.py)
```

## 3. 目录结构

```
ct2us/
  io_utils.py        NIfTI 读写、仿射工具、平面重采样
  anatomy.py         标签融合、组织分类、声学参数表
  geometry.py        探头几何（线阵/凸阵）与 raycasting 采样
  pose.py            探头位姿采样（肝脏表面网格 + 姿态随机化）
  speckle.py         相关散斑场生成（高斯–瑞利包络）
  render.py          B-mode 渲染主流程 + TGC + 对数压缩
  artifacts.py       声影 / 后方增强 / 边缘声影 / 混响 / 噪声
  deformation.py     呼吸运动 + 探头压迫位移场
  physic_sim.py      解析物理代理 RF 仿真 + 可选 k-Wave 适配器
  calibrate.py       6 项真实性指标 + 渲染参数自动校准
  dataset.py         数据集编排与导出
scripts/
  render_demo.py     单个体积快速演示，输出预览图
  generate_dataset.py 大规模生成 {CT,US,T_CT→US} 配对数据集
  generate_for_new.py 生成3D CT-US体积，适配E:\new配准项目格式
  calibrate_quick.py  渲染 vs 物理代理指标对比与参数校准
  check_consistency.py 生成几何一致性自检（配准前回归测试）
  evaluate_registration.py 配准验收：TRE 统计、捕获范围、置信度、延迟
train/
  baseline_registration.py  基于合成配对的 CT–US 配准（粗回归 + 可微精配准）
```

## 4. 快速开始

> 数据路径注意：Task03/Task08 下载包内还有一层同名子目录
> （`dataset/Task03_Liver/Task03_Liver`），命令需用**内层**路径。

```bash
pip install -r requirements.txt

# GPU PyTorch（如未安装；conda 示例）
# conda create -n build python=3.12
# pip install torch --index-url https://download.pytorch.org/whl/cu126

# 演示：对一个 CT 生成若干仿真超声切片并输出预览
python scripts/render_demo.py --data dataset/Task08_HepaticVessel/Task08_HepaticVessel --out outputs/demo

# 生成配对数据集（可指定任务、片数、随机种子）
python scripts/generate_dataset.py \
  --liver dataset/Task03_Liver/Task03_Liver \
  --vessel dataset/Task08_HepaticVessel/Task08_HepaticVessel \
  --out outputs/pairs --volumes 10 --per_volume 12 --seed 0 --workers 4

# 校准实验
python scripts/calibrate_quick.py --data dataset/Task08_HepaticVessel/Task08_HepaticVessel
```

### 生成3D体积（适配E:\new配准项目）

```bash
# 生成3D CT-US体积，格式兼容E:\new的RigidRegistrationDataset
python scripts/generate_for_new.py \
  --liver dataset/Task03_Liver/Task03_Liver \
  --out E:/new/data \
  --n_cases 10 --n_elev 32 --elev_spacing 1.0 --seed 0

# 输出格式:
#   E:/new/data/ct/*.nii.gz  # 3D CT体积 (D,H,W), HU值
#   E:/new/data/us/*.nii.gz  # 3D US体积 (D,H,W), float32 [0,1]
#
# E:\new的归一化:
#   CT: (vol + 200) / 500.0
#   US: (vol - vol.min()) / (vol.max() - vol.min())
```

### 数据集两档

- **`no_deform`（刚性集，配准严格验证用）**：生成时关闭呼吸/压迫变形，此时
  US 图像是 CT 某平面像素级对应的刚体切片，`us_to_ct` 可直接复现。
- **默认（变形集，真实场景鲁棒性用）**：叠加呼吸位移 + 探头压迫位移，模拟术中
  非刚体差异；此时保存的 `ct_slice` 与刚体重采样会有可控偏差（这是设计意图）。

```bash
# 刚性集
python scripts/generate_dataset.py --liver ... --vessel ... \
  --out outputs/pairs_rigid --volumes 10 --per_volume 12 --no_deform
# 变形集（不带 --no_deform）
```

### 一致性自检（配准前必跑）

```bash
python scripts/check_consistency.py \
  --index outputs/pairs_rigid/liver_*/index_index.jsonl \
         outputs/pairs_rigid/vessel_*/index_index.jsonl --strict
```

对每个样本重采样 CT/分割并验证：刚性集 CT 像素级一致（RMS≈0）、组织标签一致、
`transform.npz` 与位姿推导的 `us_to_ct` 完全相等且 det=+1。这从构造上保证
**生成不会破坏位置/结构**——配准真值无系统性偏差。

### 配准验收（训练后 / 引擎验证）

```bash
# 引擎几何验证：用 oracle 分割（真值 seg_slice）从 GT+扰动精配准，测捕获范围
python scripts/evaluate_registration.py \
  --index outputs/pairs_rigid/vessel_*/index_index.jsonl \
  --oracle_seg --no_coarse --max_samples 30

# 端到端（训练完 checkpoint 后）
python scripts/evaluate_registration.py --index ... --checkpoint outputs/model.pt
```

输出 TRE 均值/中位数、成功率 %<5mm/%<10mm、捕获范围矩阵、每帧延迟与预测置信度。

### 导出的数据形态（每个样本一个子目录 `sample_<case>_<idx>`）

```
sample_000_04/
  us.png              # B-mode 超声切片（uint8，可再带原始灰度）
  params.json         # 几何 + 渲染 + 伪影全部参数
  ct_slice.nii.gz     # 沿同一扫描网格重采样的 CT 切片（配准参考/moving image）
  seg_slice.nii.gz    # 组织标签切片
  transform.npz       # 数组：us_to_ct(4x4)、ct_to_us(4x4)、probe 位姿、深度轴等
```

`transform.npz` 中的 `us_to_ct` 满足（US 物理坐标：`x=横向(mm), y=深度(mm), z=0`）：

```
p_ct = us_to_ct @ [x_us, y_us, 0, 1]^T
```

它是该样本配准任务的 **ground-truth 刚体变换**。

## 5. 物理真实性校准

`calibrate.py` 对渲染结果与“参考图像”（默认使用 `physic_sim` 的解析物理代理，
启用 k-Wave 后可用真值仿真）计算以下 6 项指标：

1. **Speckle Statistics** — 肝脏实质区域包络的 Rayleigh/对数正态拟合、SNR、自相关长度；
2. **Depth Attenuation** — 均匀区平均强度随深度的 dB 斜率；
3. **Tissue Contrast** — 肝实质 vs 肿瘤/血管的对比度比（dB）；
4. **Boundary Echo** — 包膜/壁界面的峰值强度、厚度与信噪；
5. **Vessel Appearance** — 腔体低回声深度、壁回声明亮、腔中心强度比；
6. **Acoustic Shadow** — 强反射体后方强度衰减的量级与范围。

校准过程：对若干参数（散斑强度、衰减系数、回声强度、伪影权重）做小规模网格搜索，
最小化指标差异后写出最优参数，供 `render_demo` / `generate_dataset` 复用。

## 6. 说明与限制

- 目标为**术中肝脏超声（B 超）** 的形态学仿真，重点保证解剖结构、几何对应关系与
  纹理/伪影的统计真实感；不承诺声速/密度绝对物理精度（该部分由物理代理/k-Wave 校准）。
- Task03 与 Task08 为不同病例，标签不在同一坐标系。`generate_dataset.py` 分别指定
  `--liver` 与 `--vessel` 目录；如使用仅带肝脏标签的数据，血管外观可由 CT HU 阈值
  （低密度管腔 + 高密度壁）启发式补充。
- k-Wave 适配需要 MATLAB 及 k-Wave 工具箱（见 `physic_sim.py` 头注释），默认使用
  纯 Python 解析物理代理，因此本仓库可在无 MATLAB 环境下完整运行。
- **生成对配准的影响**：位置/结构一致性由构造保证（几何检查见上文）；唯一有意的
  破坏是默认启用呼吸/压迫变形。评价纯刚性配准性能请用 `--no_deform` 刚性集，
  需要对抗术中变形时再用变形集做鲁棒性测试。