#!/bin/bash
# Auto-download papers for World Model Survey
# arXiv PDF pattern: https://arxiv.org/pdf/{ID}

BASE="/home/pan/workspace/umi-project/UMI-project/paper/worldmodel"

download() {
    local arxiv_id="$1"
    local filename="$2"
    local dir="$3"
    local url="https://arxiv.org/pdf/${arxiv_id}"
    local outfile="${dir}/${filename}.pdf"
    if [ -f "$outfile" ]; then
        echo "[SKIP] $filename already exists"
        return 0
    fi
    echo "[DL] $filename ($arxiv_id)"
    wget -q --timeout=60 --tries=3 -O "$outfile" "$url"
    if [ $? -ne 0 ] || [ ! -s "$outfile" ]; then
        echo "[FAIL] $filename"
        rm -f "$outfile"
    else
        echo "[OK] $filename"
    fi
    sleep 1
}

# ========== H. 经典基础 ==========
DIR="${BASE}/H_经典基础"
download "1803.10122" "[1] World Models (Ha & Schmidhuber 2018)" "$DIR"
download "1811.04551" "[2] PlaNet (Hafner 2019)" "$DIR"
download "1912.01603" "[3] DreamerV1 (Hafner 2020)" "$DIR"
download "2010.02193" "[4] DreamerV2 (Hafner 2021)" "$DIR"
download "2301.04104" "[5] DreamerV3 (Hafner 2023)" "$DIR"
download "2209.00588" "[6] IRIS (Micheli 2023)" "$DIR"
download "2405.12399" "[7] DIAMOND (2024)" "$DIR"
download "2402.15391" "[8] Genie (Google DeepMind 2024)" "$DIR"
download "2404.08471" "[9] V-JEPA (Meta 2024)" "$DIR"

# ========== A. 视频生成型 ==========
DIR="${BASE}/A_视频生成型"
download "2309.17080" "[1] GAIA-1 (Wayve 2023)" "$DIR"
download "2309.09777" "[2] DriveDreamer (GigaAI ECCV2024)" "$DIR"
download "2402.11467" "[3] DriveDreamer-2 (GigaAI AAAI2025)" "$DIR"
download "2411.01451" "[4] DriveDreamer4D (CVPR2025)" "$DIR"
download "2312.11020" "[5] Drive-WM (Tsinghua CVPR2024)" "$DIR"
download "2403.09630" "[6] GenAD (NJU CVPR2024)" "$DIR"
download "2310.02098" "[7] MagicDrive (CUHK ICLR2024)" "$DIR"
download "2405.14475" "[8] MagicDrive3D (CUHK 2024)" "$DIR"
download "2311.16813" "[9] Panacea (LiAuto CVPR2024)" "$DIR"
download "2405.17398" "[10] Vista (MIT CVPR2025)" "$DIR"
download "2312.02338" "[11] WoVoGen (2024)" "$DIR"
download "2409.05463" "[12] DriveScape (2024)" "$DIR"
download "2401.09985" "[13] WorldDreamer (Tsinghua 2024)" "$DIR"

# ========== B. 3D占用/神经场景型 ==========
DIR="${BASE}/B_3D占用_神经场景型"
download "2311.16038" "[1] OccWorld (Tsinghua ECCV2024)" "$DIR"
download "2311.01017" "[2] Copilot4D (Waabi ICLR2024)" "$DIR"
download "2311.12754" "[3] SelfOcc (Tsinghua CVPR2024)" "$DIR"
download "2406.08691" "[4] UnO (Waabi CVPR2024)" "$DIR"
download "2405.20337" "[5] OccSora (Tsinghua 2024)" "$DIR"
download "2412.10373" "[6] GaussianWorld (Tsinghua 2024)" "$DIR"

# ========== C. 端到端AD集成型 ==========
DIR="${BASE}/C_端到端AD集成型"
download "2210.09539" "[1] MILE (Wayve NeurIPS2022)" "$DIR"
download "2308.07234" "[2] UniWorld (NUDT 2023)" "$DIR"
download "2305.17330" "[3] TrafficBots (ETH ICRA2023)" "$DIR"
download "2303.05760" "[4] GameFormer (NTU NeurIPS2023)" "$DIR"
download "2402.16720" "[5] Think2Drive (ECCV2024)" "$DIR"
download "2405.04390" "[6] DriveWorld (Beihang CVPR2024)" "$DIR"
download "2306.04307" "[7] LAW (Shanghai AI Lab 2024)" "$DIR"
download "2410.23262" "[8] EMMA (Waymo 2024)" "$DIR"

# ========== D. 仿真导向型 ==========
DIR="${BASE}/D_仿真导向型"
download "2308.01534" "[1] UniSim (Waabi CVPR2023)" "$DIR"
download "2408.00415" "[2] DriveArena (ECNU 2024)" "$DIR"
download "2404.00815" "[3] LidarDM (UIUC CVPR2024)" "$DIR"
download "2405.15107" "[4] SMART (NeurIPS2024)" "$DIR"
download "2212.09723" "[5] CTG++ (Columbia 2024)" "$DIR"
download "2104.15060" "[6] DriveGAN (NVIDIA 2021)" "$DIR"

# ========== E. 基础/工业模型 ==========
DIR="${BASE}/E_基础工业模型"
download "2501.03575" "[1] Cosmos (NVIDIA 2025)" "$DIR"

# ========== F. NeRF/3DGS型 ==========
DIR="${BASE}/F_NeRF_3DGS型"
download "2311.02077" "[1] EmerNeRF (NVIDIA ICLR2024)" "$DIR"
download "2311.15260" "[2] NeuRAD (Zenseact CVPR2024)" "$DIR"
download "2401.01339" "[3] Street Gaussians (ECCV2024)" "$DIR"
download "2303.14661" "[4] UrbanGIRAFFE (ZJU ICCV2023)" "$DIR"
download "2303.00749" "[5] S-NeRF (Fudan ICLR2023)" "$DIR"
download "2307.15058" "[6] MARS (2024)" "$DIR"
download "2404.02742" "[7] LiDAR4D (CVPR2024)" "$DIR"

# ========== G. LLM/VLM融合型 ==========
DIR="${BASE}/G_LLM_VLM融合型"
download "2309.14819" "[1] ADriver-I (MEGVII 2024)" "$DIR"
download "2402.12289" "[2] DriveVLM (Tsinghua 2024)" "$DIR"
download "2310.01412" "[3] DriveGPT4 (HKU 2023)" "$DIR"
download "2310.03026" "[4] LanguageMPC (PKU 2023)" "$DIR"
download "2403.09939" "[5] Lingo-2 (Wayve 2024)" "$DIR"
download "2406.07392" "[6] LMDrive (CUHK CVPR2024)" "$DIR"

echo ""
echo "=== Download complete ==="
find "$BASE" -name "*.pdf" | sort | while read f; do
    size=$(du -sh "$f" | cut -f1)
    echo "  $size  ${f#$BASE/}"
done
