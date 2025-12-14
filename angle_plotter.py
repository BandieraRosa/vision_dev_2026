import pandas as pd
import matplotlib.pyplot as plt
import os

# 定义文件名映射
# 组 1: Proposed Method
FILE_PROPOSED_TGT = "EulerLog/euler_angles.csv"
FILE_PROPOSED_GMB = "EulerLog/gimbal_angle.csv"

# 组 2: Traditional Method
FILE_TRAD_TGT = "EulerLog/traditional_euler_angles.csv"
FILE_TRAD_GMB = "EulerLog/traditional_gimbal_angle.csv"


def load_data(filename):
    """加载CSV并排序，如果文件不存在返回None"""
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return None
    df = pd.read_csv(filename)
    # 确保时间戳是数值型并排序
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df = df.sort_values('timestamp')
    return df


def process_group(df_tgt, df_gmb):
    """
    处理一组数据（Target + Gimbal）：
    1. 计算该组的相对时间（以该组最早的时间戳为0点）
    2. 对齐时间戳并计算差值
    """
    # 1. 确定该组的起始时间（取两个文件中最早的那个时刻）
    start_time = min(df_tgt['timestamp'].iloc[0], df_gmb['timestamp'].iloc[0])

    # 辅助函数：计算相对时间（秒）
    def to_rel_time(series):
        return (series - start_time) / 1e6  # 假设时间戳单位是微秒

    # 为原始数据添加相对时间列
    df_tgt = df_tgt.copy()
    df_gmb = df_gmb.copy()
    df_tgt['rel_time'] = to_rel_time(df_tgt['timestamp'])
    df_gmb['rel_time'] = to_rel_time(df_gmb['timestamp'])

    # 2. 使用 merge_asof 基于原始时间戳进行对齐（寻找最近邻）
    # direction='nearest' 表示找时间上最近的点
    merged = pd.merge_asof(
        df_tgt,
        df_gmb,
        on='timestamp',
        suffixes=('_tgt', '_gmb'),
        direction='nearest',
        tolerance=50000  # 可选：设置容差（比如50ms），防止匹配到太远的数据
    )

    # 计算差值
    merged['pitch_diff'] = merged['pitch_tgt'] - merged['pitch_gmb']
    merged['yaw_diff'] = merged['yaw_tgt'] - merged['yaw_gmb']

    # 计算合并后的相对时间（用于绘图）
    merged['rel_time'] = to_rel_time(merged['timestamp'])

    return df_tgt, df_gmb, merged


def main():
    # --- 1. 加载数据 ---
    df_prop_tgt = load_data(FILE_PROPOSED_TGT)
    df_prop_gmb = load_data(FILE_PROPOSED_GMB)
    df_trad_tgt = load_data(FILE_TRAD_TGT)
    df_trad_gmb = load_data(FILE_TRAD_GMB)

    if any(d is None for d in [df_prop_tgt, df_prop_gmb, df_trad_tgt, df_trad_gmb]):
        return

    # --- 2. 分别处理两组数据 ---
    # Proposed 组的处理：拥有自己的时间零点
    p_tgt, p_gmb, p_diff = process_group(df_prop_tgt, df_prop_gmb)

    # Traditional 组的处理：拥有自己的时间零点
    t_tgt, t_gmb, t_diff = process_group(df_trad_tgt, df_trad_gmb)

    # --- 3. 绘图 ---
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    # 调整子图间距
    plt.subplots_adjust(hspace=0.35, wspace=0.2)

    # 通用绘图辅助函数
    def plot_single(ax, x, y_pitch, y_yaw, title, ylabel="Angle (rad)"):
        ax.plot(x, y_pitch, label='Pitch', color='tab:red', linewidth=1.5)
        ax.plot(x, y_yaw, label='Yaw', color='tab:blue',
                linewidth=1.5, linestyle='--')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize='small')
        ax.set_ylabel(ylabel)

    # === 左列：Proposed Method ===
    # Row 1: Target
    plot_single(axes[0, 0], p_tgt['rel_time'], p_tgt['pitch'], p_tgt['yaw'],
                "Proposed: Target Euler Angles")

    # Row 2: Gimbal
    plot_single(axes[1, 0], p_gmb['rel_time'], p_gmb['pitch'], p_gmb['yaw'],
                "Proposed: Gimbal Angle")

    # Row 3: Difference
    plot_single(axes[2, 0], p_diff['rel_time'], p_diff['pitch_diff'], p_diff['yaw_diff'],
                "Proposed: Error (Target - Gimbal)", ylabel="Error (rad)")
    axes[2, 0].set_xlabel("Time (s) - Relative to Proposed Start")

    # === 右列：Traditional Method ===
    # Row 1: Target
    plot_single(axes[0, 1], t_tgt['rel_time'], t_tgt['pitch'], t_tgt['yaw'],
                "Traditional: Target Euler Angles")

    # Row 2: Gimbal
    plot_single(axes[1, 1], t_gmb['rel_time'], t_gmb['pitch'], t_gmb['yaw'],
                "Traditional: Gimbal Angle")

    # Row 3: Difference
    plot_single(axes[2, 1], t_diff['rel_time'], t_diff['pitch_diff'], t_diff['yaw_diff'],
                "Traditional: Error (Target - Gimbal)", ylabel="Error (rad)")
    axes[2, 1].set_xlabel("Time (s) - Relative to Traditional Start")

    print("绘图完成。两组数据已分别进行时间归一化。")
    plt.show()


if __name__ == "__main__":
    main()
