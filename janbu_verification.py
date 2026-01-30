"""
Janbu 方法验证测试 - 使用图片中红框内的表格数据直接验证

根据图片中红框内的"Initial step"表格数据：
- 输入参数：tan α, Δx, p, u, c', tan φ', ΔQ
- 预期结果：F₀ = 1.385, F₁ = 1.485, F₂ = 1.48
"""

# -*- coding: utf-8 -*-
import numpy as np
import sys
import os

# 设置标准输出编码为UTF-8（Windows兼容）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 添加当前目录到路径，以便导入 janbu 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from janbu import JanbuSolver


def create_slices_from_table_data():
    """
    根据图片中红框内的表格数据创建slices数据结构
    
    从图片描述中提取的数据：
    Slice 1: tan α=1.13, Δx=4.4, p=5.3, u=2.12, c'=1.0, tan φ'=0.67, ΔQ=0
    Slice 2: tan α=0.50, Δx=11.0, p=10.1, u=4.04, c'=1.0, tan φ'=0.67, ΔQ=0
    Slice 3: tan α=0.18, Δx=11.0, p=8.6, u=3.44, c'=1.0, tan φ'=0.67, ΔQ=0
    Slice 4: tan α=-0.04, Δx=6.0, p=2.9, u=1.16, c'=1.0, tan φ'=0.67, ΔQ=0
    """
    # 从表格中提取的数据
    tan_alpha = [1.13, 0.50, 0.18, -0.04]  # tan α
    delta_x = [4.4, 11.0, 11.0, 6.0]      # Δx
    p = [5.3, 10.1, 8.6, 2.9]              # p (kPa)
    u = [2.12, 4.04, 3.44, 1.16]           # u (kPa)
    c_prime = [1.0, 1.0, 1.0, 1.0]         # c' (kPa)
    tan_phi_prime = [0.67, 0.67, 0.67, 0.67]  # tan φ'
    delta_Q = [0.0, 0.0, 0.0, 0.0]         # ΔQ
    
    n_slices = len(tan_alpha)
    
    # 创建slices数据结构
    slices = []
    for i in range(n_slices):
        alpha_rad = np.arctan(tan_alpha[i])
        
        # 计算W（重量），根据 p = W/Δx，所以 W = p * Δx
        # 但这里我们只需要p值，W值可以任意（因为u是直接给出的，不通过ru计算）
        W_i = p[i] * delta_x[i]  # 这个值用于满足代码要求，但实际计算中u是直接给出的
        
        slice_data = {
            'width': delta_x[i],           # Δx
            'alpha_rad': alpha_rad,        # α (弧度)
            'alpha_deg': np.degrees(alpha_rad),  # α (度)
            'p': p[i],                     # p (kPa)
            'W': W_i,                      # W (kN/m) - 用于满足代码要求
            'h_mid': 1.0,                  # 条高（占位值，实际计算中不使用）
            'x_mid': sum(delta_x[:i]) + delta_x[i]/2,  # 中点x坐标（占位值）
            'x_left': sum(delta_x[:i]),    # 左边界x坐标（占位值）
            'x_right': sum(delta_x[:i+1]), # 右边界x坐标（占位值）
        }
        slices.append(slice_data)
    
    return slices, u, c_prime, tan_phi_prime, delta_Q


def main():
    print("=" * 100)
    print("Janbu 方法验证测试 - 使用图片中红框内的表格数据直接验证")
    print("=" * 100)
    
    # ============================================================
    # 1. 从表格数据创建slices
    # ============================================================
    slices, u_table, c_prime_table, tan_phi_prime_table, delta_Q_table = create_slices_from_table_data()
    
    n_slices = len(slices)
    
    print(f"\n从表格中提取的输入数据（{n_slices}个土条）：")
    print("-" * 100)
    c_prime_label = "c' (kPa)"
    tan_phi_label = "tan φ'"
    print(f"{'Slice':<8} {'tan α':<10} {'Δx':<10} {'p (kPa)':<12} {'u (kPa)':<12} {c_prime_label:<12} {tan_phi_label:<12} {'ΔQ':<10}")
    print("-" * 100)
    for i in range(n_slices):
        print(f"{i+1:<8} {np.tan(slices[i]['alpha_rad']):<10.2f} {slices[i]['width']:<10.1f} "
              f"{slices[i]['p']:<12.1f} {u_table[i]:<12.2f} {c_prime_table[i]:<12.1f} "
              f"{tan_phi_prime_table[i]:<12.2f} {delta_Q_table[i]:<10.1f}")
    
    # ============================================================
    # 2. 创建求解器（使用表格中的参数）
    # ============================================================
    # 将tan φ'转换为角度
    phi_prime_table = [np.degrees(np.arctan(tan_phi)) for tan_phi in tan_phi_prime_table]
    
    # 创建求解器，直接使用表格中的u值（不通过ru计算）
    solver = JanbuSolver(
        c_prime=c_prime_table,
        phi_prime=phi_prime_table,
        ru=0.0,  # 不使用ru，直接使用u_i
        u_i=u_table,  # 直接使用表格中的u值
        delta_Q_i=delta_Q_table,
    )
    
    print(f"\n求解器参数：")
    print(f"  c' = {c_prime_table} kPa")
    print(f"  φ' = {[f'{p:.2f}°' for p in phi_prime_table]}")
    print(f"  u = {u_table} kPa（直接使用表格值）")
    print(f"  ΔQ = {delta_Q_table} kN/m")
    
    # ============================================================
    # 3. 计算 F0（初始安全系数，t=0）
    # ============================================================
    print(f"\n" + "=" * 100)
    print("步骤 1: 计算初始安全系数 F₀ (t=0)")
    print("=" * 100)
    
    F0, conv0, it0 = solver.calculate_fos_initial(
        slices=slices,
        F_init=1.0,
        tolerance=1e-6,
        max_iter=50,
    )
    
    print(f"\n计算结果：")
    print(f"  F₀ = {F0:.6f}")
    print(f"  预期值 F₀ = 1.385")
    print(f"  误差 = {abs(F0 - 1.385):.6f} ({abs(F0 - 1.385)/1.385*100:.2f}%)")
    print(f"  收敛状态: {'已收敛' if conv0 else '未收敛'}")
    print(f"  迭代次数: {it0}")
    
    # 验证中间计算值（根据图片中的表格）
    print(f"\n验证中间计算值（根据图片中的表格）：")
    print("-" * 100)
    
    # 手动计算一次以验证
    tan_phi = tan_phi_prime_table[0]  # 所有土条相同
    F_test = F0
    
    # 定义表头字符串（避免f-string中的反斜杠问题）
    header_A_prime = "A'₀"
    header_n_alpha = "nα₀"
    header_A0 = "A₀"
    header_delta_E = "ΔE₀"
    header_E0 = "E₀"
    header_B0 = "B₀"
    
    print(f"{'Slice':<8} {header_B0:<12} {header_A_prime:<12} {header_n_alpha:<12} {header_A0:<12} {header_delta_E:<12} {header_E0:<12}")
    print("-" * 100)
    
    sum_B0 = 0.0
    sum_A_prime0 = 0.0
    sum_A0_over_n = 0.0
    sum_delta_E0 = 0.0
    E_interface = [0.0]  # Eₐ = 0
    
    for i in range(n_slices):
        p_i = slices[i]['p']
        alpha_rad = slices[i]['alpha_rad']
        dx_i = slices[i]['width']
        c_i = c_prime_table[i]
        u_i = u_table[i]
        dQ_i = delta_Q_table[i]
        
        tan_alpha = np.tan(alpha_rad)
        cos_alpha = np.cos(alpha_rad)
        
        # B₀ = p_i * Δx_i * tan α_i + ΔQ_i
        B0_i = p_i * dx_i * tan_alpha + dQ_i
        
        # A'₀ = [c'_i + (p_i - u_i) tan φ'_i] Δx_i (1 + tan²α_i)
        A_prime0_i = (c_i + (p_i - u_i) * tan_phi) * dx_i 
        
        # nα₀ = cos²α_i (1 + tan α_i tan φ'_i / F₀)
        n_alpha0_i = (cos_alpha ** 2) * (1.0 + tan_alpha * tan_phi / F_test)
        
        # A₀ = A'₀ / nα₀
        A0_i = A_prime0_i / n_alpha0_i
        
        # ΔE₀ = B₀ - A₀ / F₀
        delta_E0_i = B0_i - A0_i / F_test
        
        # E₀ = E_{i-1} + ΔE₀
        E_interface.append(E_interface[-1] + delta_E0_i)
        
        sum_B0 += B0_i
        sum_A_prime0 += A_prime0_i
        sum_A0_over_n += A0_i
        sum_delta_E0 += delta_E0_i
        
        print(f"{i+1:<8} {B0_i:<12.1f} {A_prime0_i:<12.1f} {n_alpha0_i:<12.4f} "
              f"{A0_i:<12.1f} {delta_E0_i:<12.1f} {E_interface[i+1]:<12.1f}")
    
    print(f"{'Σ':<8} {sum_B0:<12.1f} {sum_A_prime0:<12.1f} {'':<12} "
          f"{sum_A0_over_n:<12.1f} {sum_delta_E0:<12.1f} {E_interface[-1]:<12.1f}")
    
    print(f"\nF₀ = Σ(A₀) / Σ(B₀) = {sum_A0_over_n:.3f} / {sum_B0:.3f} = {F0:.3f}")
    
    # 与图片中的预期值对比
    expected_B0 = [26.4, 55.6, 17.0, 0.7]
    expected_A_prime0 = [13.8, 55.5, 49.0, 13.0]
    expected_n_alpha0 = [0.68, 0.99, 1.05, 0.98]
    expected_A0 = [20.3, 56.0, 46.6, 13.3]
    expected_delta_E0 = [11.7, 15.1, -16.5, -10.3]
    expected_E0 = [11.7, 26.8, 10.3, 0.0]
    
    print(f"\n与图片中预期值对比：")
    print("-" * 100)
    print(f"{'Slice':<8} {'B₀ (计算)':<15} {'B₀ (预期)':<15} {'误差':<12}")
    print("-" * 100)
    for i in range(n_slices):
        p_i = slices[i]['p']
        dx_i = slices[i]['width']
        tan_alpha_i = np.tan(slices[i]['alpha_rad'])
        B0_calc = p_i * dx_i * tan_alpha_i + delta_Q_table[i]
        error = abs(B0_calc - expected_B0[i])
        print(f"{i+1:<8} {B0_calc:<15.1f} {expected_B0[i]:<15.1f} {error:<12.3f}")
    
    # ============================================================
    # 4. GPS 完整迭代（计算 F₁, F₂）
    # ============================================================
    print(f"\n" + "=" * 100)
    print("步骤 2: GPS 完整迭代（计算 F₁, F₂）")
    print("=" * 100)
    
    # 为了进行GPS迭代，需要构建一个滑动面
    # 根据表格数据，我们可以从tan α和Δx构建滑动面
    slip_profile = []
    x_current = 0.0
    y_current = 0.0
    
    for i in range(n_slices):
        # 滑动面起点
        if i == 0:
            slip_profile.append([x_current, y_current])
        
        # 根据tan α和Δx计算下一个点
        dx_i = slices[i]['width']
        tan_alpha_i = np.tan(slices[i]['alpha_rad'])
        dy_i = dx_i * tan_alpha_i
        
        x_current += dx_i
        y_current += dy_i
        slip_profile.append([x_current, y_current])
    
    slip_profile = np.array(slip_profile)
    
    print(f"\n根据表格数据构建的滑动面坐标：")
    for i, (x, y) in enumerate(slip_profile):
        print(f"  点 {i+1}: ({x:.2f}, {y:.2f})")
    
    # 进行GPS迭代
    F, converged, iterations, debug = solver.calculate_fos_gps(
        slices=slices,
        slip_profile=slip_profile.tolist(),
        F_init=F0,
        tolerance=1e-6,
        max_iter=100,
        lambda_thrust=0.33,
        print_iteration_table=True,
        return_debug=True,
    )
    
    print(f"\n最终计算结果：")
    print(f"  F₀ = {F0:.6f} (预期: 1.385)")
    if iterations >= 1:
        F1 = debug['F'][0] if len(debug['F']) > 0 else F0
        print(f"  F₁ = {F1:.6f} (预期: 1.485)")
    if iterations >= 2:
        F2 = debug['F'][1] if len(debug['F']) > 1 else F
        print(f"  F₂ = {F2:.6f} (预期: 1.48)")
    print(f"  最终 F = {F:.6f} (预期: 1.48)")
    print(f"  收敛状态: {'已收敛' if converged else '未收敛'}")
    print(f"  迭代次数: {iterations}")
    
    # ============================================================
    # 5. 总结
    # ============================================================
    print(f"\n" + "=" * 100)
    print("验证总结")
    print("=" * 100)
    
    error_F0 = abs(F0 - 1.385)
    error_F_final = abs(F - 1.48)
    
    print(f"F₀ 误差: {error_F0:.6f} ({error_F0/1.385*100:.2f}%)")
    print(f"最终F 误差: {error_F_final:.6f} ({error_F_final/1.48*100:.2f}%)")
    
    if error_F0 < 0.01 and error_F_final < 0.01:
        print("\n✓ 验证通过！计算结果与预期值一致。")
    else:
        print("\n✗ 验证未通过！计算结果与预期值存在差异。")
        print("  请检查janbu.py中的计算逻辑。")
    
    print("=" * 100)


if __name__ == "__main__":
    main()
