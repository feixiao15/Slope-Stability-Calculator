import numpy as np

class JanbuReproductionSolver:
    """
    专门用于复现 Janbu 1973 Table 1 的求解器。
    核心逻辑与标准 GPS 一致，但允许强制指定推力线几何参数。
    """
    
    def __init__(self):
        pass

    def calculate_fos_gps_reproduction(self, slices, geometry_override, F_init=1.4, max_iter=3):
        """
        执行 GPS 迭代。
        
        Parameters
        ----------
        slices : list
            包含土条基本数据的列表 (p, u, c, phi, width, alpha...)
        geometry_override : dict
            包含 'ht' 和 'tan_alpha_t' 的数组，用于强制覆盖自动计算的几何参数。
            这是复现论文数据的关键，因为 Janbu 是手测的这些值。
        """
        n = len(slices)
        
        # 1. 提取基础数据 (对应 Table 1 Cols 1-7)
        dx = np.array([s['dx'] for s in slices])
        tan_alpha = np.array([s['tan_alpha'] for s in slices])
        alpha = np.arctan(tan_alpha)
        cos_alpha = np.cos(alpha)
        
        p = np.array([s['p'] for s in slices])
        u = np.array([s['u'] for s in slices])
        c = np.array([s['c'] for s in slices])
        tan_phi = np.array([s['tan_phi'] for s in slices])
        dQ = np.zeros(n) # 论文中 ΔQ = 0

        # 2. 提取强制几何参数 (对应 Table 1 Cols 16-17)
        # 注意：Janbu 表格给出的是 Interface 1, 2, 3 的值
        # 数组长度应为 n+1 (包含左右边界 0 和 4)
        # 我们需要把 geometry_override 映射到完整的 interface 数组
        h_t_interface = geometry_override['ht']          # 长度 n+1
        tan_alpha_t = geometry_override['tan_alpha_t']   # 长度 n+1

        # 初始化变量
        F_old = F_init
        t = np.zeros(n)  # 初始 t=0
        
        # 记录每一步结果用于打印
        history = []

        print(f"{'='*80}")
        print(f"开始 Janbu GPS 复现 (F_init = {F_init})")
        print(f"{'='*80}")

        # ----------------------------------------------------------------
        # 迭代循环
        # ----------------------------------------------------------------
        for it in range(max_iter):
            print(f"\n>>> Iteration Step {it} (计算 F_{it})")
            
            # --- Step A: 计算 A, B, n_alpha, ΔE ---
            # n_alpha 依赖于 F_old (Eq. 96)
            n_alpha = (cos_alpha**2) * (1.0 + (tan_alpha * tan_phi) / F_old)
            
            # A' (不除以 n_alpha 的部分) = [c' + (p+t-u)tan_phi] * dx * (1+tan^2 alpha)
            # 注意：Janbu 表格里的 A' (Col 9) 实际上是不乘 (1+tan^2) 的，
            # 但为了计算方便，我们算标准项，打印时还原回表格格式
            # 有效应力 sigma' 近似项
            sigma_term = p + t - u
            
            # Janbu Table 1 Col 9: A0' = [c' + (p-u)tan_phi] * dx
            # 这是一个中间量，我们按表格定义计算以便对比
            A_prime_table = (c + sigma_term * tan_phi) * dx
            
            # 真正的 A (Col 11) = A_prime_table / n_alpha
            # 注意：这里有一个微妙的数学等价。
            # Janbu 定义 n_alpha = (1 + tan a tan phi / F) / (1 + tan^2 a)
            # 所以 A / n_alpha = (A_prime * (1+tan^2)) / ( (1+...) / (1+tan^2) ) ? 
            # 不，按论文 Eq 97: A = A' / n_alpha. 
            # 让我们直接用代码中的标准公式算 A_term (Col 11 的分子部分)
            # A_term = A_prime_table * (1 + tan_alpha**2) / (1 + tan_alpha*tan_phi/F) * ...
            # 让我们严格遵循表格公式：
            # A (Col 11) = A_prime_table / n_alpha
            A_final = A_prime_table / n_alpha
            
            # B (Col 8/21)
            B = dQ + (p + t) * dx * tan_alpha
            
            # ΔE (Col 12/25) = B - A/F
            delta_E = B - A_final / F_old
            
            # E (Col 13/26) - 累加
            E_interface = np.zeros(n + 1)
            E_interface[1:] = np.cumsum(delta_E)
            
            # --- Step B: 计算 F_new ---
            sum_A = np.sum(A_final)
            sum_B = np.sum(B)
            E_a = 0
            E_b = 0 # E_interface[-1]
            # 按公式 114: F = ΣA / (Ea - Eb + ΣB)
            # 注意：Janbu 假设整体平衡 ΣΔE = 0，所以分母其实就是 ΣB - (Eb-Ea)
            # 在表格中，他直接用 ΣA / (ΣB + Ea - Eb)
            denom = sum_B + E_a - E_interface[-1] 
            F_new = sum_A / denom

            # --- Step C: 计算 dE/dx, T, t (为下一轮准备) ---
            # dE/dx 使用加权平均 (Eq. 117)
            dE_dx = np.zeros(n + 1)
            for i in range(1, n):
                # 界面 i 位于 slice i-1 和 slice i 之间
                dE_dx[i] = (delta_E[i-1] + delta_E[i]) / (dx[i-1] + dx[i])
            # 边界简单处理 (参考表格数据，Row 1 dE/dx=1.74)
            # Slice 1 左边是 0? Janbu 表格第一行没有填 Interface 0 的数据
            # 我们只需要计算 Interface 1, 2, 3
            
            # T (Interface) = -E * tan_alpha_t + h_t * dE/dx
            # 注意：计算 T 时使用的是"当前这一轮算出来的 E"吗？
            # Janbu 表格 Iteration 1 中，计算 T1 用的是 E0 (Col 14) !
            # 所以我们计算 T 时，要用 E_interface (当前轮) 还是 E_prev (上一轮)?
            # Table 1 显示：Iteration 1 这一栏里，左边算 T，右边算 F1, E1
            # 列 (14) E0 是来自 Initial Step 的。
            # 所以：计算 T 用的是上一轮的 E！
            
            T_interface = -E_interface * tan_alpha_t + h_t_interface * dE_dx
            # 强制边界条件
            T_interface[0] = 0
            T_interface[-1] = 0

            # t (Slice) = ΔT / Δx
            t_new = np.zeros(n)
            delta_T = np.zeros(n)
            for i in range(n):
                delta_T[i] = T_interface[i+1] - T_interface[i]
                t_new[i] = delta_T[i] / dx[i]

            # 打印本轮结果 (模拟表格格式)
            self._print_table_row(it, slices, B, A_prime_table, n_alpha, A_final, delta_E, E_interface, 
                                  F_old, F_new, E_interface, dE_dx, tan_alpha_t, h_t_interface, T_interface, delta_T, t_new)
            
            # 更新状态
            F_old = F_new
            t = t_new # t 将在下一轮用于计算 A 和 B
            
        return F_new

    def _print_table_row(self, it, slices, B, A_prime, n_alpha, A, dE, E, 
                         F_in, F_out, E_prev, dE_dx, tan_at, ht, T, dT, t):
        print(f"\n--- Iteration {it} 数据核对 ---")
        print(f"Input F = {F_in:.3f} --> Calculated F = {F_out:.3f}")
        
        a_prime_label = "A'"
        print(f"{'Slice':<6} {'B':<8} {a_prime_label:<8} {'n_a':<6} {'A':<8} {'dE':<8} {'E_int':<8} | {'dE/dx':<8} {'tan_at':<8} {'ht':<6} {'T_int':<8} {'t_slice':<8}")
        print("-" * 110)
        
        for i in range(len(slices)):
            # 界面索引：i+1 是右侧界面
            idx_int = i + 1
            print(f"{i+1:<6} {B[i]:<8.1f} {A_prime[i]:<8.1f} {n_alpha[i]:<6.2f} {A[i]:<8.1f} {dE[i]:<8.1f} {E[idx_int]:<8.1f} | "
                  f"{dE_dx[idx_int]:<8.2f} {tan_at[idx_int]:<8.2f} {ht[idx_int]:<6.1f} {T[idx_int]:<8.2f} {t[i]:<8.2f}")
        
        print(f"{'Sum':<6} {np.sum(B):<8.1f} {'-':<8} {'-':<6} {np.sum(A):<8.1f} {np.sum(dE):<8.1f}")


# ==============================================================================
# 数据输入 (来源于 Janbu 1973 Table 1 & Fig 22)
# ==============================================================================

# 1. 土条基础数据 (Cols 1-6)
# 注意：gamma=2, c=1, phi=33.8 (tan=0.67)
# Slice 1: tan_a=1.13, dx=4.4, p=5.3, u=2.12
# Slice 2: tan_a=0.50, dx=11.0, p=10.1, u=4.04
# Slice 3: tan_a=0.18, dx=11.0, p=8.6, u=3.44
# Slice 4: tan_a=-0.04, dx=6.0, p=2.9, u=1.16

slices_data = [
    {'id': 1, 'tan_alpha': 1.13,  'dx': 4.4,  'p': 5.3,  'u': 2.12, 'c': 1.0, 'tan_phi': 0.67},
    {'id': 2, 'tan_alpha': 0.50,  'dx': 11.0, 'p': 10.1, 'u': 4.04, 'c': 1.0, 'tan_phi': 0.67},
    {'id': 3, 'tan_alpha': 0.18,  'dx': 11.0, 'p': 8.6,  'u': 3.44, 'c': 1.0, 'tan_phi': 0.67},
    {'id': 4, 'tan_alpha': -0.04, 'dx': 6.0,  'p': 2.9,  'u': 1.16, 'c': 1.0, 'tan_phi': 0.67},
]

# 2. 推力线几何数据 (Cols 16-17)
# 这些是界面(Interface)属性。长度为 5 (0, 1-2, 2-3, 3-4, 4)
# Table 1 数据：
# Interface 1 (Slice 1右): tan_at = 0.63, ht = 1.2
# Interface 2 (Slice 2右): tan_at = 0.33, ht = 1.8
# Interface 3 (Slice 3右): tan_at = 0.16, ht = 1.1
# Interface 0 和 4 (边界) 设为 0
ht_input = np.array([0.0, 1.2, 1.8, 1.1, 0.0])
tan_at_input = np.array([0.0, 0.63, 0.33, 0.16, 0.0])

geo_override = {
    'ht': ht_input,
    'tan_alpha_t': tan_at_input
}

# ==============================================================================
# 运行求解
# ==============================================================================

solver = JanbuReproductionSolver()

# 初始 F 设为 1.4 (对应表格 Start of Iteration)
# max_iter = 3 (Initial -> Iter 1 -> Iter 2)
# 注意：我们的代码逻辑中，Step 0 相当于表格的 "Initial Step"
# Step 1 相当于表格的 "Calculation of T1 ... Calculation of F1"
# Step 2 相当于表格的 "Calculation of T2 ... Calculation of F2"

solver.calculate_fos_gps_reproduction(slices_data, geo_override, F_init=1.4, max_iter=3)