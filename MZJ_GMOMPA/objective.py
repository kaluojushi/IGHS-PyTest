import numpy as np
from test_data import test_data as td

# Fitness function
# 适应度函数
def fitness(p, u=None):
    if u is None:
        u = td.values()
    E, T, _, _ = energy_and_time(p, u)
    Q = quality(p, u)
    return np.array([E, T, Q])


def fitness_(p):
    z0, da0, n0, f = p
    Pr = 23.33409 * da0 ** -0.26 * n0 ** 0.74 * f ** 0.65
    E = ((2400 - 0.078 * n0 + 2e-6 * n0 ** 2) * (104.5 / 1500 + 3157.56 / (z0 * n0 * f)) + (1.035 * Pr + 1.3e-5 * Pr ** 2) * 2205 / (z0 * n0 * f) + 1.1e4) / 6e4
    T = 304.18 + 189453.6 / (z0 * n0 * f)
    Q = 4.27525e-2 * f ** 2 / da0 + 8.11131e-5 * z0 ** 2
    return np.array([E, T, Q])


def energy_and_time(p, u):
    z0, da0, n0, f = p
    zk, gamma = 17, 0.087
    m, z1, alpha, beta, de1, B, ap = u
    lamda = beta + gamma
    Ps, Ts = 2400, 4.5
    Es = Ps * Ts
    Psc = 225
    kappa1, kappa2, kappa3 = 68.2, 1.824, 0.0024
    Pn = kappa1 + kappa2 * n0 + kappa3 * n0 ** 2
    Pa = Ps + Psc + Pn
    vf, vr = 4000, 1500
    va = np.pi * m * z0 * n0 * f / (np.pi * m * z1 - f * np.sin(beta))
    dp01, dp02 = ap / 2, ap / 2
    ra0, de0 = da0 / 2, de1 / 2
    Lin1 = np.sqrt(2 * ra0 * dp01 - dp01 ** 2) / np.cos(lamda)
    Lin2 = np.sqrt(2 * ra0 * dp02 - dp02 ** 2) / np.cos(lamda)
    LfX1, LfZ1 = 85, 152.5
    La1 = Lin1
    La2 = ra0 / np.cos(lamda) + 2
    Lr1 = 105
    Lr2 = dp02
    Lr3 = dp02 + Lr1
    LfX2, LfZ2 = LfX1, La2 - Lin2 + LfZ1
    Ta = (np.sqrt(LfX1 ** 2 + LfZ1 ** 2) + np.sqrt(LfX2 ** 2 + LfZ2 ** 2)) / vf + (La1 + La2) / va + (Lr1 + Lr2 + Lr3) / vr
    Ea = Pa * Ta

    def phi_(t, dp0):
        return np.sqrt((ra0 - dp0) ** 2 + 2 * np.cos(lamda) * va * np.sqrt(2 * ra0 * dp0 - dp0 ** 2) * t - (np.cos(lamda) * va * t) ** 2)

    def dp_(t, dp0, Lin):
        if t == 0:
            return 0
        if t == (B + Lin) / va:
            return 0
        if t <= Lin / va:
            return phi_(t, dp0) - ra0 + dp0
        elif t <= B / va:
            return dp0
        else:
            return ra0 - phi_(t - B / va, dp0)

    def Pr_(t, dp0, Lin):
        g = 9.80665
        Kma = 1
        Kz0 = [1, 1.5, 2.05][int(z0 - 1)]
        beta_d = beta * 180 / np.pi
        return 2000 * 2 ** 0.6 * g * Kma * Kz0 * m ** 1.55 * f ** 0.8 * dp_(t, dp0, Lin) ** 0.15 * da0 ** 0.12 * n0 ** 0.72 * np.exp(0.012 * beta_d + 0.65 * z1 ** -0.35) * np.pi ** 0.72 / (60 * 1000 ** 0.72 * zk ** 0.7)

    def Pc_(t, dp0, Lin):
        eps1, eps2, eps3 = 0, 0.035, 1.3e-5
        Pr = Pr_(t, dp0, Lin)
        Pap = eps1 + eps2 * Pr + eps3 * Pr ** 2
        return Ps + Psc + Pn + Pr + Pap

    Tc1, Tc2 = (B + Lin1) / va, (B + Lin2) / va
    tc1, tc2 = np.linspace(0, Tc1, 1000), np.linspace(0, Tc2, 1000)
    Pc1, Pc2 = [Pc_(t, dp01, Lin1) for t in tc1], [Pc_(t, dp02, Lin2) for t in tc2]
    Ec1, Ec2 = np.trapz(Pc1, tc1), np.trapz(Pc2, tc2)
    # Pc1, Pc2 = max(Pc1), max(Pc2)
    # Ec1, Ec2 = Pc1 * Tc1, Pc2 * Tc2
    Ec = Ec1 + Ec2
    Tc = Tc1 + Tc2
    E = (Es + Ea + Ec) / 6e4
    T = (Ts + Ta + Tc) * 60
    return E, T, Pc1, Pc2


def quality(p, u):
    z0, da0, n0, f = p
    zk = 17
    m, z1, alpha, beta, de1, B, ap = u
    omega1, omega2 = 0.5, 0.5
    delta1 = np.pi ** 2 * m * z0 ** 2 * np.sin(alpha) / (4 * z1 * zk ** 2)
    delta2 = f ** 2 * np.sin(alpha) / (4 * da0)
    Q = omega1 * delta1 + omega2 * delta2
    return Q


# Constraints
# 约束条件
def constraints(p, u=None):
    if u is None:
        u = td.values()
    z0, da0, n0, f = p
    vc = np.pi * da0 * n0 / 1000
    _, _, Pc1, Pc2 = energy_and_time(p, u)
    Fc1, Fc2 = max(Pc1) * 60 / vc, max(Pc2) * 60 / vc
    # Fc1, Fc2 = Pc1 * 60 / vc, Pc2 * 60 / vc
    Fc = max(Fc1, Fc2)
    Fc_max = 1e5
    rt, Ra_limit = 0.6, 1.6
    Ra = 0.0312 * f * f / rt
    return np.array([Fc - Fc_max, Ra - Ra_limit]) <= 0

if __name__ == '__main__':
    print(fitness([2, 88.5, 745, 2.5]))
    print(fitness([2, 86.5, 700, 2.5]))
    print(constraints([2, 89, 756, 2.6]))
    print(fitness([2, 88.5, 745, 2]))
