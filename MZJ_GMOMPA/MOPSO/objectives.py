import numpy as np


def get_objective(flag, funct, num_obj, Input):
    """
    Evaluates fitness values of the swarm
    :param flag: Flag for returning bounds and computing fitness
    :param funct: Function number
    :param num_obj: Number of objective functions
    :param Input: Input swarm
    Returns bounds and dimensions of particles when flag is "init"
    Returns computed fitness values when flag is "value"
    """
    if flag == "init":
        if funct == 1:
            # ZDT2
            dim = 30
            MaxValue = np.ones((1, dim))
            MinValue = np.zeros((1, dim))
        elif funct == 2:
            # Tanaka
            MaxValue = np.array([np.pi, np.pi])
            MinValue = np.array([0, 0])
        elif funct == 3:
            # Osyczka
            MaxValue = np.array([10, 10, 5, 6, 5, 10])
            MinValue = np.array([0, 0, 1, 0, 1, 0])
        elif funct == 4:
            # Binh and Korn
            MaxValue = np.array([5, 3])
            MinValue = np.array([0, 0])
        else:
            # IGHS
            MaxValue = np.array([2, 90.4, 732, 2.6]) + 1e-5
            MinValue = np.array([2, 74.1, 609, 1.8])
        bounds = np.vstack((MinValue, MaxValue))
        return bounds
    elif flag == "value":
        FunctionValue = np.zeros((Input.shape[0], num_obj))
        if funct == 1:
            # ZDT2
            FunctionValue[:, 0] = Input[:, 0]
            c = np.sum(FunctionValue[:, 1:], axis=1)
            g = 1.0 + 9.0 * c / 29
            FunctionValue[:, 1] = g * (1 - ((FunctionValue[:, 0] / g)**2))
        elif funct == 2:
            # Tanaka
            FunctionValue[:, 0] = Input[:, 0]
            FunctionValue[:, 1] = Input[:, 1]
        elif funct == 3:
            # Osyczka
            FunctionValue[:, 0] = -25 * \
                ((Input[:, 0]-2)**2) - \
                ((Input[:, 1]-2)**2) - \
                ((Input[:, 2]-1)**2) - \
                ((Input[:, 3]-4)**2) - \
                ((Input[:, 4]-1)**2)
            FunctionValue[:, 1] = (Input[:, 0] ** 2) + \
                (Input[:, 1] ** 2) + \
                (Input[:, 2] ** 2) + \
                (Input[:, 3] ** 2) + \
                (Input[:, 4] ** 2) + \
                (Input[:, 5] ** 2)
        elif funct == 4:
            # Binh and Korn
            FunctionValue[:, 0] = (4 * (Input[:, 0]**2)) + \
                (4 * (Input[:, 1]**2))
            FunctionValue[:, 1] = ((Input[:, 0]-5)**2) + \
                ((Input[:, 1]-5)**2)
        else:
            # IGHS
            for i in range(Input.shape[0]):
                FunctionValue[i, :] = IGHS_fitness(Input[i, :])
        return FunctionValue


def IGHS_fitness(p, u=None):
    if u is None:
        u = np.array([2.5, 45, 0.349, 0.297, 128.63, 45, 8.45])
    E, T, _, _ = energy_and_time(p, u)
    Q = quality(p, u)
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
