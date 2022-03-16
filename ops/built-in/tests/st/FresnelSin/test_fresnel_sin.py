import numpy as np


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # Taylor coefficient
    THRESHOLD_A = 2.5625
    THRESHOLD_B = 36974.0
    THRESHOLD_PI = 3.1415926535897932384626433832795
    SN_COUNT = 5
    SD_COUNT = 6
    CN_COUNT = 5
    CD_COUNT = 6
    FN_COUNT = 9
    FD_COUNT = 10
    GN_COUNT = 10
    GD_COUNT = 11
    COEF_CN = (-4.98843114573573548651E-8, 9.50428062829859605134E-6,
               -6.45191435683965050962E-4, 1.88843319396703850064E-2,
               -2.05525900955013891793E-1, 9.99999999999999998822E-1)
    COEF_CD = (3.99982968972495980367E-12, 9.15439215774657478799E-10,
               1.25001862479598821474E-7, 1.22262789024179030997E-5,
               8.68029542941784300606E-4, 4.12142090722199792936E-2,
               1.00000000000000000118E0)
    COEF_SN = (-2.99181919401019853726E3, 7.08840045257738576863E5,
               -6.29741486205862506537E7, 2.54890880573376359104E9,
               -4.42979518059697779103E10, 3.18016297876567817986E11)
    COEF_SD = (1.0,
               2.81376268889994315696E2,
               4.55847810806532581675E4,
               5.17343888770096400730E6,
               4.19320245898111231129E8,
               2.24411795645340920940E10,
               6.07366389490084639049E11)

    COEF_FN = (4.21543555043677546506E-1, 1.43407919780758885261E-1,
               1.15220955073585758835E-2, 3.45017939782574027900E-4,
               4.63613749287867322088E-6, 3.05568983790257605827E-8,
               1.02304514164907233465E-10, 1.72010743268161828879E-13,
               1.34283276233062758925E-16, 3.76329711269987889006E-20)
    COEF_FD = (1.0,
               7.51586398353378947175E-1,
               1.16888925859191382142E-1,
               6.44051526508858611005E-3,
               1.55934409164153020873E-4,
               1.84627567348930545870E-6,
               1.12699224763999035261E-8,
               3.60140029589371370404E-11,
               5.88754533621578410010E-14,
               4.52001434074129701496E-17,
               1.25443237090011264384E-20)
    COEF_GN = (5.04442073643383265887E-1, 1.97102833525523411709E-1,
               1.87648584092575249293E-2, 6.84079380915393090172E-4,
               1.15138826111884280931E-5, 9.82852443688422223854E-8,
               4.45344415861750144738E-10, 1.08268041139020870318E-12,
               1.37555460633261799868E-15, 8.36354435630677421531E-19,
               1.86958710162783235106E-22)
    COEF_GD = (1.0,
               1.47495759925128324529E0,
               3.37748989120019970451E-1,
               2.53603741420338795122E-2,
               8.14679107184306179049E-4,
               1.27545075667729118702E-5,
               1.04314589657571990585E-7,
               4.60680728146520428211E-10,
               1.10273215066240270757E-12,
               1.38796531259578871258E-15,
               8.39158816283118707363E-19,
               1.86958710162783236342E-22)


def polevl(data_x, coef, num):
    """
    y = polevl( x, coef, num );
    DESCRIPTION:    
    Evaluates polynomial of degree N:
                        2          N
    y  =  C  + C x + C x  +...+ C x
             0    1     2          N
    Coefficients are stored in reverse order:
    coef[0] = C  , ..., coef[N] = C  .
                 N                   0
    Parameters:
    ----------
    data_x : the placeholder of data input
    coef : coef of the data
    iter_n : number of the coef
     Returns : A Tensor. Has the same type as data.
    -------
    """
    res = coef[0]
    for index in range(1, num + 1):
        mul_res = np.multiply(res, data_x)
        res = np.add(mul_res, coef[index])
    return res


def p1evl(data_x, coef, num):
    """
    y = p1evl( x, coef, num );
    DESCRIPTION:    
    Evaluates polynomial of degree N:
                        2          N
    y  =  C  + C x + C x  +...+ C x
             0    1     2          N
    Coefficients are stored in reverse order:
    coef[0] = C  , ..., coef[N] = C  .
                 N                   0
    The function p1evl() assumes that coef[N] = 1.0 and is
    omitted from the array.  Its calling arguments are
    otherwise the same as polevl().
    -------
    """
    res = np.add(data_x, coef[0])
    for index in range(1, num):
        mul_res = np.multiply(res, data_x)
        res = np.add(mul_res, coef[index])
    return res


def generic_fresnel_asymp(x):
    """
    do internel fresnel sin asymp compute
    rec = 1 / (π * x^2), 
    f = 1 - rec^2 * polevel(rec^2, FN, 9) / plevel(rec^2, FD, 10),
    g = rec * polevel(rec^2, GN, 10) / plevel(rec^2, GD, 11)
    res = 0.5 - 1 / (π * x) * (f * cos(π * x^2 / 2) + g * sin(π * x^2 / 2))
    """
    tmp_x2 = np.multiply(x, x)
    data_pi = np.multiply(tmp_x2, Constant.THRESHOLD_PI)
    data_rec = np.true_divide(1.0, data_pi)
    data_square = np.multiply(data_rec, data_rec)
    data_fn_polevl = polevl(data_square, Constant.COEF_FN, Constant.FN_COUNT)
    data_fd_p1evl = p1evl(data_square, Constant.COEF_FD, Constant.FD_COUNT)
    data_gn_polevl = polevl(data_square, Constant.COEF_GN, Constant.GN_COUNT)
    data_gd_p1evl = p1evl(data_square, Constant.COEF_GD, Constant.GD_COUNT)

    data_f = np.multiply(data_square, data_fn_polevl)
    data_f = np.true_divide(data_f, data_fd_p1evl)
    data_f = np.subtract(1.0, data_f)

    data_g = np.multiply(data_rec, data_gn_polevl)
    data_g = np.true_divide(data_g, data_gd_p1evl)

    data_z = np.multiply(tmp_x2, Constant.THRESHOLD_PI / 2)
    data_c = np.cos(data_z)
    data_s = np.sin(data_z)
    data_y = np.multiply(x, Constant.THRESHOLD_PI)
    data_y = np.true_divide(1.0, data_y)

    res_1 = np.multiply(data_f, data_c)
    res_2 = np.multiply(data_g, data_s)
    res_add = np.add(res_1, res_2)
    res_3 = np.multiply(res_add, data_y)
    res = np.subtract(0.5, res_3)
    return res


def _generic_fresnel_sin_interval(x):
    """
    do internel fresnel_sin compute
    res = x^3 * polevel(x^4, SN, 5) / plevel(x^4, SD, 6)
    """
    tmp_x2 = np.multiply(x, x)
    tmp_x3 = np.multiply(x, tmp_x2)
    tmp_x4 = np.multiply(tmp_x2, tmp_x2)
    data_sn_polevl = polevl(tmp_x4, Constant.COEF_SN, Constant.SN_COUNT)
    data_sd_p1evl = p1evl(tmp_x4, Constant.COEF_SD, Constant.SD_COUNT)
    res = np.multiply(tmp_x3, data_sn_polevl)
    res = np.true_divide(res, data_sd_p1evl)

    return res


# 'pylint': disable=unused-argument
def calc_expect_func(x, y):
    """
    do element-wise fresnel_sin compute
    |x| > b, y = 0.5
    x^2 >= a, rec = 1 / (π * x^2), 
              f = 1 - rec^2 * polevel(rec^2, FN, 9) / plevel(rec^2, FD, 10),
              g = rec * polevel(rec^2, GN, 10) / plevel(rec^2, GD, 11)
              y = 0.5 - 1 / (π * x) * (f * cos(π * x^2 / 2) + g * sin(π * x^2 / 2))
    x^2 < a, y = x^3 * polevel(x^4, SN, 5) / plevel(x^4, SD, 6)
    Parameters:
    x: the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name, default value is "fresnel_sin"

    Returns : A Tensor. Has the same type as data_input.
    -------
    """
    input_x = x["value"]
    for i in np.nditer(input_x, op_flags=['readwrite']):
        sign = 1
        if i < 0:
            i[...] = -i
            sign = -1
        if i > Constant.THRESHOLD_B:
            res = 0.5
        elif i * i < Constant.THRESHOLD_A:
            res = _generic_fresnel_sin_interval(i)
        else:
            res = generic_fresnel_asymp(i)
        i[...] = np.multiply(res, sign)
    return [input_x, ]
