import cupy as cp
from cupy.linalg import inv, norm, solve


# NOTE: Ported to Python from l1-MAGIC: Cand\'es & Romberg,
# ``l_1-MAGIC: Recovery of Sparse Signals via Convex Programming,"
# Technical Report, 2005.
def basis_pursuit(A, b, tol=1e-4, niter=100, biter=32):
    """
    solves min |x|_1 s.t. Ax=b using a Primal-Dual Interior Point Method

    Args:
      A: design matrix of size (d, n)
      b: measurement vector of length d
      tol: solver tolerance
      niter: maximum length of central path
      biter: maximum number of steps in backtracking line search

    Returns:
      vector of length n
    """
    A = cp.asarray(A)
    b = cp.asarray(b)
    d, n = A.shape
    alpha = 0.01
    beta = 0.5
    mu = 10
    e = cp.ones(n)
    gradf0 = cp.hstack([cp.zeros(n), e])
    x = (A.T).dot(inv(A.dot(A.T))).dot(b)
    absx = cp.abs(x)
    u = 0.95 * absx + 0.1 * cp.max(absx)

    fu1 = x - u
    fu2 = -x - u
    lamu1 = -1.0 / fu1
    lamu2 = -1.0 / fu2
    v = A.dot(lamu2 - lamu1)
    ATv = (A.T).dot(v)
    sdg = -(cp.inner(fu1, lamu1) + cp.inner(fu2, lamu2))
    tau = 2.0 * n * mu / sdg
    ootau = 1.0 / tau

    rcent = cp.hstack([-lamu1 * fu1, -lamu2 * fu2]) - ootau
    rdual = gradf0 + cp.hstack([lamu1 - lamu2 + ATv, -lamu1 - lamu2])
    rpri = A.dot(x) - b
    resnorm = cp.sqrt(norm(rdual) ** 2 + norm(rcent) ** 2 + norm(rpri) ** 2)
    rdp = cp.empty(2 * n)
    rcp = cp.empty(2 * n)

    for i in range(niter):

        oofu1 = 1.0 / fu1
        oofu2 = 1.0 / fu2
        w1 = -ootau * (oofu2 - oofu1) - ATv
        w2 = -1.0 - ootau * (oofu1 + oofu2)
        w3 = -rpri

        lamu1xoofu1 = lamu1 * oofu1
        lamu2xoofu2 = lamu2 * oofu2
        sig1 = -lamu1xoofu1 - lamu2xoofu2
        sig2 = lamu1xoofu1 - lamu2xoofu2
        sigx = sig1 - sig2 ** 2 / sig1
        if cp.min(cp.abs(sigx)) == 0.0:
            break

        w1p = -(w3 - A.dot(w1 / sigx - w2 * sig2 / (sigx * sig1)))
        H11p = A.dot((A.T) * (e / sigx)[:, cp.newaxis])
        if cp.min(sigx) > 0.0:
            dv = solve(H11p, w1p)
        else:
            dv = solve(H11p, w1p)
        dx = (w1 - w2 * sig2 / sig1 - (A.T).dot(dv)) / sigx
        Adx = A.dot(dx)
        ATdv = (A.T).dot(dv)

        du = (w2 - sig2 * dx) / sig1
        dlamu1 = lamu1xoofu1 * (du - dx) - lamu1 - ootau * oofu1
        dlamu2 = lamu2xoofu2 * (dx + du) - lamu2 - ootau * oofu2

        s = 1.0
        indp = cp.less(dlamu1, 0.0)
        indn = cp.less(dlamu2, 0.0)
        if cp.any(indp):
            s = min(s, cp.min(-lamu1[indp] / dlamu1[indp]))
        if cp.any(indn):
            s = min(s, cp.min(-lamu2[indn] / dlamu2[indn]))
        indp = cp.greater(dx - du, 0.0)
        indn = cp.greater(-dx - du, 0.0)
        if cp.any(indp):
            s = min(s, cp.min(-fu1[indp] / (dx[indp] - du[indp])))
        if cp.any(indn):
            s = min(s, cp.min(-fu2[indn] / (-dx[indn] - du[indn])))
        s = 0.99 * s

        for j in range(biter):
            xp = x + s * dx
            up = u + s * du
            vp = v + s * dv
            ATvp = ATv + s * ATdv
            lamu1p = lamu1 + s * dlamu1
            lamu2p = lamu2 + s * dlamu2
            fu1p = xp - up
            fu2p = -xp - up
            rdp[:n] = lamu1p - lamu2p + ATvp
            rdp[n:] = -lamu1p - lamu2p
            rdp += gradf0
            rcp[:n] = -lamu1p * fu1p
            rcp[n:] = lamu2p * fu2p
            rcp -= ootau
            rpp = rpri + s * Adx
            s *= beta
            if (
                cp.sqrt(norm(rdp) ** 2 + norm(rcp) ** 2 + norm(rpp) ** 2)
                <= (1 - alpha * s) * resnorm
            ):
                break
        else:
            break

        x = xp
        lamu1 = lamu1p
        lamu2 = lamu2p
        fu1 = fu1p
        fu2 = fu2p
        sdg = -(cp.inner(fu1, lamu1) + cp.inner(fu2, lamu2))
        if sdg < tol:
            return cp.asnumpy(x)

        u = up
        v = vp
        ATv = ATvp
        tau = 2.0 * n * mu / sdg
        rpri = rpp
        rcent[:n] = lamu1 * fu1
        rcent[n:] = lamu2 * fu2
        ootau = 1.0 / tau
        rcent -= ootau
        rdual[:n] = lamu1 - lamu2 + ATv
        rdual[n:] = -lamu1 + lamu2
        rdual += gradf0
        resnorm = cp.sqrt(norm(rdual) ** 2 + norm(rcent) ** 2 + norm(rpri) ** 2)

    return cp.asnumpy(x)
