import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import solve


def omp(A, b, tol=1e-4, n_nonzero_coefs=None, positive=False):
    """approximately solves min_x |x|_0 s.t. Ax=b using Orthogonal Matching Pursuit
  Args:
    A: design matrix of size (d, n)
    b: measurement vector of length d
    tol: solver tolerance
    nnz = maximum number of nonzero coefficients (if None set to n)
    positive: only allow positive nonzero coefficients
  Returns:
     vector of length n
  """
    nnz = n_nonzero_coefs
    AT = A.T
    d, n = A.shape
    if nnz is None:
        nnz = n
    x = np.zeros(n)
    resid = np.copy(b)
    normb = norm(b)
    indices = []

    for i in range(nnz):
        if norm(resid) / normb < tol:
            break
        projections = AT.dot(resid)
        if positive:
            index = np.argmax(projections)
        else:
            index = np.argmax(abs(projections))
        if index in indices:
            break
        indices.append(index)
        if len(indices) == 1:
            A_i = A[:, index]
            x_i = projections[index] / A_i.T.dot(A_i)
        else:
            A_i = np.vstack([A_i, A[:, index]])
            x_i = solve(A_i.dot(A_i.T), A_i.dot(b), assume_a="sym")
            if positive:
                while min(x_i) < 0.0:
                    argmin = np.argmin(x_i)
                    indices = indices[:argmin] + indices[argmin + 1 :]
                    A_i = np.vstack([A_i[:argmin], A_i[argmin + 1 :]])
                    x_i = solve(A_i.dot(A_i.T), A_i.dot(b), assume_a="sym")
        resid = b - A_i.T.dot(x_i)

    for i, index in enumerate(indices):
        try:
            x[index] += x_i[i]
        except IndexError:
            x[index] += x_i
    return np.maximum(np.round(x), 0.0)


# NOTE: Ported to Python from l1-MAGIC: Cand\'es & Romberg,
# ``l_1-MAGIC: Recovery of Sparse Signals via Convex Programming,"
# Technical Report, 2005.
def basis_pursuit(A, b, x0=None, ATinvAAT=None, tol=1e-4, niter=100, biter=32):
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
    AT = A.T
    d, n = A.shape
    alpha = 0.01
    beta = 0.5
    mu = 10
    e = np.ones(n)
    gradf0 = np.hstack([np.zeros(n), e])
    x = AT.dot(inv(A.dot(AT))).dot(b)
    absx = np.abs(x)
    u = 0.95 * absx + 0.1 * max(absx)

    fu1 = x - u
    fu2 = -x - u
    lamu1 = -1.0 / fu1
    lamu2 = -1.0 / fu2
    v = A.dot(lamu2 - lamu1)
    ATv = AT.dot(v)
    sdg = -(np.inner(fu1, lamu1) + np.inner(fu2, lamu2))
    tau = 2.0 * n * mu / sdg
    ootau = 1.0 / tau

    rcent = np.hstack([-lamu1 * fu1, -lamu2 * fu2]) - ootau
    rdual = gradf0 + np.hstack([lamu1 - lamu2 + ATv, -lamu1 - lamu2])
    rpri = A.dot(x) - b
    resnorm = np.sqrt(norm(rdual) ** 2 + norm(rcent) ** 2 + norm(rpri) ** 2)
    rdp = np.empty(2 * n)
    rcp = np.empty(2 * n)

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
        if min(np.abs(sigx)) == 0.0:
            break

        w1p = -(w3 - A.dot(w1 / sigx - w2 * sig2 / (sigx * sig1)))
        H11p = A.dot(AT * (e / sigx)[:, np.newaxis])
        if min(sigx) > 0.0:
            dv = solve(H11p, w1p, assume_a="pos")
        else:
            dv = solve(H11p, w1p, assume_a="sym")
        dx = (w1 - w2 * sig2 / sig1 - AT.dot(dv)) / sigx
        Adx = A.dot(dx)
        ATdv = AT.dot(dv)

        du = (w2 - sig2 * dx) / sig1
        dlamu1 = lamu1xoofu1 * (du - dx) - lamu1 - ootau * oofu1
        dlamu2 = lamu2xoofu2 * (dx + du) - lamu2 - ootau * oofu2

        s = 1.0
        indp = np.less(dlamu1, 0.0)
        indn = np.less(dlamu2, 0.0)
        if np.any(indp):
            s = min(s, min(-lamu1[indp] / dlamu1[indp]))
        if np.any(indn):
            s = min(s, min(-lamu2[indn] / dlamu2[indn]))
        indp = np.greater(dx - du, 0.0)
        indn = np.greater(-dx - du, 0.0)
        if np.any(indp):
            s = min(s, min(-fu1[indp] / (dx[indp] - du[indp])))
        if np.any(indn):
            s = min(s, min(-fu2[indn] / (-dx[indn] - du[indn])))
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
                np.sqrt(norm(rdp) ** 2 + norm(rcp) ** 2 + norm(rpp) ** 2)
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
        sdg = -(np.inner(fu1, lamu1) + np.inner(fu2, lamu2))
        if sdg < tol:
            return x

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
        resnorm = np.sqrt(norm(rdual) ** 2 + norm(rcent) ** 2 + norm(rpri) ** 2)

    return x
