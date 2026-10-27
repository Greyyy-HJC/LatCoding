"""Uncompressed Wilson-clover time-slice matching auxiliaries.

These are measurements of the actual mass-normalized Dirac inverse and local
operators, not an asserted canonical or hole covariance. No production caller
is modified. Spatial insertions act on fields before any momentum projection.
"""
import numpy as np


def backend(a):
    if type(a).__module__.split('.')[0] == 'cupy':
        import cupy
        return cupy
    return np


def host(a):
    return a.get() if hasattr(a, 'get') else np.asarray(a)


def phase(size, q, xp=np):
    z, y, x = xp.indices(tuple(size[:3])[::-1])
    return xp.exp(2j*xp.pi*(q[0]*x/size[0]+q[1]*y/size[1]+q[2]*z/size[2]))


def project(field, positive_phase):
    xp = backend(field)
    return xp.einsum('zyx,zyxsc->sc', positive_phase, field)/positive_phase.size


def hops(links, field):
    xp = backend(field)
    for j, axis in enumerate((2, 1, 0)):
        u = links[j]
        forward = xp.einsum('...ab,...sb->...sa', u, xp.roll(field, -1, axis=axis))
        backward = xp.einsum('...ba,...sb->...sa', xp.roll(u, 1, axis=axis).conj(),
                             xp.roll(field, 1, axis=axis))
        yield forward, backward


def bw(links, field, mass):
    out = (mass+4)*field
    for f, b in hops(links, field): out = out-.5*(f+b)
    return out


def spatial_gamma(links, field, gammas):
    xp = backend(field)
    out = xp.zeros_like(field)
    for j, (f, b) in enumerate(hops(links, field)):
        out += .5*xp.einsum('ij,...jc->...ic', gammas[j], f-b)
    return out


def local_apply(matrix, field):
    xp = backend(field)
    return xp.einsum('...ij,...j->...i', matrix, field.reshape(*field.shape[:3], 12)).reshape(field.shape)


def split_clover(matrix, gamma4):
    xp = backend(matrix)
    g = xp.kron(gamma4, xp.eye(3))
    conjugated = xp.einsum('ij,...jk,kl->...il', g, matrix, g)
    return .5*(matrix+conjugated), .5*(matrix-conjugated)


def cg_solve(operator, rhs, tol=2e-10, maxiter=2000):
    """Positive spatial operator only; reject breakdown and check true residual."""
    xp = backend(rhs)
    x = xp.zeros_like(rhs); r = rhs.copy(); p = r.copy()
    norm = float(xp.vdot(rhs, rhs).real)
    if not norm: return x, 0., 0
    rr = norm
    for iteration in range(1, maxiter+1):
        ap = operator(p); pap = complex(xp.vdot(p, ap))
        if pap.real <= 0 or abs(pap.imag) > 1e-10*abs(pap):
            raise ValueError('Spatial CG operator is not positive Hermitian on this direction')
        alpha = rr/pap.real; x += alpha*p; r -= alpha*ap
        new = float(xp.vdot(r, r).real)
        if np.sqrt(new/norm) <= tol:
            residual = np.sqrt(float(xp.vdot(rhs-operator(x), rhs-operator(x)).real)/norm)
            if residual > 5*tol: raise ValueError('Spatial solve true residual failed')
            return x, residual, iteration
        p = r+(new/rr)*p; rr = new
    raise ValueError('Spatial CG did not converge')


def boundary_signs(t, lt):
    return (-1 if t == lt-1 else 1, -1 if t == 0 else 1)


def transport(links_t, link_previous, plus, minus, signs):
    xp = backend(plus)
    return (signs[0]*xp.einsum('...ab,...sb->...sa', links_t[3], plus),
            signs[1]*xp.einsum('...ba,...sb->...sa', link_previous.conj(), minus))


def eo_slice_to_lex(data, local_t):
    """One checkerboard slice, eo,z,y,xh,... -> z,y,x,... ."""
    a = host(data); nz, ny, nxh = a.shape[1:4]
    out = np.empty((nz, ny, 2*nxh)+a.shape[4:], a.dtype)
    for z in range(nz):
        for y in range(ny):
            for eo in range(2):
                out[z,y,(eo-local_t-z-y)%2::2] = a[eo,z,y]
    return out


def lex_to_eo_slice(data, local_t):
    a = np.asarray(data); nz, ny, nx = a.shape[:3]
    out = np.empty((2,nz,ny,nx//2)+a.shape[3:], a.dtype)
    for z in range(nz):
        for y in range(ny):
            for eo in range(2): out[eo,z,y] = a[z,y,(eo-local_t-z-y)%2::2]
    return out


def gather_slice(info, data, t, *, gauge=False):
    local_t = t-info.gt*info.Lt
    if 0 <= local_t < info.Lt:
        a = data[:, :, local_t] if gauge else data[:, local_t]
        if gauge: a = a.transpose(1,2,3,4,0,5,6)
        item = (tuple(info.grid_coord), eo_slice_to_lex(a, local_t))
    else: item = None
    parts = info.mpi_comm.gather(item, root=0)
    if info.mpi_rank: return None
    sample = next(a for a in parts if a is not None)[1]
    out = np.empty(tuple(info.global_size[:3])[::-1]+sample.shape[3:], sample.dtype)
    for item in parts:
        if item is None: continue
        (gx,gy,gz,gt),a = item; nz,ny,nx = a.shape[:3]
        out[gz*nz:(gz+1)*nz,gy*ny:(gy+1)*ny,gx*nx:(gx+1)*nx] = a
    return out.transpose(3,0,1,2,4,5) if gauge else out


def scatter_rhs(info, array, t):
    """Insert an actual spatial operator applied to a wall into the 4D RHS."""
    from pyquda.field import LatticeFermion
    coords = info.mpi_comm.gather(tuple(info.grid_coord), root=0)
    parts = None
    if info.mpi_rank == 0:
        a = host(array); nx,ny,nz,nt = info.size
        parts = [lex_to_eo_slice(a[gz*nz:(gz+1)*nz,gy*ny:(gy+1)*ny,gx*nx:(gx+1)*nx],
                                t-gt*nt) if gt*nt <= t < (gt+1)*nt else None
                 for gx,gy,gz,gt in coords]
    part = info.mpi_comm.scatter(parts, root=0)
    rhs = LatticeFermion(info)
    if part is not None: rhs.data[:,t-info.gt*info.Lt] = backend(rhs.data).asarray(part)
    return rhs


def actual_clover_slice(dirac, t, links, gammas):
    """Extract actual mass-form C via MatQuda, without another inversion.

    The installed dynamic/reconstructed-clover build does not support a direct
    CloverQuda application. MatQuda on a time-local field instead gives D_s at
    this slice. Subtract the independently implemented B_W+K_space action.
    """
    from pyquda.field import LatticeFermion
    info = dirac.latt_info; matrix = None
    if info.mpi_rank == 0: matrix = np.empty(tuple(info.global_size[:3])[::-1]+(12,12),complex)
    local_t = t-info.gt*info.Lt
    for col in range(12):
        rhs = LatticeFermion(info)
        if 0 <= local_t < info.Lt: rhs.data[:,local_t,...,col//3,col%3] = 1
        out = dirac.mat(rhs)
        a = gather_slice(info,out.data,t)
        if matrix is not None:
            xp = backend(links)
            unit = xp.zeros(tuple(info.global_size[:3])[::-1]+(4,3),complex)
            unit[...,col//3,col%3] = 1
            standard = bw(links,unit,dirac.invert_param.mass)+spatial_gamma(links,unit,gammas)
            matrix[...,col] = (a-host(standard)).reshape(*a.shape[:3],12)
        del rhs,out
    return matrix


def free_inverse(q, size, mass, gammas, delta=0):
    """Exact finite-T antiperiodic Wilson Fourier sum in mass normalization."""
    spatial = -2*np.pi*np.asarray(q)/np.asarray(size[:3])
    b = mass+4-np.cos(spatial).sum()
    out = np.zeros((4,4),complex)
    for omega in (2*np.arange(size[3])+1)*np.pi/size[3]:
        d = (b-np.cos(omega))*np.eye(4)+1j*np.sin(omega)*gammas[3]
        d += sum(1j*np.sin(spatial[j])*gammas[j] for j in range(3))
        out += np.exp(1j*omega*delta)*np.linalg.inv(d)/size[3]
    return np.kron(out,np.eye(3))


def wilson_baseline(s0, b_inverse, gamma4):
    """Wilson coherent-state baseline only; explicitly uncertified for clover.

    For mass-form Wilson and specified ordering: A=B_W^{-1},
    N=B_W^{-1} P_+ - S_0 gamma4. The clover caller must not promote this
    diagnostic into a certified covariance without an independent derivation.
    """
    g = np.kron(gamma4,np.eye(3)); p = .5*(np.eye(12)+g)
    return b_inverse, b_inverse@p-s0@g
