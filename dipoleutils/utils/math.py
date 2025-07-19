from numpy.typing import NDArray
import numpy as np
import healpy as hp
import functools
import itertools
import operator
import math
from scipy.special import binom

def uniform_to_uniform_transform(
        uniform_deviates: NDArray[np.float64],
        minimum: float,
        maximum: float 
) -> NDArray[np.float64]:
    '''
    Transform uniform deviates on [0, 1] to uniform deviates on
    [minimum, maximum].

    :param uniform_deviates: Array of uniform deviates, of shape (n_deviates,).
    :param minimum: Minimum of the target distribution.
    :param maximum: Maximum of the target distribution.
    :return: Transformed deviates, shape (n_deviates,).
    '''
    return (minimum - maximum) * uniform_deviates + maximum

def uniform_to_polar_transform(
        uniform_deviates: NDArray[np.float64],
        minimum: float = 0.,
        maximum: float = np.pi
    ) -> NDArray[np.float64]:
    '''
    In spherical coordinates, we need to account for the area element
    changing with polar angle when sampling uniformly.
    This function allows one to transform a uniform deviate on [0,1]
    to a 'polar' distribution on [minimum, maximum] where this is taken care of.
    - theta ~ acos(u * (cos maximum - cos minimum) + cos minimum)
    - theta in [minimum, maximum], subdomain of [0, pi]

    :param uniform_deviates: Uniform deviate between 0 and 1.
    :param minimum: Minimum value to bound polar angles between.
    :param maximum: Maximum value to bound polar angles between.
    :return: uniform deviate on latitudinal (polar) angles
    '''
    unif_theta = np.arccos(
        np.cos(minimum) + uniform_deviates * (np.cos(maximum) - np.cos(minimum))
    )
    return unif_theta

def compute_dipole_signal(
        dipole_amplitude: NDArray[np.float64],
        dipole_longitude: NDArray[np.float64],
        dipole_colatitude: NDArray[np.float64],
        pixel_vectors: NDArray[np.float64]
) -> NDArray[np.float64]:
        '''
        For a vectorised call of the dipole model, compute the term D cos(theta),
        where theta is the angle between the dipole vector and a given pixel.
        In other words, compute the pure l=1 spherical harmonic.

        :param dipole_amplitude: Vector of dipole amplitudes, shape (n_live,).
        :param dipole_longitude: Vector of dipole longitudes, shape (n_live,).
        :param dipole_colatitude: Vector of dipole colatitudes, shape (n_live,).
        :param pixel_vectors: Matrix of pixel vectors, of shape (n_pix, 3).
        :return: Dipole spherical harmonic, of shape (n_pix, n_live).
        '''
        dipole_vector = (
              dipole_amplitude[:, None]
            * hp.ang2vec(dipole_colatitude, dipole_longitude)
        )
        dipole_signal = np.einsum('ki,ji->jk', dipole_vector, pixel_vectors)
        return dipole_signal

def vectorised_quadrupole_tensor(
        amplitude_like: NDArray[np.float64],
        cartesian_quadrupole_vectors: NDArray[np.float64]
    ) -> NDArray[np.float64]:
    '''
    :param amplitude_like: Vector of amplitudes of shape (n_live,).
    :param cartesian_quadrupole_vectors: (3, n_live, 2) matrix, containing the
        two unit vectors (along axis 0 and 2) for each n_live sample (along axis 1).
    :return: Rank-3 tensor of shape (n_live, 3, 3), containing the quadrupole
        tensor for each live point.
    '''
    n_samps = cartesian_quadrupole_vectors.shape[1]
    Q_dash = np.einsum(
        'a...,b...',
        cartesian_quadrupole_vectors[:,:,0],
        cartesian_quadrupole_vectors[:,:,1]
    )
    Q = np.zeros_like(Q_dash)

    for i in range(n_samps):
        Q_star = symmetrise_nxn(Q_dash[i,:,:])
        Q[i] = amplitude_like[i] * make_3x3_traceless(Q_star)
    return Q

def symmetrise_nxn(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    '''
    Symmetrises square nxn matrix.

    :param matrix: Square matrix.
    :return: Symmetrised matrix.
    '''
    return (matrix + matrix.T) / 2

def make_3x3_traceless(matrix: NDArray) -> NDArray[np.float64]:
    '''
    Makes a 3x3 matrix traceless.

    :param matrix: 3x3 matrix.
    :return: Traceless 3x3 matrix.
    '''
    return matrix - (np.trace(matrix) / 3) * np.eye(3)

def vectorised_spherical_to_cartesian(phi_like, theta_like):
    '''
    :param phi_like: Matrix of azimuthal-like angles, shape (n_live, ell).
    :param theta_like: Matrix of polar-like angles, shape (n_live, ell).
    :return: (3, n_live, l) matrix corresponding to the Cartesian parametrisation
        of the input spherical coordinates, with (x,y,z) lying in the first
        dimension.
    '''
    X = np.sin(theta_like) * np.cos(phi_like)
    Y = np.sin(theta_like) * np.sin(phi_like)
    Z = np.cos(theta_like)
    return np.asarray([X, Y, Z])

def multipole_tensor_vectorised(
        amplitude_like: NDArray[np.float64],
        cartesian_multipole_vectors: NDArray[np.float64]
    ) -> NDArray[np.float64]:
    '''
    :param amplitude_like: Vector of amplitudes as long as n_live.
    :param cartesian_multipole_vectors: (3, n_live, ell) tensor, where axis 0
        gives the Cartesian coordinates, axis 1 the nth live point sample and
        axis 2 the i-th unit vector of ell unit vectors.
    :return: Rank-(ell+1) tensor, where the first axis gives the multipole
        tensor for each sample.
    '''
    n_live = cartesian_multipole_vectors.shape[1]
    M_dash = vectorised_outer_product(cartesian_multipole_vectors)
    M = np.zeros_like(M_dash)
    for i in range(n_live):
        M[i] = amplitude_like[i] * (
            make_rankn_traceless(
                symmetrise_tensor(M_dash[i])
            )
        )
    return M

def vectorised_outer_product(
        cartesian_multipole_vectors: NDArray[np.float64]
    ) -> NDArray[np.float64]:
    '''
    :param cartesian_multipole_vectors: (3, n_live, ell) tensor, where axis 0
        gives the Cartesian coordinates, axis 1 the nth live point sample and
        axis 2 the i-th unit vector of ell unit vectors.
    :return: Rank-(ell+1) tensor, where the first axis gives the multipole
        tensor for each sample. Note that this tensor has not been made
        traceless and symmetric, and is simply computed from the outer product
        of the multipole unit vectors.
    '''
    order = cartesian_multipole_vectors.shape[-1]
    einstring = make_vectorised_outer_einstring(order)
    tensor_samps = np.einsum(
        einstring, *[cartesian_multipole_vectors[:,:,i] for i in range(order)]
    )
    return tensor_samps

def make_vectorised_outer_einstring(ell: int) -> str:
    '''
    Make a string, according to Einstein summation convention and numpy's
    `einsum` function, to compute the vectorised outer product of l vectors.
    Example output:
    >>> make_new_einstring(4)
    'a...,b...,c...,d...'

    :param ell: Rank of the desired output tensor.
    :return: Einsum string for computing the outer product of l vectors
        in the vectorized implementation, i.e. where many live points are
        considered at once.
    '''
    einstring = ''
    idxs = np.arange(1, ell+1)
    for i in idxs:
        letter = chr(ord('`')+i)
        einstring += (letter + '...,')
    return einstring[:-1]

def make_vectorised_signal_einstring(ell: int) -> str:
    '''
    Make a string, according to Einstein summation convention and numpy's
    `einsum` function, to compute the inner product between a multipole tensor
    and ell vectors, where ell is the rank of the tensor.
    Example usage:
    >>> make_vecsig_einstring(4)
    'abcde,b...,c...,d...,e...'
    
    :param ell: Rank of the tensor.
    :return: Einsum string for computing the signal from the tensor in the
        vectorized implementation, i.e. where many live points are considered
        at once.
    '''
    einstring_left_comma = ''
    einstring_right_comma = ''
    idxs = np.arange(1, ell+2)

    for i in idxs:
        letter = chr(ord('`')+i)
        einstring_left_comma += letter
        if i != 1:
            einstring_right_comma += (letter + '...,')

    einstring = einstring_left_comma + ',' + einstring_right_comma[:-1]
    return einstring

def symmetrise_tensor(
        tensor: NDArray[np.float64],
    ) -> NDArray[np.float64]:
    '''
    Make a tensor symmetric by summing all permutations of a tensor's indices
    and dividing by the number of permutations. 
    Adapted to use numpy and not torch, from:
    https://stackoverflow.com/questions/72380459/symmetric-random-tensor-with-high-dimension-numpy-pytorch

    :param tensor: Tensor of arbitrary rank to make symetric.
    :return: Symmetrised tensor.
    '''
    dims = len(tensor.shape)
    permutations = itertools.permutations(range(dims))
    n_permutations = math.factorial(dims)
    sym_tensor = functools.reduce(
        operator.add,
        (
            np.transpose(tensor, permutation)
            for permutation in permutations
        ),
    )
    sym_tensor /= n_permutations
    return sym_tensor

def make_rankn_traceless(sym_tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    '''
    Make a rank-ell tensor traceless. 
    The expression is from
    https://math.stackexchange.com/questions/4925863/finding-general-expression-for-symmetric-trace-free-tensors-stfs
    
    :param sym_tensor: Symmetric tensor or arbitrary rank.
    :return: Yraceless component of the tensor, i.e. a symmetric traceless tensor.
    '''
    l = len(sym_tensor.shape)
    range_list = 3 * np.ones(l, dtype='int')
    index_permutations = list(itertools.product(*map(range, range_list)))
    M_hat = np.zeros_like(sym_tensor)

    # compute all contracted tensors, then apply Kronecker deltas
    # to construct tensor T, then symmetrize tensor
    n_contractions = l // 2
    for k in range(0, n_contractions+1):
        ein_str = contraction_einstring(l, k)
        contracted_tensor = np.einsum(ein_str, sym_tensor)
        n_kroneckers = k
        kroneckers = [np.eye(3)]*k
        kron_einstring = kronecker_einstring(l, n_kroneckers)

        if len(kroneckers) == 0:
            T = np.einsum(kron_einstring, contracted_tensor)
        elif n_kroneckers == l / 2: # contracted tensor is rank-0 (scalar)
            assert type(contracted_tensor) == np.float64
            T = np.einsum(kron_einstring, *kroneckers) * contracted_tensor
        else:
            T = np.einsum(kron_einstring, *kroneckers, contracted_tensor)

        T_sym = symmetrise_tensor(T)
        coefficient = (-1)**k * binom(l, k) * binom(l, 2*k) / binom(2*l, 2*k)

        # (0,1,0,2,...) representing (i,j,k,l,...)
        kth_contribution = np.zeros_like(sym_tensor)
        for idxs in index_permutations:
            kth_contribution[idxs] = coefficient * T_sym[idxs]

        M_hat += kth_contribution
    
    return M_hat

def contraction_einstring(rank: int, n_contractions: int) -> str:
    '''
    For a symmetric rank-ell tensor, construct the einsum string such
    that n_contractions are performed. By default, the contracted indices
    start from the right. For example,
    
    >>> make_einstring(5, 2)
    'abbdd'

    Of course, this does not matter if the tensor is symmetric, since the
    tensor is invariant under any permutation of the indices.
    
    The alphabetical order of the string seems to matter if -> is not specified.

    :param rank: Rank of the tensor.
    :param n_contractions: Number of contractions to perform.
    :raises IndexError: If n_contractions > rank // 2.
    '''
    einsum_str = ''
    idxs = np.arange(1, rank+1)
    for k in range(1, n_contractions+1):
        if k == 1:
            idxs[-2*k:] = idxs[-2*k]
        else:
            idxs[-2*k:-2*k+2] = idxs[-2*k]

    for i in idxs:
        letter = chr(ord('`')+i)
        einsum_str += letter

    return einsum_str

def kronecker_einstring(rank: int, n_kroneckers: int) -> str:
    '''
    This is one of the components in making a rank-ell tensor traceless.
    Refer to equation (22) in Oayda+25. In the sum, if for example k = 1 and
    ell = 4, then we would need the expression delta_{ (ij } T_{ kl)mm }. This
    function constructs the symmetrised part of the expression. Explicitly:

    >>> make_kronecker_einstring(4, 1)
    'ab,cd'
    
    :param rank: Rank of the tensor being made traceless.
    :param n_kroneckers: Number of Kronecker deltas to insert.
    :return: Einsum string with Kronecker deltas.
    :raises AssertionError: If 2*n_kroneckers > rank.
    '''
    einstring = ''
    idxs = np.arange(1, rank+1)
    assert not (2*n_kroneckers > rank), 'Too many Kronecker deltas.'

    for i in idxs:
        letter = chr(ord('`')+i)
        einstring += letter
        if (i % 2 == 0) and (i / (2*n_kroneckers) <= 1):
            einstring += ','

    if 2 * n_kroneckers == rank:
        einstring = einstring[:-1]

    return einstring

def multipole_pixel_product_vectorised(
        multipole_tensors: NDArray[np.float64],
        pixel_vectors: list[NDArray[np.float64]],
        ell: int
    ) -> NDArray[np.float64]:
    '''
    Compute the vectorised inner product between a rank-ell multipole tensor
    and ell pixel vectors. 

    :param multipole_tensors_vectorised: Rank-(ell+1) tensor, where the first
        axis gives the tensor for each n_live sample. For example, for a
        quadrupole (rank-[2+1]), the slice [0, :, :] would be the quadrupole
        matrix for the first live point/parameter sample.
    :param pixel_vectors: List of 3 Cartesian coordinates xyz, i.e. the pixel
        vectors, where axis 0 gives the coordinate and axis 1 the pixel.
    :param ell: Order of the multipole.
    '''
    xyzs = [pixel_vectors] * ell
    einstring = make_vectorised_signal_einstring(ell)
    return np.einsum(
        einstring,
        multipole_tensors,
        *xyzs,
        optimize='optimal'
    )

def sigma_to_prob2D(sigma: list[float]) -> NDArray[np.float64]:
    '''
    Convert sigma significance to mass enclosed inside a 2D normal
    distribution using the explicit formula for a 2D normal.
    
    :param sigma: The levels of significance, e.g. `sigma = [1., 2.]`.
    :returns: The probability enclosed within each significance level.
    '''
    return 1.0 - np.exp(-0.5 * np.asarray(sigma)**2)