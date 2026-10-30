"""Module optikon for finding optimal conjunction of propositions about numerical data.

(c) Mario Boley
"""

import numpy as np

from numba import njit, types
from numba.experimental import jitclass
from numba.types import int64, float64, intp
from numba.typed import List
import numba.types as numbatypes
# from numba.types import unicode_type

### Utility ###
###############

@njit
def argsort_columns(x):
    n, p = x.shape
    out = np.empty((n, p), dtype=np.int64)
    for j in range(p):
        out[:, j] = np.argsort(x[:, j])
    return out

@njit
def sort_columns(x):
    _, p = x.shape
    out = np.empty_like(x)
    for j in range(p):
        out[:, j] = np.sort(x[:, j])
    return out

@njit
def compute_bounds(x):
    """
    Compute per-variable bounds over a dataset.

    Args:
        x (ndarray): Data matrix of shape (n, d), where each row is a sample 
        and each column is a variable.

    Returns:
        Tuple[ndarray, ndarray]: A pair (l, u) of arrays, each of shape (d,), where 
        l[j] is the minimum and u[j] is the maximum of variable j over all n samples.

    Notes:
        If the input has zero rows (n == 0), the returned bounds are (inf, -inf) 
        for each variable. This convention ensures that all propositions are 
        treated as trivially satisfied on the empty domain.

    Examples:
        >>> x = np.array([[1.0, -1.0], [0.0, 0.0]])
        >>> compute_bounds(x)
        (array([ 0., -1.]), array([1., 0.]))

        >>> x = np.empty((0, 2))
        >>> compute_bounds(x)
        (array([inf, inf]), array([-inf, -inf]))
    """
    n, d = x.shape
    l = np.full(d, np.inf)
    u = np.full(d, -np.inf)
    for i in range(n):
        for j in range(d): # TODO: benchmark loop inversion with parallelisation
            if x[i, j] < l[j]:
                l[j] = x[i, j]
            if x[i, j] > u[j]:
                u[j] = x[i, j]
    return l, u

@njit
def num_finite_bounds(l, u):
    res = 0
    for j in range(len(l)):
        if l[j] > -np.inf:
            res += 1
        if u[j] < np.inf:
            res += 1
    return res

sort_columns.compile("(float64[:, :],)")
argsort_columns.compile("(float64[:, :],)")
compute_bounds.compile("(float64[:, :],)")
num_finite_bounds.compile('(float64[:], float64[:])')

def make_maxheap_class(KeyType, NodeType):
    """Create a max-heap jitclass specialized for the given KeyType and NodeType.
    
    Heaps store data in a list of (key, node) tuples. The heap class is force-compiled before return
    for transparent performance tests.
    """

    PairType = numbatypes.Tuple((KeyType, NodeType))
    heap_spec = [
        ('data', numbatypes.ListType(PairType))
    ]

    @jitclass(heap_spec)
    class Heap:
        def __init__(self):
            self.data = List.empty_list(PairType)

        def __bool__(self):
            return len(self.data) > 0

        def push(self, key, node):
            self.data.append((key, node))
            i = len(self.data) - 1
            while i > 0:
                parent = (i - 1) // 2
                if self.data[i][0] <= self.data[parent][0]:
                    break
                self.data[i], self.data[parent] = self.data[parent], self.data[i]
                i = parent

        def pop(self):
            if len(self.data) == 0:
                raise IndexError('pop from empty heap')
            top = self.data[0]
            last = self.data.pop()
            if len(self.data) == 0:
                return top
            self.data[0] = last
            i = 0
            while True:
                left = 2 * i + 1
                right = 2 * i + 2
                largest = i
                if left < len(self.data) and self.data[left][0] > self.data[largest][0]:
                    largest = left
                if right < len(self.data) and self.data[right][0] > self.data[largest][0]:
                    largest = right
                if largest == i:
                    break
                self.data[i], self.data[largest] = self.data[largest], self.data[i]
                i = largest
            return top

    _ = Heap()  # force compile
    return Heap

##### Propositionalisation #####
###############################

@jitclass
class Propositionalization:
    """
    Array of simple threshold propositions over a d-dimensional dataset.

    Each of the p propositions represents an inequality s*x_v >= t defined by:
      - a variable index v in {0, ..., d-1},
      - a float64 threshold t,
      - and a sign s in {-1, 1} indicating the direction of comparison.

    That is, for s=-1, the represented proposition can be read as an upper
    bound x_v <= -t .

    Propositionalizations can be used to represent conjunctions 
    (see `suppport_all` and `as_conj_str`), disjunctions, or some base 
    collection of propositions, from which to find optimal subsets.

    For the latter, class supports NumPy-style indexing to extract subsets
    of propositions, including slicing, integer arrays, and boolean masks
    (see `__getitem__`).
    
    For example:
        `p[:2]` returns the first two propositions.
        `p[p.s == -1]` selects all propositions with negative sign.
    
    Note that even a single index like `p[0]` returns a Propositionalization
    with one proposition, not a scalar.

    Parameters
    ----------
    v : int64[:]
        Array of variable indices for each proposition.
    t : float64[:]
        Array of thresholds for each proposition.
    s : int64[:]
        Array of signs (-1 or 1) for each proposition.
    """

    v: int64[:]
    t: float64[:]
    s: int64[:]

    def __init__(self, v, t, s):
        self.v = v
        self.t = t
        self.s = s

    def support_specific(self, x, p):
        return np.flatnonzero(self.s[p]*x[:,self.v[p]] >= self.t[p])
    
    def support_all(self, x, q=None):
        """Returns indices of samples satisfying all propositions in q.

        If q is None, all propositions are used (i.e. the entire propositionalisation).

        Args:
            x (ndarray): Input data of shape (n, d).
            q (ndarray or None): Indices of propositions. If None, uses all.

        Returns:
            ndarray: 1D array of indices where all selected propositions hold.
        """
        if q is None: q = np.arange(len(self))

        if len(q)==0: return np.arange(len(x))
        
        res = np.flatnonzero(self.s[q[0]]*x[:, self.v[q[0]]] >= self.t[q[0]])
        for i in range(1, len(q)):
            res = res[np.flatnonzero(self.s[q[i]]*x[res, self.v[q[i]]] >= self.t[q[i]])]
        return res
    
    def trivial(self, l, u, subset):
        """
        Identify trivial (tautological) propositions over the given variable bounds.

        Args:
            l (ndarray): Lower bounds for each of the d variables (shape: [d]).
            u (ndarray): Upper bounds for each of the d variables (shape: [d]).
            subset (ndarray): Indices of the propositions to check (shape: [m], values in [0, p)).

        Returns:
            ndarray: Indices in `subset` of propositions that are tautological 
            (i.e., always satisfied given the bounds).

        Examples:
            >>> from opticon import Propositionalization
            >>> import numpy as np
            >>> prop = Propositionalization(np.array([0, 1]), np.array([0.5, -1.0]), np.array([1, -1]))
            >>> l = np.array([0.0, -2.0])
            >>> u = np.array([1.0, 0.0])
            >>> prop.tautologies(l, u, np.array([0, 1]))
            array([1])
        """
        v = self.v[subset]
        t = self.t[subset]
        s = self.s[subset]

        res = np.zeros(len(subset), dtype=np.bool_)

        lower = s == 1
        upper = s == -1

        res[lower] = l[v[lower]] >= t[lower]
        res[upper] = -u[v[upper]] >= t[upper]

        return subset[res] #return np.flatnonzero(res)

    def nontrivial(self, l, u, subset):
        """
        Identify propositions that are not tautological over the given variable bounds.

        Args:
            l (ndarray): Lower bounds for each of the d variables (shape: [d]).
            u (ndarray): Upper bounds for each of the d variables (shape: [d]).
            subset (ndarray): Indices of the propositions to check (shape: [m], values in {0, ..., p-1}).

        Returns:
            ndarray: Indices in `subset` of propositions that are not tautological
            (i.e., not always satisfied under the given bounds).

        Examples:
            >>> from opticon import Propositionalization
            >>> import numpy as np
            >>> prop = Propositionalization(np.array([0, 1]), np.array([0.5, -1.0]), np.array([1, -1]))
            >>> l = np.array([0.0, -2.0])
            >>> u = np.array([1.0, 0.0])
            >>> nontrivial(prop, l, u, np.array([0, 1]))
            array([0])
        """
        v = self.v[subset]
        t = self.t[subset]
        s = self.s[subset]

        res = np.zeros(len(subset), dtype=np.bool_)

        lower = s == 1
        upper = s == -1

        res[lower] = l[v[lower]] < t[lower]
        res[upper] = -u[v[upper]] < t[upper]

        return subset[res]# np.flatnonzero(res) 
    
    def binarize(self, x):
        """
        Binarizes a dataset based on the propositionalisation.

        Args:
            x (ndarray): Data matrix of shape (n, d), where each row is a sample.

        Returns:
            ndarray: Binary matrix of shape (n, p), where entry (i, j) is 1 if 
            the j-th proposition is satisfied by the i-th sample, and 0 otherwise.
        """        
        return self.s*x[:, self.v] >= self.t
    
    def __getitem__(self, idxs):
        """
        Return a new Propositionalization with a subset of propositions.

        Parameters
        ----------
        idxs : int, slice, array-like of int, or boolean array
            Indices selecting the propositions to keep. Supports all 
            standard NumPy indexing modes.

        Returns
        -------
        Propositionalization
            A new object containing only the selected propositions.

        Notes
        -----
        Even a single integer index (e.g., `p[0]`) returns a new
        Propositionalization with one element, not a scalar proposition.

        Examples
        --------
        >>> p[:3]          # first three propositions
        >>> p[[0, 2, 4]]   # specific indices
        >>> p[p.s == -1]   # boolean mask
        >>> p[0]           # propositionalization with first proposition
        """
        return Propositionalization(self.v[idxs], self.t[idxs], self.s[idxs])

    def __len__(self):
        """
        Returns the number of propositions (p) in this propositionalization.

        Returns:
            int: Total number of propositions.

        Examples:
            >>> len(prop)
            2
        """
        return len(self.v)

    # not njit compatible but useful as template for external function
    # def str_from_prop(prop, j):
    #     return f'x{prop.v[j]+1} {'>=' if prop.s[j]==1 else '<='} {prop.s[j]*prop.t[j]:0.3f}'

    def str_from_conj(self, q):
        # print('Deprecated method "str_from_conj" will be removed in version 0.3; use "prop[q].as_conj_str()" instead', flush=True)
        result = ''
        for i in range(len(q)):
            if i > 0:
                result += ' & '
            result += self.str_from_prop(q[i])
        return result

    def str_from_prop(self, j, dec=3):
        """
        Returns a string representation of the j-th proposition with basic float formatting
        using fixed number of decimal digit.

        Args:
            j (int): Index of the proposition.
            dec (int): Number of decimal digits to show. Must be >= 0.

        Returns:
            str: Formatted string of the proposition in format: "x{v+1} >= int.frac" or "x{v+1} <= int.frac".

        Note:
            Negative values for `dec` are currently not supported. The whole method is a workaround to deal
            with current numba limitations.
        """
        name_str = 'x' + str(self.v[j] + 1)
        rel_str = '>=' if self.s[j] == 1 else '<='
        value = self.s[j] * self.t[j]

        scale = 10 ** dec
        rounded = int(value * scale + 0.5 * (1 if value >= 0 else -1)) / scale

        if dec <= 0:
            val_str = str(int(rounded))
        else:
            int_part = int(rounded)
            frac_part = int(abs(rounded - int_part) * (10 ** dec) + 0.5)
            pos_int_part = abs(int_part)
            sign_str = '' if value >= 0 else '-'
            int_str = str(pos_int_part)
            frac_str = str(frac_part).rjust(dec, '0')
            val_str = sign_str + int_str + '.' + frac_str

        return name_str + ' ' + rel_str + ' ' + val_str
    
    def as_str(self, start='[', end=']', sep=', ', dec=3):
        parts = List.empty_list(numbatypes.unicode_type)
        parts.append(start)
        for i in range(len(self)):
            if i > 0:
                parts.append(sep)
            parts.append(self.str_from_prop(i, dec))
        parts.append(end)
        return ''.join(parts)
    
    def as_conj_str(self, dec=3):
        return self.as_str('', '', ' & ', dec)
    
    def as_disj_str(self, dec=3):
        return self.as_str('', '', ' | ', dec)

    def __str__(self):
        return self.as_str()

@njit
def full_propositionalization(x):
    """
    Constructs propositionalization with all non-trivial threshold propositions from x.
     
    Args:
        x (ndarray): An (n, d) array 

    Returns:
        Propositionalization with variable indices, signs, and threshold pointers v, s, and t
        ordered lexicographically with respect to (v, -s, -t) implying logically stronger
        propositions to have a smaller index than logically weaker propositions on the same
        variabe
    """
    n, d = x.shape
    max_props = 2 * n * d
    v_out = np.empty(max_props, dtype=np.int64)
    t_out = np.empty(max_props, dtype=np.float64)
    s_out = np.empty(max_props, dtype=np.int64)
    count = 0

    for v in range(d):
        thresholds = np.unique(x[:, v])
        # lower bounds strictest to weakest, exluding trivial 
        for t in thresholds[-1:0:-1]:  
            v_out[count] = v
            t_out[count] = t
            s_out[count] = 1
            count += 1
        # upper bounds: strictest to weeakest, excluding trivial
        for t in thresholds[:-1]:  
            v_out[count] = v
            t_out[count] = -t
            s_out[count] = -1
            count += 1

    return Propositionalization(v_out[:count], t_out[:count], s_out[:count])

@njit
def propositionalization_from_intervals(l, u):
    k = num_finite_bounds(l, u)
    v = np.zeros(k, dtype=np.int64)
    t = np.zeros(k, dtype=np.float64)
    s = np.zeros(k, dtype=np.int64)

    r = 0        
    for j in range(len(l)):
        if l[j] > -np.inf:
            v[r] = j
            t[r] = l[j]
            s[r] = 1
            r += 1
        if u[j] < np.inf:
            v[r] = j
            t[r] = -u[j]
            s[r] = -1
            r += 1
    return Propositionalization(v, t, s)

def equal_frequency_propositionalization(x, k=None):
    n, d = x.shape
    k = k if k is not None else 2*np.ceil(n**(1/3)).astype(int)
    quantile_targets = np.linspace(0, 1, k + 1)[1:-1]

    quantiles = np.quantile(x, quantile_targets, axis=0)  # shape (n_splitpoints, n_cols)
    v = np.repeat(np.arange(d), quantiles.shape[0])
    t = quantiles.flatten()

    keep = np.empty_like(v, dtype=bool)
    keep[0] = True
    keep[1:] = (v[1:] != v[:-1]) | (t[1:] != t[:-1])
    v, t = v[keep], t[keep]

    s = np.repeat([1, -1], len(v))
    return Propositionalization(np.concatenate((v, v)), np.concatenate((t, -t)), s)

@njit
def equal_width_propositionalization(x):
    """
    Generate propositionalizaton using equal-width binning (according to the Freedman-Diaconis rule
    as also implemented in numpy.histogram_bin_edges(data, bins='fd')).

    Specifically, for each column in the input, this function determines a bin width using the rule:

        width = 2 * IQR / n**(1/3)

    where IQR is the interquartile range of the column (75th percentile - 25th percentile) and then
    provides upper and lower bound proposition for each threshold that separates two bins.

    Args:
        x (ndarray): An (n, d) array 

    Returns:
        Propositionalization with variable indices, signs, and threshold pointers v, s, and t
        ordered lexicographically with respect to (v, -s, -t) implying logically stronger
        propositions to have a smaller index than logically weaker propositions on the same
        variabe 
            
    Example:
        >>> import numpy as np
        >>> x = np.linspace(0, 12, 27).reshape(-1, 1)
        >>> result = equal_width_propositionalization(x)
        >>> len(result) # n**(1/3)=3, ICR=6 results in 12/4 = 3 bins, hence 2 non-trivial thresholds per direction
        4
    """
    return equal_width_propositionalization_sorted(sort_columns(x))

@njit
def equal_width_propositionalization_sorted(x_sorted):
    n, d = x_sorted.shape

    max_possible = 2 * d * n
    v = np.empty(max_possible, dtype=np.int64)
    t = np.empty(max_possible, dtype=np.float64)
    s = np.empty(max_possible, dtype=np.int64)
    idx = 0

    for j in range(d):
        col_data = x_sorted[:, j]
        l_j = col_data[0]
        u_j = col_data[-1]

        if u_j == l_j:
            continue
        q25 = col_data[int(0.25 * (n-1))]
        q75 = col_data[int(0.75 * (n-1))]
        iqr = q75 - q25

        width = 2 * iqr / max(1, n**(1/3))
        if width == 0:
            continue

        n_bins = int(np.ceil((u_j - l_j) / width))
        if n_bins <= 1:
            continue

        edges = l_j + width * np.arange(1, n_bins)

        positions = np.searchsorted(col_data, edges, side='left')
        positions_ext = np.empty(len(positions) + 1, dtype=np.int64)
        positions_ext[0] = 0
        positions_ext[1:] = positions
        diffs = np.diff(positions_ext)
        nontrivial = diffs > 0

        for k in range(len(edges)):
            if nontrivial[k]:
                # upper bound first (s=1, decreasing thresholds)
                v[idx] = j
                t[idx] = edges[len(edges) - 1 - k]
                s[idx] = 1
                idx += 1

        for k in range(len(edges)):
            if nontrivial[k]:
                # lower bound second (s=-1, increasing thresholds)
                v[idx] = j
                t[idx] = -edges[k]
                s[idx] = -1
                idx += 1

    return Propositionalization(v[:idx], t[:idx], s[:idx])

@njit
def empty_propositionalization(x=None):
    return Propositionalization(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64))

@njit
def apx_minsize_conj_descr(x, support):
    n = x.shape[0]
    
    complement_mask = np.ones(n, np.bool_)
    complement_mask[support] = False
    complement = np.flatnonzero(complement_mask)

    l, u = compute_bounds(x[support])
    props = propositionalization_from_intervals(l, u)
    selected = np.zeros(len(props), dtype=np.bool_)

    while len(complement) > 0:
        best_p = -1
        best_new_compl = complement
        for p in range(len(props)):
            if selected[p]:
                continue
            compl_p = complement[props.s[p]*x[complement, props.v[p]] >= props.t[p]]
            if len(compl_p) < len(best_new_compl):
                best_p = p
                best_new_compl = compl_p
        
        if best_p > -1:
            selected[best_p] = True
            complement = best_new_compl
        else:
            break

    return props[np.flatnonzero(selected)]

empty_propositionalization.compile('(float64[:, :],)')
full_propositionalization.compile('(float64[:, :],)')
equal_width_propositionalization.compile('(float64[:, :],)')
equal_width_propositionalization_sorted.compile('(float64[:, :],)')

##### Fast Interval Pattern Search   #####
##########################################

@jitclass
class IntervalPatternSearchNode:
    l: float64[:]
    u: float64[:]
    support: int64[:]
    pos_support: int64[:]
    min_active_j: int64

    def __init__(self, l, u, support, pos_support, min_active_j):
        self.l = l
        self.u = u
        self.support = support
        self.pos_support = pos_support
        self.min_active_j = min_active_j

    # def num_non_trivial_bounds(self):
    #     res = 0
    #     for j in range(len(self.l)):
    #         if self.l[j] > -np.inf:
    #             res += 1
    #         if self.u[j] < np.inf:
    #             res += 1
    #     return res

    # def to_propositionalization(self):
    #     k = self.num_non_trivial_bounds()
    #     v = np.zeros(k, dtype=np.int64)
    #     t = np.zeros(k, dtype=np.float64)
    #     s = np.zeros(k, dtype=np.int64)

    #     r = 0        
    #     for j in range(len(self.l)):
    #         if self.l[j] > -np.inf:
    #             v[r] = j
    #             t[r] = self.l[j]
    #             s[r] = 1
    #             r += 1
    #         if self.u[j] < np.inf:
    #             v[r] = j
    #             t[r] = -self.u[j]
    #             s[r] = -1
    #             r += 1
    #     return Propositionalization(v, t, s)

@njit
def make_interval_search_root(x, w):
    n, d = x.shape
    l, u = np.full(d, -np.inf), np.full(d, np.inf)
    return IntervalPatternSearchNode(l, u, np.arange(n), np.flatnonzero(w > 0), 0)
    
IntervallPatternNodeHeap = make_maxheap_class(float64, IntervalPatternSearchNode.class_type.instance_type)

@njit
def range_preserving_suffix_and_prefix(x):
    """
    Computes smallest suffix and prefix of x that preserve range (min and max 
    values are retained).

    Args:
        x (ndarray): non-empty array of shape (m,).

    Returns:
        Tuple[int, int]: (i, j) such that i is the largest index such that at least on
        occurrences of each x.min() and x.max() are still present in x[i:], and j is
        the smallest index j such that at least one of those occurrences are
        still present in x[:j+1].

    Note:
        The function runs in O(m)

    Examples:
        >>> range_preserving_suffix_and_prefix(np.array([1., 2., 3., 4., 5.]))
        (0, 4)
        >>> range_preserving_suffix_and_prefix(np.array([2., 2., 2., 2., 2.]))
        (4, 0)
        >>> range_preserving_suffix_and_prefix(np.array([2., 1., 3., 4., 5.]))
        (1, 4)
        >>> range_preserving_suffix_and_prefix(np.array([4., 2., 2., 3., 4.]))
        (2, 1)
    """
    m = x.shape[0]
    z_min = x.min()
    z_max = x.max()
    count_z_min = (x == z_min).sum()
    count_z_max = (x == z_max).sum()

    z_min_remaining = count_z_min
    z_max_remaining = count_z_max
    i = 0
    while i < m and z_min_remaining > 0 and z_max_remaining > 0:
        if x[i] == z_min:
            z_min_remaining -= 1
        if x[i] == z_max:
            z_max_remaining -= 1
        i += 1

    z_min_remaining = count_z_min
    z_max_remaining = count_z_max
    j = m - 1
    while j >= 0 and z_min_remaining > 0 and z_max_remaining > 0:
        if x[j] == z_min:
            z_min_remaining -= 1
        if x[j] == z_max:
            z_max_remaining -= 1
        j -= 1

    return i - 1, j + 1

@njit
def prefix_preserving_index_bounds(x, orders, min_k=0):
    """
    Computes longest non-empty prefix-preserving index ranges for each variable in a dataset.

    Specifically, for each variable k >= min_k, this function computes:
      - the largest index l for restricting the dataset via x[:, k] >= x[orders[l, k]],
        or equivalently to x[orders[l:, k]], and
      - the smallest index u for restricting the dataset via x[:, k] <= x[orders[u, k]]
        or equivalently to x[orders[:u+1, k]]
    such that those restrictions do not reduce the value range of all variables j < k.

    Args:
        x (ndarray): A dataset of shape (m, d).
        orders (ndarray): An array of shape (m, d), where each column contains the 
            indices that would sort x[:, k] in ascending order.
        min_k (int): smallest index for which to compute value ranges

    Returns:
        Tuple[ndarray, ndarray]: Two arrays of shape (d,), where the first contains 
        the maximal prefix-preserving lower-bound indices, and the second contains 
        the minimal prefix-preserving upper-bound indices. Arrays are padded with 
        default values n-1 and 0 for max lower and min upper bounds indices, respectively,

    Notes:
        - The function runs in time O((d-min_k)^2 m) <= O(d^2 m)
        - Default values for k < min_k are n-1 and 0

    Examples:
        >>> import numpy as np
        >>> x = np.array([[0.1, 1.0, -0.5], 
        ...               [0.3, 2.0, -1.0],
        ...               [0.2, 0.5, 0.0]])
        >>> orders = np.argsort(x, axis=0)
        >>> l, u = prefix_preserving_index_bounds(x, orders)
        >>> np.round(l, 2)
        array([2, 1, 0])
        >>> np.round(u, 2)
        array([0, 2, 2])
    """
    n, d = x.shape
    max_pp_lb_indices = np.full(d, n-1, dtype=np.int64)
    min_pp_ub_indices = np.full(d, 0, dtype=np.int64)

    for k in range(min_k, d):
        for j in range(k):
            l, u = range_preserving_suffix_and_prefix(x[orders[:, k], j])
            if l < max_pp_lb_indices[k]:
                max_pp_lb_indices[k] = l
            if u > min_pp_ub_indices[k]:
                min_pp_ub_indices[k] = u

    return max_pp_lb_indices, min_pp_ub_indices

@njit
def max_weighted_support_fips(x, w, max_depth=4):
    _, d = x.shape
    heap = IntervallPatternNodeHeap()
    
    root = make_interval_search_root(x, w)
    root_bound = w[root.pos_support].sum()
    root_value = w.sum()
    heap.push(root_bound, root)

    best_value = root_value
    best_node = root
    nodes_created = 1
    nodes_enqueued = 1

    while heap:
        bound, node = heap.pop()

        if len(node.support) == 0:
            print('warning: zero support dequeued')
            continue

        if bound < best_value:
            continue
        if num_finite_bounds(node.l, node.u) >= max_depth:
            continue

        x_sub = x[node.support]
        sub_orders = argsort_columns(x_sub) # np.argsort(x_sub, axis=0)
        x_sub_pos = x[node.pos_support]
        w_sub = w[node.support]
        w_sub_pos = w[node.pos_support]
        
        max_pp_lb, min_pp_ub = prefix_preserving_index_bounds(x_sub, sub_orders, node.min_active_j)

        for j in range(node.min_active_j, d):

            # should we create view: vals = x_sub[sub_orders[:, j], j]
            # or would this be detremental for performance?

            col_data = x_sub[:, j]
            order = sub_orders[:, j]
            pos_col_data = x_sub_pos[:, j]

            if np.isneginf(node.l[j]):
                
                # create all canonical nodes from lower bounds

                sum_w = w_sub.sum()
                sum_pos_w = w_sub_pos.sum()

                for i in range(1, max_pp_lb[j]+1):
                    t = col_data[order[i]]
                    w_rem = w_sub[order[i-1]]
                    sum_w -= w_rem
                    if w_rem > 0:
                        sum_pos_w -= w_rem

                    if t > col_data[order[i-1]]:
                        _l = node.l.copy()
                        _l[j] = t
                        _sup = node.support[np.flatnonzero(col_data >= t)]
                        _pos_sup = node.pos_support[np.flatnonzero(pos_col_data >= t)]
                        # probably cheaper but changes order: _sup = node.support[order[i:]]
                        
                        child = IntervalPatternSearchNode(_l, node.u, _sup, _pos_sup, j)
                        nodes_created += 1
                        if sum_w > best_value:
                            best_value = sum_w
                            best_node = child
                        if sum_pos_w > best_value:
                            heap.push(sum_pos_w, child)
                            nodes_enqueued += 1
                        else:
                            break
                        
            # create all canonical nodes from upper bounds
            sum_w = w_sub.sum()
            sum_pos_w = w_sub_pos.sum()
            for i in range(len(node.support)-2, min_pp_ub[j]-1, -1):
                t = col_data[order[i]] 
                w_rem = w_sub[order[i+1]]
                sum_w -= w_rem
                if w_rem > 0:
                    sum_pos_w -= w_rem
                if t < col_data[order[i+1]]:
                    _u = node.u.copy()
                    _u[j] = t
                    _sup = node.support[np.flatnonzero(col_data <= t)]
                    _pos_sup = node.pos_support[np.flatnonzero(pos_col_data <= t)]
                    # probably cheaper but changes order: _sup = node.support[order[:i+1]]
                    child = IntervalPatternSearchNode(node.l, _u, _sup, _pos_sup, j+1)
                    nodes_created += 1
                    if sum_w > best_value:
                        best_value = sum_w
                        best_node = child
                    if sum_pos_w > best_value:
                        heap.push(sum_pos_w, child)
                        nodes_enqueued += 1
                    else:
                        break

    return propositionalization_from_intervals(best_node.l, best_node.u), \
                                               best_value, {'nodes_created': nodes_created,
                                                             'nodes_enqueued': nodes_enqueued}

max_weighted_support_fips.compile((types.Array(float64, 2, 'C'), types.Array(float64, 1, 'C'), intp))
max_weighted_support_fips.compile((types.Array(float64, 2, 'F'), types.Array(float64, 1, 'C'), intp))
max_weighted_support_fips.compile((types.Array(float64, 2, 'A'), types.Array(float64, 1, 'A'), intp))

##### Lexicographic Treesearch   #####
######################################

@jitclass
class LexTreeSearchNode:
    
    key: int64[:]
    critical: int64[:]
    remaining: int64[:]
    support: int64[:]
    pos_support: int64[:]

    def __init__(self, key, critical, remaining, support, pos_support):
        self.key = key
        self.critical = critical
        self.remaining = remaining
        self.support = support
        self.pos_support = pos_support

# NODE_TYPE = Node.class_type.instance_type  
NodeHeap = make_maxheap_class(float64, LexTreeSearchNode.class_type.instance_type)

@njit
def make_lex_treesearch_root(x, y, prop):
    l, u = compute_bounds(x)
    remaining = prop.nontrivial(l, u, np.arange(len(prop)))
    empty = np.empty(0, dtype=np.int64)
    support = np.arange(len(x))
    pos_support = support[y > 0]
    return LexTreeSearchNode(empty, empty, remaining, support, pos_support)

@njit
def max_weighted_support_bb(x, y, prop, max_depth=4):
    heap = NodeHeap()

    root = make_lex_treesearch_root(x, y, prop)
    root_bound = y[root.pos_support].sum()
    root_value = y.sum()
    heap.push(root_bound, root)

    best_key = root.key
    best_val = root_value
    nodes_created = 1
    nodes_enqueued = 1
    candidate_edges = 0

    while heap:
        key, node = heap.pop()
        
        if key <= best_val:
            break

        if len(node.key) >= max_depth:
            continue

        candidate_edges += len(node.remaining)
        for p_idx in range(len(node.remaining)):
            p = node.remaining[p_idx]

            _key = np.empty(len(node.key) + 1, dtype=np.int64)
            _key[:-1] = node.key
            _key[-1] = p

            _sup = node.support[prop.support_specific(x[node.support], p)]
            _pos_sup = node.pos_support[prop.support_specific(x[node.pos_support], p)]

            _val = y[_sup].sum()
            _bound = y[_pos_sup].sum()
            nodes_created += 1

            if _val > best_val:
                best_val = _val
                best_key = _key

            if _bound <= best_val:
                continue

            _crit = np.empty(len(node.critical) + p_idx, dtype=np.int64)
            _crit[:len(node.critical)] = node.critical
            _crit[len(node.critical):] = node.remaining[:p_idx]

            l, u = compute_bounds(x[_sup])
            if len(prop.trivial(l, u, _crit)) > 0:
                continue

            _rem = prop.nontrivial(l, u, node.remaining[p_idx+1:])

            heap.push(_bound, LexTreeSearchNode(_key, _crit, _rem, _sup, _pos_sup))

            nodes_enqueued += 1

    return prop[best_key], best_val, {'nodes_created': nodes_created,
                                      'nodes_enqueued': nodes_enqueued,
                                      'candidate_edges': candidate_edges}
    
##### Greedy Search #####
#########################

@jitclass
class WeightedSupport:
    """
    Objective function tracking total weight over a support set.

    Computes the sum of weights for a given support and allows 
    incremental updates (see `remove` and `reset`).

    Parameters
    ----------
    w : float64[:]
        Weight vector over the dataset.

    Attributes
    ----------
    value : float
        Total weight of the current support.
    value_removed : float
        Cumulative weight of removed elements.
    value_remaining : float
        Remaining weight (i.e., value - value_removed).

    Examples
    --------
    >>> w = np.array([1.0, 2.0, 3.0])
    >>> obj = WeightedSupport(w)
    >>> obj.support(np.array([0, 2]))
    >>> obj.value
    4.0

    >>> obj.remove(2)
    >>> obj.value_removed
    3.0
    >>> obj.value_remaining
    1.0

    >>> obj.reset()
    >>> obj.value_remaining
    4.0
    """

    w: float64[:]
    value: float64
    value_removed: float64
    value_remaining: float64

    def __init__(self, w):
        self.w = w.astype('float64')
        self.support(np.arange(len(w)))
        self.reset()

    def support(self, support):
        self.value = np.sum(self.w[support])

    def reset(self):
        self.value_removed = 0
        self.value_remaining = self.value

    def remove(self, i):
        yi = self.w[i]
        self.value_removed += yi
        self.value_remaining -= yi

@jitclass
class NormalizedWeightedSupport:
    """
    Objective function: normalized sum of weights over a support set, i.e.,

        value = sum(w[support]) / (sum(u[support])**(1/norm) + lam)

    where:
      - `w` are sample weights,
      - `u` are normalization weights (defaults to ones),
      - `norm` specifies the p-norm used for normalization,
      - and `lam` is an additive regularization constant.

    The object also supports incremental updates via `remove` and `reset`.

    Parameters
    ----------
    w : float64[:]
        Weight vector over the dataset.
    u : float64[:], optional
        Normalization weights. Should be positive. If None, defaults to all ones.
    norm : float, default=2
        The p of the p-norm used for normalization.
    lam : float, default=0
        Additive regularization in the denominator.

    Attributes
    ----------
    value : float
        Normalized objective value of the current support.
    value_removed : float
        Objective value of the removed portion.
    value_remaining : float
        Objective value of the remaining portion.

    sum_w_all : float
        Total weight of the current support.
    sum_u_all : int
        Total normalization weight of the current support.

    Examples
    --------
    >>> w = np.array([2.0, 4.0, 6.0])
    >>> u = np.array([1.0, 1.0, 1.0])
    >>> obj = NormalizedWeightedSupport(w, u, norm=1, lam=0)
    >>> obj.support(np.array([0, 1, 2]))
    >>> obj.value
    12.0 / 3.0  # = 4.0

    >>> obj.remove(1)
    >>> obj.value_removed
    4.0 / 1.0  # = 4.0
    >>> obj.value_remaining
    8.0 / 2.0  # = 4.0
    """

    w: float64[:]
    u: float64[:]
    power: float64
    lam: float64
    
    sum_w_all: float64
    sum_w_removed: float64
    sum_w_remaining: float64
    
    sum_u_all: int64
    sum_u_removed: int64
    sum_u_remaining: int64

    value: float64
    value_removed: float64
    value_remaining: float64

    def __init__(self, w, u=None, norm=2, lam=0):
        self.w = w.astype('float64')
        self.u = u.astype('float64') if u is not None else np.ones(len(w))
        self.power = 1/norm
        self.lam = lam
        self.support(np.arange(len(w)))
        self.reset()

    def support(self, support):
        self.sum_w_all = np.sum(self.w[support])
        self.sum_u_all = np.sum(self.u[support])
        self.value = self.sum_w_all / (self.sum_u_all**self.power + self.lam)

    def reset(self):
        self.sum_w_removed = 0
        self.sum_w_remaining = self.sum_w_all
        self.sum_u_removed = 0
        self.sum_u_remaining = self.sum_u_all
        self.value_removed = 0
        self.value_remaining = self.sum_w_remaining / (self.sum_u_remaining**self.power + self.lam)

    def remove(self, i):
        wi = self.w[i]
        ui = self.u[i]
        self.sum_w_removed += wi
        self.sum_w_remaining -= wi
        self.sum_u_removed += ui
        self.sum_u_remaining -= ui
        self.value_removed = self.sum_w_removed / (self.sum_u_removed**self.power + self.lam)
        self.value_remaining = self.sum_w_remaining / (self.sum_u_remaining**self.power + self.lam)

# @njit
def greedy_maximization(x, obj, max_depth=5):
    n, p = x.shape
    orders = argsort_columns(x)
    support = np.ones(n, dtype=np.bool_)
    support_count = n

    cum_support_count = 0
    non_separable = 0

    v = np.zeros(max_depth, dtype=np.int64)
    s = np.zeros(max_depth, dtype=np.int64)
    t = np.zeros(max_depth, dtype=np.float64)

    current = np.zeros(p, dtype=np.int64) # cursor buffer for order updates
    
    best_value = obj.value
    num_cond = 0

    for k in range(1, max_depth+1):
        cum_support_count += support_count

        obj.support(orders[:support_count, 0])
        
        best_j, best_i, best_s = -1, -1, 1
        improvement = False
        for j in range(p):

            obj.reset()
            
            for i in range(support_count - 1): 
                # test splits between x^j_i (last left) and x^j_i+1 (first right)
                
                obj.remove(orders[i, j])
                
                if x[orders[i, j], j]==x[orders[i+1, j], j]:
                    non_separable += 1
                    continue

                if obj.value_removed > best_value:
                    best_i = i
                    best_j = j
                    best_s = -1
                    best_value = obj.value_removed
                    improvement = True
                elif obj.value_remaining > best_value:
                    best_i = i
                    best_j = j
                    best_s = 1
                    best_value = obj.value_remaining
                    improvement = True

        if not improvement:
            break

        v[k-1] = best_j
        s[k-1] = best_s
        t[k-1] = best_s*(x[orders[best_i, best_j], best_j] + x[orders[best_i + 1, best_j], best_j]) / 2
        num_cond = k

        if best_s == 1: # lower bound
            support[orders[:best_i+1, best_j]] = False
        else: # upper bound
            support[orders[best_i+1:, best_j]] = False

        current[:] = 0
        for i in range(support_count): # need old support count here
            for j in range(p): # can this loop be vectorised?
                if support[orders[i, j]]:
                    orders[current[j], j] = orders[i, j]
                    current[j] += 1

        if best_s == 1: # lower bound
            support_count = support_count - best_i - 1
        else: # upper bound
            support_count = best_i + 1
    if num_cond == 0:
        res = Propositionalization(v, t, s)
    else:
        res = Propositionalization(v[:num_cond], t[:num_cond], s[:num_cond])
    return res, best_value, {'cum_support_count': cum_support_count,
                           'non_separable': non_separable}


if __name__=='__main__':
    import doctest
    doctest.testmod()
