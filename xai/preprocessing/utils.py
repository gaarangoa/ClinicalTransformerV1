
from collections import defaultdict
import itertools
import array
import warnings

import numpy as np
import scipy.sparse as sp
import numbers 

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils.sparsefuncs import min_max_axis
from sklearn.utils import column_or_1d
from sklearn.utils.validation import _num_samples, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import type_of_target
from sklearn.utils._encode import _unique

def is_scalar_nan(x):
    """Tests if x is NaN.
    This function is meant to overcome the issue that np.isnan does not allow
    non-numerical types as input, and that np.nan is not float('nan').
    Parameters
    ----------
    x : any type
    Returns
    -------
    boolean
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils import is_scalar_nan
    >>> is_scalar_nan(np.nan)
    True
    >>> is_scalar_nan(float("nan"))
    True
    >>> is_scalar_nan(None)
    False
    >>> is_scalar_nan("")
    False
    >>> is_scalar_nan([np.nan])
    False
    """
    return isinstance(x, numbers.Real) and math.isnan(x)

class _nandict(dict):
    """Dictionary with support for nans."""

    def __init__(self, mapping):
        super().__init__(mapping)
        for key, value in mapping.items():
            if is_scalar_nan(key):
                self.nan_value = value
                break

    def __missing__(self, key):
        if hasattr(self, "nan_value") and is_scalar_nan(key):
            return self.nan_value
        raise KeyError(key)

def _map_to_integer(values, uniques):
    """Map values based on its position in uniques."""
    table = _nandict({val: i+1 for i, val in enumerate(uniques)})
    return np.array([table[v] for v in values])

def _encode(values, *, uniques, check_unknown=True):
    """Helper function to encode values into [0, n_uniques - 1].
    Uses pure python method for object dtype, and numpy method for
    all other dtypes.
    The numpy method has the limitation that the `uniques` need to
    be sorted. Importantly, this is not checked but assumed to already be
    the case. The calling method needs to ensure this for all non-object
    values.
    Parameters
    ----------
    values : ndarray
        Values to encode.
    uniques : ndarray
        The unique values in `values`. If the dtype is not object, then
        `uniques` needs to be sorted.
    check_unknown : bool, default=True
        If True, check for values in `values` that are not in `unique`
        and raise an error. This is ignored for object dtype, and treated as
        True in this case. This parameter is useful for
        _BaseEncoder._transform() to avoid calling _check_unknown()
        twice.
    Returns
    -------
    encoded : ndarray
        Encoded values
    """
    if values.dtype.kind in "OUS":
        try:
            return _map_to_integer(values, uniques)
        except KeyError as e:
            raise ValueError(f"y contains previously unseen labels: {str(e)}")
    else:
        if check_unknown:
            diff = _check_unknown(values, uniques)
            if diff:
                raise ValueError(f"y contains previously unseen labels: {str(diff)}")
        return np.searchsorted(uniques, values)

class LabelEncoder1Plus(TransformerMixin, BaseEstimator):
    """Encode target labels with value between 0 and n_classes-1.
    This transformer should be used to encode target values, *i.e.* `y`, and
    not the input `X`.
    Read more in the :ref:`User Guide <preprocessing_targets>`.
    .. versionadded:: 0.12
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Holds the label for each class.
    See Also
    --------
    OrdinalEncoder : Encode categorical features using an ordinal encoding
        scheme.
    OneHotEncoder : Encode categorical features as a one-hot numeric array.
    Examples
    --------
    `LabelEncoder` can be used to normalize labels.
    >>> from sklearn import preprocessing
    >>> le = preprocessing.LabelEncoder()
    >>> le.fit([1, 2, 2, 6])
    LabelEncoder()
    >>> le.classes_
    array([1, 2, 6])
    >>> le.transform([1, 1, 2, 6])
    array([0, 0, 1, 2]...)
    >>> le.inverse_transform([0, 0, 1, 2])
    array([1, 1, 2, 6])
    It can also be used to transform non-numerical labels (as long as they are
    hashable and comparable) to numerical labels.
    >>> le = preprocessing.LabelEncoder()
    >>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
    LabelEncoder()
    >>> list(le.classes_)
    ['amsterdam', 'paris', 'tokyo']
    >>> le.transform(["tokyo", "tokyo", "paris"])
    array([2, 2, 1]...)
    >>> list(le.inverse_transform([2, 2, 1]))
    ['tokyo', 'tokyo', 'paris']
    """

    def fit(self, y):
        """Fit label encoder.
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.
        Returns
        -------
        self : returns an instance of self.
            Fitted label encoder.
        """
        y = column_or_1d(y, warn=True)
        self.classes_ = _unique(y)
        return self

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels.
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.
        Returns
        -------
        y : array-like of shape (n_samples,)
            Encoded labels.
        """
        y = column_or_1d(y, warn=True)
        self.classes_, y = _unique(y, return_inverse=True)
        return y

    def transform(self, y):
        """Transform labels to normalized encoding.
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.
        Returns
        -------
        y : array-like of shape (n_samples,)
            Labels as normalized encodings.
        """
        check_is_fitted(self)
        y = column_or_1d(y, warn=True)
        # transform of empty array is empty array
        if _num_samples(y) == 0:
            return np.array([])

        return _encode(y, uniques=self.classes_)

    def inverse_transform(self, y):
        """Transform labels back to original encoding.
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target values.
        Returns
        -------
        y : ndarray of shape (n_samples,)
            Original encoding.
        """
        check_is_fitted(self)
        y = column_or_1d(y, warn=True)
        # inverse transform of empty array is empty array
        if _num_samples(y) == 0:
            return np.array([])

        diff = np.setdiff1d(y, np.arange(len(self.classes_)))
        if len(diff):
            raise ValueError("y contains previously unseen labels: %s" % str(diff))
        y = np.asarray(y)
        return self.classes_[y]

    def _more_tags(self):
        return {"X_types": ["1dlabels"]}