"""Implementation of domain split API."""
import sys
from enum import Enum
from functools import wraps, lru_cache

from dask import delayed

from .config import config
from .domain_split_API import (
    Domain,
    Border,
    Solver,
    SplitVisitor,
    MergeVisitor,
    Tailor,
    Splitable,
)
from .datastructure import State, Variable, np, Parameters
from dask.distributed import Client, Future
from struct import pack
from collections import deque
from typing import Callable, Optional, Sequence, Tuple, Dict
from dataclasses import dataclass, fields
from copy import copy, deepcopy

if sys.version_info < (3, 9):
    from typing import Deque
else:
    Deque = deque


def _ensure_others_is_tuple(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if "others" in kwargs and type(kwargs["others"]) is not Tuple:
            kwargs["others"] = tuple(kwargs["others"])
        elif type(args[0]) is not Tuple:
            args = tuple(a if i != 0 else tuple(a) for i, a in enumerate(args))
        return func(self, *args, **kwargs)

    return wrapper


class RegularSplitMerger(SplitVisitor, MergeVisitor):
    """Implements splitting and merging into regular grid."""

    __slots__ = ["dim", "_parts"]

    def __init__(self, parts: int, dim: Tuple[int]):
        """Initialize class instance."""
        self._parts = parts
        self.dim = dim

    def split_array(self, array: Optional[np.ndarray]) -> Tuple[np.ndarray, ...]:
        """Split array.

        Parameter
        ---------
        array: np.ndarray
          Array to split.

        Returns
        -------
        Tuple[np.ndarray, ...]
        """
        return np.array_split(array, indices_or_sections=self.parts, axis=self.dim[0])

    def merge_array(self, arrays: Sequence[np.ndarray]) -> np.ndarray:
        """Merge array.

        Parameter
        ---------
        arrays: Sequence[np.ndarray]
          Arrays to merge.

        Returns
        -------
        np.ndarray
        """
        return np.concatenate(arrays, axis=self.dim[0])

    def __hash__(self):
        """Return hash based on number of parts and dimension."""
        return hash((self.parts, self.dim))

    def __eq__(self, other):
        """Compare based on hashes."""
        return hash(self) == hash(other)

    @property
    def parts(self) -> int:
        """Return number of parts created by split."""
        return self._parts


class BorderDirection(Enum):
    """Enumerator of border possible directions."""

    LEFT = lambda w: slice(None, w)  # noqa: E731
    LEFT_HALO = lambda w: slice(w, 2 * w)  # noqa: E731
    RIGHT = lambda w: slice(-w, None)  # noqa: E731
    RIGHT_HALO = lambda w: slice(-2 * w, -w)  # noqa: E731
    CENTER = lambda w1, w2: slice(w1, -w2)  # noqa: E731


class BorderSplitter(SplitVisitor):
    """Implements splitting off stripes of a DomainState along a dimension."""

    __slots__ = ["_axis", "_slice"]

    def __init__(self, slice: slice, axis: int):
        """Initialize class instance."""
        self._axis = axis
        self._slice = slice

    def split_array(self, array: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split array.

        Parameter
        ---------
        array: np.ndarray
          Array to split.

        Returns
        -------
        Tuple[np.ndarray, ...]
        """
        slices = array.ndim * [slice(None)]
        slices[self._axis] = self._slice
        return (array[tuple(slices)],)

    @property
    def parts(self) -> int:
        """Return number of parts created by split."""
        return 1

    def __hash__(self):
        """Return hash based on axis and slice object."""
        return hash((self._axis, self._slice.start, self._slice.stop))

    def __eq__(self, other):
        """Compare based on hashes."""
        return hash(self) == hash(other)

class DomainState(Domain, Splitable):
    """Implements Domain and Splitable interface on State class."""

    __slots__ = ["u", "v", "eta", "id", "it", "history",]

    def __init__(
        self,
        data: np.ndarray,
        it: int = 0,
        id: int = 0,
    ):
        """Create new DomainState instance from references on Variable objects."""
        self.data = data
        self.id = id
        self.it = it

    def set_id(self, id):
        """Set id value."""
        self.id = id
        return self

    def get_id(self) -> int:
        """Get domain's ID."""
        return self.id

    def get_iteration(self) -> int:
        """Get domain's iteration."""
        return self.it

    def get_data(self):
        """Provide tuple of all Variables in this order: (u, v, eta)."""
        return self.data

    def increment_iteration(self) -> int:
        """Return incremented iteration from domain, not modify object itself."""
        return self.it + 1

    def split(
        self, splitter: SplitVisitor
    ) -> Tuple["DomainState", ...]:  # TODO: raise error if shape[dim[0]] // parts < 2
        """Implement the split method from API."""
        splitted = (
            splitter.split_array(self.data)
        )

        out = tuple(
            self.__class__(
                data,
                self.it,
                self.get_id(),
            )
            for data in zip(*splitted)
        )

        return out

    @classmethod
    def merge(cls, others: Sequence["DomainState"], merger: MergeVisitor):
        """Implement merge method from API."""
        if any(tuple(o.it != others[0].it for o in others)):
            raise ValueError(
                "Try to merge DomainStates that differ in iteration counter."
            )
        return DomainState(
            merger.merge_array([o.data for o in others]),
            others[0].get_iteration(),
            others[0].get_id(),
        )

    def copy(self):
        """Return a deep copy of the object."""
        return deepcopy(self)

    def __eq__(self, other) -> bool:
        """Return true if other is identical or the same as self."""
        if not isinstance(other, DomainState):
            return NotImplemented
        if self is other:
            return True
        for a in self.__slots__:
            if not getattr(self, a) == getattr(other, a):
                return False
        return True

class BorderState(DomainState, Border):
    """Implementation of Border class from API on State class."""

    def __init__(  # differs from Border.__init__ signature
        self,
        data: np.ndarray,
        width: int,
        dim: int,
        iteration: int,
        id: int = 0,
    ):
        """Create BorderState in the same way as DomainState."""
        super().__init__(data, iteration, id)
        self.width = width
        self.dim = dim

    @classmethod
    def create_border(cls, base: DomainState, width: int, direction: bool, dim: int):
        """Split border of a DomainState instance.

        The data of the boarder will be copied to avoid data races.
        """
        if direction:
            border_slice = BorderDirection.RIGHT(width)
        else:
            border_slice = BorderDirection.LEFT(width)
        splitter = BorderSplitter(slice=border_slice, axis=dim)
        splitted_state = base.split(splitter)[0]

        return cls.from_domain_state(splitted_state, width=width, dim=dim)

    @classmethod
    def from_domain_state(cls, domain_state: DomainState, width: int, dim: int):
        """Create an instance from a DomainState instance.

        No copies are created.
        """
        return cls(
            data=domain_state.data,
            width=width,
            dim=dim,
            iteration=domain_state.get_iteration(),
            id=domain_state.get_id(),
        )

    def get_width(self) -> int:
        """Get border's width."""
        return self.width

    def get_dim(self) -> int:
        """Get border's dimension."""
        return self.dim


class BorderMerger(MergeVisitor):
    """Implements merging of the borders with a DomainState along a dimension.

    This merger is suppose to be used in the merge classmethod of the DomainState class.
    The order of arguments must be (left_border, domain, right_border).
    """

    __slots__ = ["_axis", "_slice_left", "_slice_right", "_slice_center", "_width"]

    def __init__(self, width: int, axis: int):
        """Initialize class instance."""

        self._axis = axis
        self._slice_left = BorderDirection.LEFT(width)
        self._slice_right = BorderDirection.RIGHT(width)
        self._slice_center = BorderDirection.CENTER(width, width)
        self._width = width

    @classmethod
    def from_borders(
        cls, left_border: BorderState, right_border: BorderState
    ) -> "BorderMerger":
        """Create BorderManager from left and right border instance."""
        assert left_border.width == right_border.width
        assert left_border.dim == right_border.dim
        return cls(width=left_border.width, axis=left_border.dim)

    def merge_array(self, arrays: Sequence[np.ndarray]) -> np.ndarray:
        """Merge array.

        Parameter
        ---------
        arrays: Sequence[np.ndarray]
          Arrays to merge.

        Returns
        -------
        np.ndarray
        """
        slices_center = arrays[1].ndim * [slice(None)]
        slices_center[self._axis] = self._slice_center

        left, base, right = arrays
        out = np.concatenate((left, base[tuple(slices_center)], right), axis=self._axis)
        return out

    def __hash__(self):
        """Return hash based on axis and slice objects."""
        return hash((self._axis, self._width))

    def __eq__(self, other):
        """Compare based on hash values."""
        return hash(self) == hash(other)


# TODO: make this work for dim != 1
class Tail(Tailor):
    """Implement Tailor class from API."""

    @staticmethod
    def split_domain(
        base: DomainState, splitter: SplitVisitor
    ) -> Tuple[DomainState, ...]:
        """Split domain in subdomains.

        When splitting, the ids of the subdomains are set to `range(0, splitter.parts)`.
        """
        splitted = base.split(splitter)
        for i, s in enumerate(splitted):
            s.id = i
        return splitted

    @staticmethod
    def make_borders(
        base: DomainState, width: int, dim: int
    ) -> Tuple[BorderState, BorderState]:
        """Implement make_borders method from API."""
        return (
            BorderState.create_border(base, width, False, dim),
            BorderState.create_border(base, width, True, dim),
        )

    @staticmethod
    def stitch(
        base: DomainState, borders: Tuple[BorderState, BorderState], dims: tuple
    ) -> DomainState:
        """Implement stitch method from API.

        borders need to be ordered left_border, right_border
        """
        left_border, right_border = borders
        border_merger = BorderMerger.from_borders(left_border, right_border)

        if (
            base.get_iteration()
            == left_border.get_iteration()
            == right_border.get_iteration()
        ):
            assert base.get_id() == left_border.get_id() == right_border.get_id()
        else:
            raise ValueError(
                "Borders iteration mismatch. Left: {}, right: {}, domain: {}".format(
                    left_border.get_iteration(),
                    right_border.get_iteration(),
                    base.get_iteration(),
                )
            )

        return DomainState(
            border_merger.merge_array([left_border.data, base.data, right_border.data]),
            base.get_iteration(),
            base.get_id(),
        )


class Blur(Solver):
    def get_border_width(self) -> int:
        return 1

    def integration(self, domain: Domain) -> Domain:
        b_w = self.get_border_width()
        data = domain.get_data()
        out = np.zeros(data.shape, dtype=float)

        for i in range(data.shape[0]):
            for j in range(b_w, data.shape[1] - b_w):
                out[i, j] = np.average(data[(i - 1 if i != 0 else 0):i + 1, j - 1:j + 1])

        return DomainState(out, domain.increment_iteration(), domain.get_id())

    def partial_integration(self,
        border: BorderState,
        domain: DomainState,
        neighbor_border: BorderState,
        direction: bool,
        dim: int
    ) -> Border:
        b_w = border.get_width()
        dim = border.dim
        assert domain.u.grid.x.shape[dim] >= 2 * b_w

        if direction:
            halo_slice = BorderDirection.RIGHT_HALO(b_w)
        else:
            halo_slice = BorderDirection.LEFT_HALO(b_w)

        halo_splitter = BorderSplitter(slice=halo_slice, axis=dim)
        list_merger = RegularSplitMerger(2, (dim,))
        area_of_interest_splitter = BorderSplitter(
            slice=BorderDirection.CENTER(b_w, b_w), axis=dim
        )

        dom = domain.split(halo_splitter)[0]

        state_list = [dom, border, neighbor_border]

        if not direction:
            state_list.reverse()

        merged_state = DomainState.merge(state_list, list_merger)

        # tmp = self.integration(tmp)
        new_data = np.average(merged_state.data)

        result = BorderState.from_domain_state(
            DomainState(new_data, domain.increment_iteration(), domain.get_id()),
            width=border.get_width(),
            dim=dim,
        )
        result.id = domain.get_id()
        return result

def new_API_with_split_and_dask_on_ray(initial_state, dt, parts=4):

    def _make_borders(base, width, dim):
        return (
            delayed(BorderState.create_border)(base, width, False, dim),
            delayed(BorderState.create_border)(base, width, True, dim)
        )

    def _split():
        subs = initial_state.split(splitter)
        return [delayed(s) for s in subs]

    border_width = 2
    dim = (1,)
    splitter = RegularSplitMerger(parts, dim)
    border_merger = BorderMerger(border_width, dim[0])
    tailor = Tail()
    gs = Blur()

    domain_stack = deque([_split()], maxlen=2)
    border_stack = deque(
        [[_make_borders(sub, border_width, dim[0]) for sub in domain_stack[-1]]],
        maxlen=2,
    )
    for _ in range(0):
        new_borders = []
        new_subdomains = []
        for i, s in enumerate(domain_stack[-1]):
            new_borders.append(
                (
                    delayed(gs.partial_integration)(
                        border=border_stack[-1][i][0],
                        domain=s,
                        neighbor_border=border_stack[-1][i - 1][1],
                        direction=False,
                        dim=dim[0],
                    ),
                    delayed(gs.partial_integration)(
                        border=border_stack[-1][i][1],
                        domain=s,
                        neighbor_border=border_stack[-1][(i + 1) % splitter.parts][0],
                        direction=True,
                        dim=dim[0],
                    ),
                )
            )
        for s, borders in zip(domain_stack[-1], new_borders):
            new_subdomains.append(
                delayed(DomainState.merge)(
                    (borders[0], delayed(gs.integration)(s), borders[1]),
                    border_merger,
                )
            )
        domain_stack.append(new_subdomains)
        border_stack.append(new_borders)

    return delayed(DomainState.merge)(domain_stack[-1], splitter).compute(rerun_exceptions_locally=True)
    # dask.compute(*domain_stack[-1])
    # return 0

