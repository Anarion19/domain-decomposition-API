"""Provide API for spliting domain and computing it asynchronously."""
from abc import ABC, abstractmethod
from typing import (
    Sequence,  # TODO: deprication handling
    # https://docs.python.org/3/library/typing.html#typing.Sequence
    Tuple,
)
from dask.distributed import Future, Client
from copy import deepcopy
import numpy as np


class SplitVisitor(ABC):
    """SplitVisitor class has methods for splitting arrays.

    Classes implementing the interface are used as arguments to `split`
    methods of classes implementing the `Splittable` interface.
    """

    @abstractmethod
    def split_array(self, array: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split numpy array in various parts."""
        pass

    @property
    @abstractmethod
    def parts(self) -> int:
        """Return number of splits."""
        pass


class MergeVisitor(ABC):
    """MergeVisitor class has methods for merging arrays.

    Classes implementing the interface are used as arguments to
    `merge` methods of classes implementing the `Splittable` interface.
    """

    @abstractmethod
    def merge_array(self, arrays: Sequence[np.ndarray]) -> np.ndarray:
        """Merge numpy array in various parts."""
        pass


class Splitable(ABC):
    """Splitable class has methods for splitting and merging its instances."""

    @abstractmethod
    def split(self, splitter: SplitVisitor):
        """Split the Domain into given number of parts along axis given by dim.

        For splitting among more than one axis pass tuple as dim.
        """
        pass

    @classmethod
    @abstractmethod
    def merge(cls, others: Sequence, merger: MergeVisitor):
        """Merge multiple Domains into one new domain."""
        pass


class Domain(ABC):
    """Domain class holds all of the interesting data.

    Class has methods allowing to split and merge with others.
    Methods set_id and get_id are just
    for easier debugging and are not used inside of API.
    """

    @abstractmethod
    def get_id(self) -> int:
        """Provide ID as int."""
        pass

    @abstractmethod
    def set_id(self, id):
        """Set ID."""
        pass

    @abstractmethod
    def get_data(self):
        """Provide all relevant data as tuple."""
        pass

    @abstractmethod
    def get_iteration(self) -> int:
        """Provide current iteration as int."""
        pass

    @abstractmethod
    def increment_iteration(self) -> int:
        """Return incremented iteration from domain, shouldn't modify object itself."""
        pass

    def __copy__(self):
        """Make deepcopy of itself."""
        return deepcopy(self)


class Border(ABC):
    """
    Border class holds minimal required data for computing subdomains.

    Should provide information about its width and dimension.
    Can either hold copy of the data or just a pointer to it.
    """

    @abstractmethod
    def get_width(self) -> int:
        """Provide border's width as int."""
        pass

    @abstractmethod
    def get_dim(self) -> int:
        """Provide border's dimension as int."""
        pass

    def __init__(self, base: Domain, width: int, direction: bool, dim: int):
        """Initialize the Border based on given Domain and width along axis given by dim.

        Direction specifies is it left (false) or right (true) border.
        """
        raise NotImplementedError


class Solver(ABC):
    """Solver class wraps methods required to solve given problem.

    Provides information of required border width.
    """

    @abstractmethod
    def get_border_width(self) -> int:
        """Provide minimal required border width."""
        pass

    @abstractmethod
    def partial_integration(
        self, domain: Domain, border: Border, past: Border, direction: bool, dim: int
    ) -> Border:
        """Compute border of given subdomain based on border from its neighbour."""
        pass

    @abstractmethod
    def integration(self, domain: Domain) -> Domain:
        """Compute next iteration, without items that are inside by borders."""
        pass

    @abstractmethod
    def window(self, domain: Future, client: Client) -> Future:
        """Get called on each subdomain after finishing iteration.

        This function receives Future on Domain and must return one.
        If one wants to modify Domain, then should submit task to Client with
        received Future and use returned one as function's output.
        All necessary communication should be performed within this function.
        One shouldn't perform extensive computation, as next iteration starts
        after this functions is completed. To perform tasks that don't modify
        domain it is suggested to use Dask's fire_and_forget function.
        """
        pass


class Tailor(ABC):
    """Tailor class keeps functions required for Border class."""

    @abstractmethod
    def make_borders(self, base: Domain, width: int, dim: int) -> Tuple[Border, Border]:
        """Wrap user-implemented Border class constructor.

        Outputs left border first.
        """
        pass

    @abstractmethod
    def stitch(self, base: Domain, borders: tuple, dims: tuple) -> Domain:
        """Copy data from Border to Domain.

        Whether it mutates Domain object is up to implementation. Borders are pact
        in tuple with length of equal number of axis along which borders are used.
        Each element is a tuple containing left and right Border (in this order).
        Order of axis in the tuple is determined by order dims tuple used in
        magic function, which is passed to this function.
        """
        pass


def compute_borders(
    center: Domain, border: Border, pst: Border, direction: bool, slv: Solver
) -> Border:
    """Compute one border on provided direction.

    Right = True; Left = False.
    If borders from neighbouring blocks are on the same iteration
    as the domain computes next iteration of borders, otherwise rise Exception.
    """
    if border.get_iteration() == center.get_iteration():
        return slv.partial_integration(center, border, pst, direction, border.get_dim())
    else:
        raise Exception(
            "Borders from neighbouring domains are in wrong iteration. "
            "border: {}, self: {}".format(
                border.get_iteration(), center.get_iteration()
            )
        )


def magic(
    domain: Domain,
    split: int,
    iterations: int,
    dims: tuple,
    slv: Solver,
    tlr: Tailor,
    client: Client,
):
    """Do the magic.

    Each Border key is a tuple containing (domain index, dimension, left or right)
    """
    subs = domain.split(split, dims)
    brd = {}

    for i in range(split):
        for d in dims:
            tmp = tlr.make_borders(subs[i], slv.get_border_width(), d)
            brd[(i, d, 0)] = tmp[0]
            brd[(i, d, 1)] = tmp[1]

    subs = client.scatter(subs)
    brd = client.scatter(brd)

    new_borders = brd.copy()
    new_subs = subs.copy()

    for it in range(iterations):
        for i in range(split):
            for d in dims:
                """Implements periodic boundary conditions"""
                bottom = i - 1 if i - 1 >= 0 else split - 1
                top = i + 1 if i + 1 < split else 0

                new_borders[(i, d, 0)] = client.submit(
                    compute_borders,
                    subs[i],
                    brd[(bottom, d, 1)],
                    brd[(i, d, 0)],
                    False,
                    slv,
                )
                new_borders[(i, d, 1)] = client.submit(
                    compute_borders,
                    subs[i],
                    brd[(top, d, 0)],
                    brd[(i, d, 1)],
                    True,
                    slv,
                )

        for i in range(len(subs)):
            new = client.submit(slv.integration, subs[i])
            tmp = tuple((new_borders[(i, d, 0)], new_borders[(i, d, 1)]) for d in dims)
            new_subs[i] = client.submit(tlr.stitch, new, tmp, dims)
            new_subs[i] = slv.window(new_subs[i], client)

        subs = new_subs.copy()
        brd = new_borders.copy()

    return subs
