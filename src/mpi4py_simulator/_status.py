"""Shared MPI constants and status metadata."""

ANY_SOURCE = -1
ANY_TAG = -1


class Status:
    """Source and tag metadata populated by receive and probe operations.

    Construct an empty status and pass it to :meth:`Comm.recv`,
    :meth:`Comm.irecv`, :meth:`Comm.probe`, or :meth:`Comm.iprobe`. After a match,
    :attr:`source` and :attr:`tag` identify the actual message. The capitalized
    getters and setters mirror mpi4py's object-status API.

    An empty status starts with :data:`ANY_SOURCE` and :data:`ANY_TAG`. Applications
    normally read status values rather than setting them.

    """

    def __init__(self) -> None:
        """Initialize an empty status."""
        self._source = ANY_SOURCE
        self._tag = ANY_TAG

    @property
    def source(self) -> int:
        """Return the matched source rank, or ``ANY_SOURCE`` while empty."""
        return self._source

    @property
    def tag(self) -> int:
        """Return the matched message tag, or ``ANY_TAG`` while empty."""
        return self._tag

    def Get_source(self) -> int:  # noqa: N802
        """Return the source rank using mpi4py's method name.

        Returns:
            The same integer as :attr:`source`.

        """
        return self._source

    def Get_tag(self) -> int:  # noqa: N802
        """Return the message tag using mpi4py's method name.

        Returns:
            The same integer as :attr:`tag`.

        """
        return self._tag

    def Set_source(self, source: int) -> None:  # noqa: N802
        """Set source metadata using mpi4py's method name.

        Args:
            source: Source rank to store. Communicator operations call this method
                when they populate a user-provided status object.

        """
        self._source = source

    def Set_tag(self, tag: int) -> None:  # noqa: N802
        """Set tag metadata using mpi4py's method name.

        Args:
            tag: Message tag to store. Communicator operations call this method when
                they populate a user-provided status object.

        """
        self._tag = tag
