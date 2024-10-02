import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias, TypeVar

import jax
from jax._src.tree_util import KeyEntry
from jax.sharding import PartitionSpec

from loadax.logger import logger
from loadax.utils import is_named_tuple

_NestedT = TypeVar("_NestedT")
Nested: TypeAlias = _NestedT | dict[str, "Nested[_NestedT]"]


def complete_partition_spec_tree(
    treedef: jax.tree_util.PyTreeDef, partition_spec_tree: Nested[Any]
) -> Nested[Any]:
    """Creates a tree of PartitionSpecs or ParamPartitionSpecs.

    Adapted from flatten_axes(), but with a simplified API and more error logging
    and messages.

    Original:
    https://github.com/google/jax/blob/cdf4177f9219c9c0f70e243a097740e46138fc35/jax/_src/api_util.py#L277-L315

    Args:
        treedef: The tree structure of data to be partitioned according to
            `partition_spec_tree`.
        partition_spec_tree: A nested structure with PartitionSpecs or
            ParamPartitionSpecs and Nones at the leaves. Must be a tree prefix of
            `treedef`.

    Returns:
        A complete tree of PartitionSpecs or ParamPartitionSpecs and Nones that have
        the exact same structure as `treedef`.

    Raises:
        ValueError: If an unsupported type is encountered, or if partition_spec_tree
            is not a tree prefix of treedef.
    """
    proxy = object()
    dummy = jax.tree_util.tree_unflatten(treedef, [object()] * treedef.num_leaves)
    axes = []

    def replace_none_with_proxy(tree: Nested[Any]) -> Any:
        if tree is None:
            return proxy
        if isinstance(tree, PartitionSpec) or dataclasses.is_dataclass(tree):
            return tree
        if is_named_tuple(tree):
            return type(tree)(*[replace_none_with_proxy(x) for x in tree])
        if isinstance(tree, tuple | list):
            return type(tree)([replace_none_with_proxy(x) for x in tree])
        if isinstance(tree, dict):
            return type(tree)(
                [(k, replace_none_with_proxy(v)) for k, v in tree.items()]
            )
        raise ValueError(f"{type(tree)}: {tree}")

    partition_spec_tree_with_proxy = replace_none_with_proxy(partition_spec_tree)

    def add_leaves(i: Any, x: Any) -> None:
        axes.extend([i] * len(jax.tree_util.tree_flatten(x)[0]))

    try:
        jax.tree_util.tree_map(add_leaves, partition_spec_tree_with_proxy, dummy)
    except ValueError as err:
        logger.info("[complete_partition_spec_tree] ValueError: %s", err)
        logger.info(
            "[complete_partition_spec_tree] partition_spec_tree_with_proxy=%s",
            jax.tree_util.tree_structure(partition_spec_tree_with_proxy),
        )
        logger.info(
            "[complete_partition_spec_tree] dummy=%s",
            jax.tree_util.tree_structure(dummy),
        )
        for path, value in flatten_items(partition_spec_tree_with_proxy):
            logger.info(
                f"[complete_partition_spec_tree] partition_spec_tree_with_proxy leaf: "
                f"{path}={value}"
            )
        for path, value in flatten_items(dummy):
            logger.info("[complete_partition_spec_tree] dummy leaf: %s=%s", path, value)

        raise ValueError(
            f"specification must be a tree prefix of the "
            f"corresponding value, got specification {partition_spec_tree} "
            f"for value tree {treedef}. Original ValueError: {err}"
        ) from None
    axes = [None if a is proxy else a for a in axes]
    assert (
        len(axes) == treedef.num_leaves
    ), f"({len(axes)} vs. {treedef.num_leaves}) {axes} {treedef}"
    return jax.tree_util.tree_unflatten(treedef, axes)


def _key_entry_to_str(key_entry: KeyEntry) -> str:
    # Although (e.g.) DictKey does have its own __str__ implementation, calling
    # str(DictKey('a')) produces "['a']" instead of just "a".
    if isinstance(key_entry, jax.tree_util.DictKey):
        key = key_entry.key
    elif isinstance(key_entry, jax.tree_util.GetAttrKey):
        key = key_entry.name
    elif isinstance(key_entry, jax.tree_util.SequenceKey):
        key = key_entry.idx
    elif isinstance(key_entry, jax.tree_util.FlattenedIndexKey):
        key = key_entry.key
    else:
        raise RuntimeError(f"Unknown key entry type {type(key_entry)}: {key_entry}.")

    # Use f-string instead of calling str() because it matches the behavior of the
    # previous implementation and differs from str() for (e.g.) enums.
    return f"{key}"


def flatten_items(
    tree: Nested[jax.Array],
    separator: str = "/",
    is_leaf: Callable[[Any], bool] | None = None,
) -> Sequence[tuple[str, jax.Array]]:
    """Flattens `tree` and returns a list of (path, value) pairs."""
    flat_paths_and_values, _ = jax.tree_util.tree_flatten_with_path(
        tree, is_leaf=is_leaf
    )
    return [
        (separator.join(_key_entry_to_str(k) for k in path), value)
        for path, value in flat_paths_and_values
    ]


def tree_paths(
    tree: Nested[Any],
    separator: str = "/",
    is_leaf: Callable[[Any], bool] | None = None,
) -> Nested[Any]:
    """Create a tree of paths from a tree of values.

    Returns a tree of the same structure as `nested_tensor` but with corresponding
    paths instead of values.

    E.g.,
        tree_paths({'a': 1, 'b': [2, {'c': 3}]}) =
              {'a': 'a', 'b': ['b/0', {'c': 'b/1/c'}]}

    Args:
        tree: A nested structure.
        separator: The separator between parts of a path.
        is_leaf: A Callable to evaluate whether the given node should be considered a
                 leaf when it otherwise would not, similarly to the is_leaf
                 in jax.tree_util.tree_map.

    Returns:
        A nested structure with the same structure as `tree`, but each leaf will be a
        string path. Note that None is not considered a leaf by jax.tree_util,
        hence also preserved by tree_paths.
    """
    return jax.tree_util.tree_map_with_path(
        lambda kp, _: separator.join(_key_entry_to_str(k) for k in kp),
        tree,
        is_leaf=is_leaf,
    )


def shapes(nested_tensor: Nested[jax.Array]) -> Nested[Any]:
    """Creates a tree of shapes from a tree of tensors.

    Returns a tree of the same structure as `nested_tensor` but with corresponding
    shapes instead of tensors.
    """
    return jax.tree_util.tree_map(lambda x: getattr(x, "shape", x), nested_tensor)
