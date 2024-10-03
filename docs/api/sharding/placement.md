# Placement Utilities

Loadax provides a few utilities to help with sharding and placement of data and models.

First we should cover why these utilities are necessary.

In a distributed setting jax wants to know about the placement of regions of data called "shards". Shards are ranges within an array that fit on a single device. When training on multiple devices, and especially when using multiple nodes, you want to simplify the synchronization and allow jax to handle as much as possible. A great way to do this is to have each node load a unique subset of your dataset. This will make every batch unique on each device. In a single node setup, this is a no-op and you would not need to shard your dataset. If you are using multiple nodes you then have the challenge of synchronizing your training such that each node trains on a unique subset of the data and the nodes coordinate the backpropagation and gradient updates.

This is where Loadax's `host_to_global_device_array` comes in. This function will take an array and communicate with all other nodes in the network to create a "global" array that is the same across all devices. This is different than sharding the array because the data never actually moves. This function annotates the array with the placement of the data so that jax can treat each node's batch as a single larger global batch with the local batches stitched together.


<!-- TODO: Visual example -->

## host_to_global_device_array

This function takes an array and annotates it with the placement of the data so that jax can treat each node's batch as a single larger global batch with the local batches stitched together.

::: loadax.sharding.placement.host_to_global_device_array


## global_to_host_array

The inverse of `host_to_global_device_array`. This function takes a global array and splits it into the local arrays for each node.

::: loadax.sharding.placement.global_to_host_array

### with_sharding_constraint

This is syntactic sugar that ensures a `with_sharding_constraint` is applied when inside a Mesh context.

::: loadax.sharding.placement.with_sharding_constraint
