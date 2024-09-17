# Naive Dataloader

The naive dataloader is the simplest form of a dataloader. Do not use this dataloader unless you have a specific reason to do so. It is fine for a quick experiment or for debuugging, but the naive dataloader can block your training loop, does not have sharding support, and is not recommended for production use. As a result, if you use the `DataLoader` builder to create your dataloader, the naive dataloader will **never** be used.

::: loadax.dataloader.naive.NaiveDataloader
    selection:
      members: false
