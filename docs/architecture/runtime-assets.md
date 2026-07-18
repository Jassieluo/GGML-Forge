# Runtime Assets

Runtime assets follow the same category/provider boundary as source code:

```text
models/
`-- <category>/
    `-- <provider>/
```

A provider directory may contain portable model compositions, small frontend
resources, and local asset namespaces. Large weights, reference media,
generated caches, and local voices remain outside version control.

Configurations resolve provider artifact paths relative to the configuration
file that declares them. New model categories are added alongside existing
categories rather than sharing an unclassified weights directory.
