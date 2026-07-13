# Examples

The repository-owned example is `main.mojo`. Run it from the repository root:

```bash
pixi run run
```

`external_binary_data.mojo` preserves the original binary-data loading
workflow. It is optional and is not part of the build or test tasks because it
depends on resources outside this repository:

- a Mojo package that provides `read_bin.bin_to_df.DfReader`;
- the 82 binary input columns under `../../quant_models_expl/tmp_data/`,
  relative to the repository root.

With the package directory available on the Mojo import path, run it from the
repository root with:

```bash
pixi run mojo run -I . -I /path/to/read-bin-parent examples/external_binary_data.mojo
```

Update `base_path`, `DfReader` dimensions, and preview dimensions in the
example when using a different data layout.
