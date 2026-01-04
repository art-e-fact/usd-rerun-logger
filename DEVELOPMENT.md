## Development

### Publishing to Pypi

After merging to `main`:

1. checkout `main`, then:
    * update the `unreleased` section of the CHANGELOG to reflect the new version you are going to release.
    * update pyproject.yaml with new version number or run `uv version x.x.x` 
    * usual `git add`, `git commit`
    * `git tag  -a x.x.x -m "x.x.x"` 
    * `git push` then `git push --tags`

2. Build and publish to pypi (you will need a pypi token)
```
# Build
uv build
# Publish
uv publish --token <pypi token>
```

### Building docs

```sh
make -C docs html
```

The [docs.yaml](./.github/workflows/docs.yml) GutHub action will build and publish the documentation after merging to main.
