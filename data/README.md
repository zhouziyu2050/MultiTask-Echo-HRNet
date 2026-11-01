# Download EchoNet-LVH

Download the original EchoNet-LVH AVI videos:

- Project and research-use agreement: https://echonet.github.io/lvh/
- Dataset access portal: https://stanford.redivis.com/datasets/cchq-0srz1fy9a

Request access through the portal and keep the original filenames:

```text
data/
  raw/
    EchoNet-LVH/
      ... original AVI videos, including subdirectories ...
```

Annotations are included in `annotations/measurements.csv`. For an external data
location, set `ECHONET_DATA_ROOT` using the same directory layout.