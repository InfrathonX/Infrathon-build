# Fast KVzip baseline (reference only)

The released gate `qwen3-8b/q4_dim16_sink16.pt` (`Jang-Hyun/Fast-KVzip`) applied as kvpress
`FastKVzipPress`: the first 4 context positions and a trailing window of
`max(1, int(0.02 n))` positions (4096 when n >= 32000) are set to 1.0. The gate file is not in
this archive: the Judge copies it in only when the host asks for this reference baseline through
the private evaluation context, and reports the weights check as fired. An agent submitting this
is rejected by the weights check.
