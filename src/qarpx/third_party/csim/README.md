# csim (vendored from qulacs)

State-vector kernel vendored from [qulacs](https://github.com/qulacs/qulacs)
`src/csim/` at **v0.6.14**, MIT-licensed — see `LICENSE` (upstream carries no
per-file headers; the root LICENSE is the required notice).

All `.cpp`/`.hpp` files are byte-identical to upstream; only `CMakeLists.txt`
is ours. Keep it that way: local fixes go upstream or into qarpx wrappers, so
upgrades stay a straight file sync against upstream `src/csim/`.
