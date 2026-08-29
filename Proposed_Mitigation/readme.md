# Proposed Mitigation Mechanisms

We have also shared the proposed mitigation mechanisms for reference.

As discussed in the paper, we provide patches for two mitigation mechanisms: **Temperature Scaling** and **Noise Injection**. These modifications should be applied to the `kernel_forward.h` file provided in this directory.

---

## Mitigation 1: Temperature Scaling

The following patch implements the proposed temperature scaling-based mitigation.

Add the following code around **lines 1357–1361**:

```cpp
// --- Mitigation 1: Temperature Scaling ---
constexpr float temperature = 1.15f;

scaling /= temperature;

frag = cutlass::multiplies<Fragment>()(scaling * kLog2e, frag);