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
```

---

## Mitigation 2: Noise Injection

The following patch implements noise injection using the Box–Muller transform.

Add the following code around **lines 1366–1404**:

```cpp
// --- Mitigation 2: Noise Injection ---
#ifdef ENABLE_NOISE
    constexpr float noise_scale = 0.05f;

    if (noise_scale > 0.0f) {
        LambdaIterator::iterateRows(
            lane_offset,
            [&](int accum_m) {},
            [&](int accum_m, int accum_n, int idx) {
                if (accum_n < max_col) {
                    // Generate two independent uniform random numbers
                    // for the Box-Muller transform.

                    // First uniform random number (u1)
                    uint32_t seed1 =
                        (thread_id * 71) ^ (idx * 113) ^ (accum_n * 3);
                    seed1 ^= seed1 << 13;
                    seed1 ^= seed1 >> 17;
                    seed1 ^= seed1 << 5;
                    accum_t u1 =
                        (static_cast<accum_t>(seed1) + 1.0f) /
                        4294967297.0f;

                    // Second uniform random number (u2)
                    uint32_t seed2 =
                        (thread_id * 19) ^ (idx * 251) ^ (accum_m * 7);
                    seed2 ^= seed2 << 13;
                    seed2 ^= seed2 >> 17;
                    seed2 ^= seed2 << 5;
                    accum_t u2 =
                        (static_cast<accum_t>(seed2) + 1.0f) /
                        4294967297.0f;

                    // Box-Muller transform to generate
                    // a standard normal random number.
                    constexpr float kPi = 3.141592653589793f;
                    accum_t noise =
                        noise_scale *
                        sqrtf(-2.0f * logf(u1)) *
                        cosf(2.0f * kPi * u2);

                    // Clip noise for stability, creating a
                    // truncated normal distribution.
                    constexpr float max_noise = 0.05f;
                    noise = cutlass::fast_max(
                        -max_noise,
                        cutlass::fast_min(max_noise, noise)
                    );

                    frag[idx] += noise;
                }
            },
            [&](int accum_m) {});
    }
#endif
```

---

## Applying the Patch

To apply either mitigation mechanism:

- Locate the corresponding `kernel_forward.h` file in the PyTorch Transformer module.
- Apply the desired mitigation patch to the file as described above.
- Replace the original `kernel_forward.h` file with the modified version.
- Rebuild the PyTorch (`torch`) library to ensure that the modifications are compiled successfully.
- Run the experiments using the rebuilt PyTorch library.

> **Note:** The noise injection mitigation is controlled using the `ENABLE_NOISE` preprocessor macro. Ensure that this macro is enabled during compilation when evaluating the noise injection mechanism.
