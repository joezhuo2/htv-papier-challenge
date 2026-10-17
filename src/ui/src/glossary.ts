// one-line plain-language definitions shown on hover, so judges can read the screen without the presenter
export const GLOSSARY = {
  adversarial: 'Adversarial batch: inputs built to stress precision (a very long K, 8x larger values). Its error is measured on every run.',
  bound: 'Bound: the largest error a run may have and still pass. The lens sets it: Speed tolerates 2e-2, Precision demands 1e-3.',
  provisional: 'Provisional: passed under the looser Speed tolerance, but not yet checked against the strict Precision bound.',
  inconclusive: 'Inconclusive: the measured speedup is within run to run noise of the prediction, so it neither clearly passed nor failed.',
  rejected: 'Rejected: the code found a measured value outside the prediction or the bound. The model does not decide this.',
  confirmed: 'Confirmed: every check of the stated prediction passed outside the noise band.',
  noise: 'Noise band: how much the speedup moved between timing rounds of this run. Results inside it are not counted as wins.',
  eager: 'Eager: plain PyTorch (matmul, bias, GELU as separate ops). The real-world baseline to beat.',
  naive: 'Naive: the deliberately poor starting Triton kernel (tiny tiles, 1 warp). The agent starts from here.',
  speedup: 'Speedup: reference latency divided by this run, timed in the same window so drift cancels out.',
} as const

export type Term = keyof typeof GLOSSARY
