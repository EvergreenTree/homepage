# CoLU vs ReLU GPT comparison

This Parcel demo mirrors the `gpt_internals` setup but trains two tiny GPT variants in sequence so their learning
curves can be compared in tfjs-vis. One model keeps the standard ReLU non-linearity in the feed-forward block, while the
other swaps in the CoLU activation described in the prompt.

## Usage

```bash
yarn install
yarn watch
```

Then open the served page. Load the dataset, tweak hyperparameters if desired, train each variant via **Train ReLU** or
**Train CoLU**, and use **Gather Curves** to overlay their loss traces. Each activation also gets its own chart so you can
track learning step by step.
