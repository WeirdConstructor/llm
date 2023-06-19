//! Defines the samplers used for generation.
//!
//! You can define your own [Sampler] by implementing the trait.

use std::fmt::Debug;

use partial_sort::PartialSort;
use rand::{distributions::WeightedIndex, prelude::Distribution};

use crate::{TokenBias, TokenId};

#[derive(Debug, Clone, Copy)]
struct Sample {
    token: TokenId,
    logit: f32,
    p: f32,
}

#[derive(Debug, Clone)]
struct Samples {
    candidates: Vec<Sample>,
}

impl Samples {
    pub fn apply_bias(&mut self, bias: &TokenBias) {
        for sample in self.candidates.iter_mut() {
            if let Some(new_logit) = bias.get(sample.token) {
                sample.logit = new_logit;
            }
        }
    }

    pub fn apply_repetition_penalty(
        &mut self,
        previous_tokens: &[TokenId],
        repeat_penalty: f32,
        repetition_penalty_last_n: usize,
    ) {
        for sample in self.candidates.iter_mut() {
            if previous_tokens[previous_tokens
                .len()
                .saturating_sub(repetition_penalty_last_n)..]
                .contains(&sample.token)
            {
                // repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
                // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main

                // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if sample.logit < 0.0 {
                    sample.logit *= repeat_penalty
                } else {
                    sample.logit /= repeat_penalty
                }
            };
        }
    }

    pub fn from_logits_with_temperature(logits: &[f32], temperature: f32) -> Self {
        let candidates = logits
            .iter()
            .enumerate()
            .map(|(token_index, logit)| Sample {
                token: token_index as TokenId,
                logit: *logit * temperature,
                p: 0.0,
            })
            .collect();

        Self { candidates }
    }

    pub fn softmax(&mut self) {
        let maxl = self
            .candidates
            .iter()
            .map(|x| x.logit)
            .max_by(f32::total_cmp)
            .unwrap();

        let mut sum = 0.0;
        for sample in self.candidates.iter_mut() {
            sample.p = (sample.logit - maxl).exp();
            sum += sample.p;
        }

        for sample in self.candidates.iter_mut() {
            sample.p /= sum;
        }
    }

    pub fn sort(&mut self) {
        self.candidates.sort_by(|a, b| {
            // Sort descending
            b.logit.total_cmp(&a.logit)
        });
    }

    pub fn sample_top_k(&mut self, top_k: usize) {
        if top_k >= self.candidates.len() {
            self.sort();
            return;
        }

        self.candidates.partial_sort(top_k, |a, b| {
            // Sort descending
            b.logit.total_cmp(&a.logit)
        });

        self.candidates.truncate(top_k);
    }

    pub fn sample_top_p(&mut self, top_p: f32) {
        if top_p >= 1.0 {
            return;
        }

        let mut cumsum = 0.0;
        let mut top_idx = None;
        for (i, sample) in self.candidates.iter().enumerate() {
            cumsum += sample.p;

            if cumsum >= top_p {
                top_idx = Some(i);
                break;
            }
        }

        if let Some(top_idx) = top_idx {
            self.candidates.truncate(top_idx + 1);
        }

        let inverse_cumsum = 1.0 / cumsum;
        for sample in self.candidates.iter_mut() {
            sample.p *= inverse_cumsum;
        }
    }
}

/// A sampler for generation.
pub trait Sampler: Debug + Send + Sync {
    /// Given the previous tokens, the logits from the most recent evaluation, and a source of randomness,
    /// sample from the logits and return the token ID.
    fn sample(
        &self,
        previous_tokens: &[TokenId],
        logits: &[f32],
        rng: &mut dyn rand::RngCore,
    ) -> TokenId;
}

/// Top-P Top-K sampling.
///
/// A standard sampler that uses top-K sampling (the top-K tokens with the highest
/// probability are considered) and top-P sampling (only tokens with a cumulative
/// probability of `P` are considered).
///
/// It also implements [CTRL](https://arxiv.org/abs/1909.05858)'s repetition penalty,
/// and the ability to bias the generation of individual tokens.
#[derive(Clone, Debug)]
pub struct TopPTopK {
    /// The top K words by score are kept during sampling.
    pub top_k: usize,
    /// The cumulative probability after which no more words are kept for sampling.
    pub top_p: f32,
    /// The penalty for repeating tokens. Higher values make the generation less
    /// likely to get into a loop, but may harm results when repetitive outputs
    /// are desired.
    pub repeat_penalty: f32,
    /// Temperature (randomness) used for sampling. A higher number is more random.
    pub temperature: f32,
    /// A list of tokens to bias against in the process of generation.
    pub bias_tokens: TokenBias,
    /// The number of tokens to consider for the repetition penalty.
    pub repetition_penalty_last_n: usize,
}
impl Default for TopPTopK {
    fn default() -> Self {
        Self {
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.30,
            temperature: 0.80,
            bias_tokens: TokenBias::empty(),
            repetition_penalty_last_n: 512,
        }
    }
}
impl Sampler for TopPTopK {
    fn sample(
        &self,
        previous_tokens: &[TokenId],
        logits: &[f32],
        rng: &mut dyn rand::RngCore,
    ) -> TokenId {
        let Self {
            top_k,
            top_p,
            repeat_penalty,
            temperature,
            repetition_penalty_last_n,
            ..
        } = *self;
        let bias_tokens = &self.bias_tokens;

        let n_logits = logits.len();
        let mut logits_id = Vec::<(f32, TokenId)>::with_capacity(n_logits);

        // TODO: consider if this can be modularized and this sampler can be composed out of multiple pieces,
        // instead of having this monolithic function that embeds the repetition penalty and token bias
        {
            let scale = 1.0 / temperature;
            for (i, &logit) in logits.iter().enumerate() {
                let tid = i as TokenId;

                let val = if let Some(logit_override) = bias_tokens.get(tid) {
                    logit_override
                } else if previous_tokens[previous_tokens
                    .len()
                    .saturating_sub(repetition_penalty_last_n)..]
                    .contains(&(i as TokenId))
                {
                    // repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
                    // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main

                    // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                    if logits[i] < 0.0 {
                        logit * scale * repeat_penalty
                    } else {
                        logit * scale / repeat_penalty
                    }
                } else {
                    logit * scale
                };
                logits_id.push((val, tid));
            }
        }

        // find the top K tokens
        {
            logits_id.partial_sort(top_k, |a, b| {
                // Sort descending
                b.0.total_cmp(&a.0)
            });
            logits_id.truncate(top_k);
        }

        let maxl = logits_id
            .iter()
            .map(|x| x.0)
            .max_by(f32::total_cmp)
            .unwrap();

        // compute probs for the top K tokens
        let mut probs: Vec<f32> = logits_id
            .iter()
            .copied()
            .map(|(k, _)| (k - maxl).exp())
            .collect();
        let sum: f32 = probs.iter().copied().sum();

        // Normalize the probs
        for p in probs.iter_mut() {
            *p /= sum;
        }

        // Top p sampling
        if top_p < 1.0 {
            let mut cumsum = 0.0;
            for i in 0..probs.len() {
                cumsum += probs[i];
                if cumsum >= top_p {
                    probs.truncate(i + 1);
                    logits_id.truncate(i + 1);
                    break;
                }
            }

            cumsum = 1.0 / cumsum;
            for p in probs.iter_mut() {
                *p *= cumsum;
            }
        }

        let dist = WeightedIndex::new(&probs).expect("WeightedIndex error");
        let idx = dist.sample(rng);

        logits_id[idx].1
    }
}
