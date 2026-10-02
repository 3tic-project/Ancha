use anyhow::{Result, ensure};

#[derive(Clone, Copy, Debug)]
pub struct ChunkPlan {
    pub samples: usize,
    pub step: usize,
}

impl ChunkPlan {
    pub fn new(samples: usize, overlap: usize) -> Result<Self> {
        ensure!(
            samples > 0 && overlap > 0 && overlap <= samples,
            "invalid chunk or overlap divisor"
        );
        Ok(Self {
            samples,
            step: samples / overlap,
        })
    }

    pub fn starts(self, length: usize) -> Vec<usize> {
        (0..length).step_by(self.step).collect()
    }

    /// Linear fade window. Edge chunks keep full weight on their exposed side.
    pub fn weight(self, i: usize, first: bool, last: bool) -> f32 {
        let fade = (self.samples / 10).max(1);
        if !first && i < fade {
            return (i + 1) as f32 / fade as f32;
        }
        if !last && i >= self.samples - fade {
            return (self.samples - i) as f32 / fade as f32;
        }
        1.0
    }
}

pub struct Accumulator {
    sums: Vec<Vec<f32>>,
    weights: Vec<f32>,
}

impl Accumulator {
    pub fn new(channels: usize, length: usize) -> Result<Self> {
        ensure!(channels > 0 && length > 0, "empty overlap accumulator");
        Ok(Self {
            sums: vec![vec![0.0; length]; channels],
            weights: vec![0.0; length],
        })
    }

    pub fn add(&mut self, plan: ChunkPlan, start: usize, planes: &[Vec<f32>]) -> Result<()> {
        ensure!(
            planes.len() == self.sums.len() && planes.iter().all(|p| p.len() == plan.samples),
            "chunk output shape mismatch"
        );
        ensure!(
            planes.iter().flatten().all(|x| x.is_finite()),
            "non-finite chunk output"
        );
        ensure!(start < self.weights.len(), "chunk starts past output");
        let end = (start + plan.samples).min(self.weights.len());
        for i in start..end {
            let w = plan.weight(
                i - start,
                start == 0,
                start + plan.samples >= self.weights.len(),
            );
            self.weights[i] += w;
            for (sum, plane) in self.sums.iter_mut().zip(planes) {
                sum[i] += w * plane[i - start];
            }
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<Vec<Vec<f32>>> {
        ensure!(
            self.weights.iter().all(|&w| w > 0.0),
            "uncovered overlap-add sample"
        );
        for sum in &mut self.sums {
            for (x, w) in sum.iter_mut().zip(&self.weights) {
                *x /= *w;
            }
        }
        Ok(self.sums)
    }
}
