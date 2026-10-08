//! Full-scale Potjans-Diesmann 2014 cortical microcircuit feasibility runner.
//!
//! This runner deliberately builds topology in Rust instead of materialising
//! hundreds of millions of Python integers or JSON values.  It is the first
//! consumer of the generic compact fixed-total projection builder.

use b2_runner::large_topology::{
    build_fixed_total, deterministic_normal, estimated_resident_bytes, CompactProjection,
    FixedTotalSpec,
};
use serde_json::json;
use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const POPULATIONS: [&str; 8] = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"];
const FULL_NEURONS: [usize; 8] = [20_683, 5_834, 21_915, 5_479, 4_850, 1_065, 14_395, 2_948];
const CONNECTION_PROBABILITIES: [[f64; 8]; 8] = [
    [0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.0, 0.0076, 0.0],
    [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0, 0.0042, 0.0],
    [0.0077, 0.0059, 0.0497, 0.1350, 0.0067, 0.0003, 0.0453, 0.0],
    [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0, 0.1057, 0.0],
    [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0],
    [0.0548, 0.0269, 0.0257, 0.0022, 0.0600, 0.3158, 0.0086, 0.0],
    [
        0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252,
    ],
    [
        0.0364, 0.0010, 0.0034, 0.0005, 0.0277, 0.0080, 0.0658, 0.1443,
    ],
];
const INITIAL_MEAN: [f64; 8] = [
    -68.28, -63.16, -63.33, -63.45, -63.11, -61.66, -66.72, -61.43,
];
const INITIAL_STD: [f64; 8] = [5.36, 4.57, 4.74, 4.94, 4.94, 4.55, 5.46, 4.48];
const EXTERNAL_INDEGREE: [f64; 8] = [
    1600.0, 1500.0, 2100.0, 1900.0, 2000.0, 1900.0, 2900.0, 2100.0,
];
const FULL_REFERENCE_RATES: [f64; 8] = [0.903, 2.965, 4.414, 5.876, 7.569, 8.633, 1.105, 7.829];

const DT_MS: f64 = 0.1;
const TAU_M_MS: f64 = 10.0;
const TAU_SYN_MS: f64 = 0.5;
const C_M_PF: f64 = 250.0;
const E_L_MV: f64 = -65.0;
const V_THRESHOLD_MV: f64 = -50.0;
const V_RESET_MV: f64 = -65.0;
const REFRACTORY_MS: f64 = 2.0;
const PSP_EXCITATORY_MV: f64 = 0.15;
const INHIBITORY_RATIO: f64 = -4.0;
const WEIGHT_RELATIVE_STD: f64 = 0.1;
const BACKGROUND_RATE_HZ: f64 = 8.0;

struct Projection {
    source_population: usize,
    topology: CompactProjection,
}

struct Options {
    neuron_scale: f64,
    indegree_scale: f64,
    duration_ms: f64,
    repetitions: usize,
    warmups: usize,
    seed: u64,
    maximum_gib: f64,
    build_only: bool,
    estimate_only: bool,
    output: Option<PathBuf>,
}

fn parse_value<T: std::str::FromStr>(args: &[String], at: &mut usize, name: &str) -> Result<T>
where
    T::Err: Error + 'static,
{
    *at += 1;
    args.get(*at)
        .ok_or_else(|| format!("missing value for {name}").into())
        .and_then(|value| value.parse::<T>().map_err(Into::into))
}

fn options() -> Result<Options> {
    let args: Vec<String> = env::args().collect();
    let mut result = Options {
        neuron_scale: 0.01,
        indegree_scale: 0.01,
        duration_ms: 10.0,
        repetitions: 5,
        warmups: 1,
        seed: 55,
        maximum_gib: 24.0,
        build_only: false,
        estimate_only: false,
        output: None,
    };
    let mut at = 1;
    while at < args.len() {
        match args[at].as_str() {
            "--full" => {
                result.neuron_scale = 1.0;
                result.indegree_scale = 1.0;
            }
            "--neuron-scale" => {
                result.neuron_scale = parse_value(&args, &mut at, "--neuron-scale")?
            }
            "--indegree-scale" => {
                result.indegree_scale = parse_value(&args, &mut at, "--indegree-scale")?
            }
            "--duration-ms" => result.duration_ms = parse_value(&args, &mut at, "--duration-ms")?,
            "--repetitions" => result.repetitions = parse_value(&args, &mut at, "--repetitions")?,
            "--warmups" => result.warmups = parse_value(&args, &mut at, "--warmups")?,
            "--seed" => result.seed = parse_value(&args, &mut at, "--seed")?,
            "--maximum-gib" => result.maximum_gib = parse_value(&args, &mut at, "--maximum-gib")?,
            "--build-only" => result.build_only = true,
            "--estimate-only" => result.estimate_only = true,
            "--output" => result.output = Some(parse_value(&args, &mut at, "--output")?),
            "--help" | "-h" => {
                println!("pd14 [--full] [--neuron-scale F] [--indegree-scale F] [--duration-ms MS] [--warmups N] [--repetitions N] [--seed N] [--maximum-gib GiB] [--build-only|--estimate-only] [--output FILE]");
                std::process::exit(0);
            }
            value => return Err(format!("unknown argument {value}").into()),
        }
        at += 1;
    }
    let valid_scale = |value: f64| value.is_finite() && value > 0.0 && value <= 1.0;
    let valid = valid_scale(result.neuron_scale)
        && valid_scale(result.indegree_scale)
        && result.duration_ms.is_finite()
        && result.duration_ms > 0.0
        && result.repetitions != 0
        && result.maximum_gib.is_finite()
        && result.maximum_gib > 0.0;
    if !valid {
        return Err("invalid PD14 options".into());
    }
    Ok(result)
}

fn fixed_total(probability: f64, target: usize, source: usize) -> usize {
    if probability == 0.0 {
        return 0;
    }
    let pairs = (target as f64) * (source as f64);
    ((-probability).ln_1p() / ((pairs - 1.0) / pairs).ln()).round() as usize
}

fn psc_per_psp() -> f64 {
    let sub = 1.0 / (TAU_SYN_MS - TAU_M_MS);
    let pre = TAU_M_MS * TAU_SYN_MS / C_M_PF * sub;
    let fraction = (TAU_M_MS / TAU_SYN_MS).powf(sub);
    1.0 / (pre * (fraction.powf(TAU_M_MS) - fraction.powf(TAU_SYN_MS)))
}

fn scaled_neurons(scale: f64) -> [usize; 8] {
    std::array::from_fn(|index| ((FULL_NEURONS[index] as f64 * scale).round() as usize).max(1))
}

fn population_offsets(neurons: &[usize; 8]) -> [usize; 9] {
    let mut result = [0usize; 9];
    for index in 0..8 {
        result[index + 1] = result[index] + neurons[index];
    }
    result
}

fn projection_counts(neuron_scale: f64, indegree_scale: f64) -> [[usize; 8]; 8] {
    std::array::from_fn(|target| {
        std::array::from_fn(|source| {
            let full = fixed_total(
                CONNECTION_PROBABILITIES[target][source],
                FULL_NEURONS[target],
                FULL_NEURONS[source],
            );
            (full as f64 * neuron_scale * indegree_scale).round() as usize
        })
    })
}

fn topology_memory(neurons: &[usize; 8], counts: &[[usize; 8]; 8]) -> Result<usize> {
    let mut bytes = 0usize;
    for row in counts {
        for (source, &edge_count) in row.iter().enumerate() {
            if edge_count != 0 {
                bytes = bytes
                    .checked_add(estimated_resident_bytes(neurons[source], edge_count)?)
                    .ok_or("topology memory estimate overflow")?;
            }
        }
    }
    Ok(bytes)
}

fn build_topology(
    neurons: &[usize; 8],
    offsets: &[usize; 9],
    counts: &[[usize; 8]; 8],
    seed: u64,
    indegree_scale: f64,
) -> Result<Vec<Projection>> {
    let current_per_psp = psc_per_psp();
    let weight_scale = indegree_scale.sqrt().recip();
    let mut projections = Vec::new();
    for target in 0..8 {
        for source in 0..8 {
            let edges = counts[target][source];
            if edges == 0 {
                continue;
            }
            let mut psp = if source % 2 == 0 {
                PSP_EXCITATORY_MV
            } else {
                PSP_EXCITATORY_MV * INHIBITORY_RATIO
            };
            if target == 0 && source == 2 {
                psp = 2.0 * PSP_EXCITATORY_MV;
            }
            let weight_mean = psp * current_per_psp * weight_scale;
            let delay_mean_ticks = if source % 2 == 0 { 15.0 } else { 7.5 };
            let projection_seed = seed
                ^ (target as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                ^ (source as u64 + 1).wrapping_mul(0xd2b7_4407_b1ce_6e93);
            projections.push(Projection {
                source_population: source,
                topology: build_fixed_total(FixedTotalSpec {
                    source_count: neurons[source],
                    target_count: neurons[target],
                    target_offset: offsets[target],
                    edge_count: edges,
                    seed: projection_seed,
                    weight_mean,
                    weight_std: weight_mean.abs() * WEIGHT_RELATIVE_STD,
                    excitatory: source % 2 == 0,
                    delay_mean_ticks,
                    delay_std_ticks: delay_mean_ticks * 0.5,
                    minimum_delay_ticks: 1,
                })?,
            });
        }
    }
    Ok(projections)
}

#[derive(Clone, Copy)]
struct Event {
    projection: u8,
    edge: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct SimulationResult {
    spike_counts: [usize; 8],
    delivered_events: u64,
}

fn simulate(
    neurons: &[usize; 8],
    offsets: &[usize; 9],
    projections: &[Projection],
    duration_ms: f64,
    seed: u64,
    indegree_scale: f64,
) -> Result<SimulationResult> {
    let steps = (duration_ms / DT_MS).round() as usize;
    let total_neurons = offsets[8];
    let current_per_psp = psc_per_psp();
    let external_weight = PSP_EXCITATORY_MV * current_per_psp / indegree_scale.sqrt();
    let mut voltage = vec![0.0; total_neurons];
    let mut current = vec![0.0; total_neurons];
    let mut refractory_until = vec![0usize; total_neurons];
    let mut dc = vec![0.0; total_neurons];
    for population in 0..8 {
        let dc_value = BACKGROUND_RATE_HZ
            * EXTERNAL_INDEGREE[population]
            * indegree_scale
            * TAU_SYN_MS
            * external_weight
            * 0.001;
        for local in 0..neurons[population] {
            let global = offsets[population] + local;
            voltage[global] = INITIAL_MEAN[population]
                + INITIAL_STD[population]
                    * deterministic_normal(seed, 100 + population as u64, local);
            dc[global] = dc_value;
        }
    }
    let max_delay = projections
        .iter()
        .flat_map(|projection| projection.topology.delay_ticks.iter().copied())
        .max()
        .unwrap_or(1) as usize;
    let mut queue: Vec<Vec<Event>> = (0..=max_delay).map(|_| Vec::new()).collect();
    let mut outgoing: [Vec<usize>; 8] = std::array::from_fn(|_| Vec::new());
    for (index, projection) in projections.iter().enumerate() {
        if index > u8::MAX as usize {
            return Err("too many compact projection blocks".into());
        }
        outgoing[projection.source_population].push(index);
    }
    let decay_m = (-DT_MS / TAU_M_MS).exp();
    let decay_syn = (-DT_MS / TAU_SYN_MS).exp();
    let synaptic_voltage = (decay_syn - decay_m) / (1.0 / TAU_M_MS - 1.0 / TAU_SYN_MS) / C_M_PF;
    let refractory_ticks = (REFRACTORY_MS / DT_MS).round() as usize;
    let mut spike_counts = [0usize; 8];
    let mut delivered_events = 0u64;
    for tick in 0..steps {
        let slot = tick % queue.len();
        let events = std::mem::take(&mut queue[slot]);
        for event in events {
            let projection = &projections[event.projection as usize].topology;
            let edge = event.edge as usize;
            current[projection.targets[edge] as usize] += f64::from(projection.weights[edge]);
            delivered_events += 1;
        }
        for population in 0..8 {
            for local in 0..neurons[population] {
                let neuron = offsets[population] + local;
                let old_current = current[neuron];
                current[neuron] = old_current * decay_syn;
                if tick < refractory_until[neuron] {
                    voltage[neuron] = V_RESET_MV;
                    continue;
                }
                let steady = E_L_MV + dc[neuron] * TAU_M_MS / C_M_PF;
                voltage[neuron] =
                    steady + (voltage[neuron] - steady) * decay_m + old_current * synaptic_voltage;
                if voltage[neuron] < V_THRESHOLD_MV {
                    continue;
                }
                voltage[neuron] = V_RESET_MV;
                refractory_until[neuron] = tick + refractory_ticks;
                spike_counts[population] += 1;
                for &projection_index in &outgoing[population] {
                    let projection = &projections[projection_index].topology;
                    let start = projection.offsets[local] as usize;
                    let end = projection.offsets[local + 1] as usize;
                    for edge in start..end {
                        let delivery = tick + projection.delay_ticks[edge] as usize;
                        let delivery_slot = delivery % queue.len();
                        queue[delivery_slot].push(Event {
                            projection: projection_index as u8,
                            edge: edge as u32,
                        });
                    }
                }
            }
        }
    }
    Ok(SimulationResult {
        spike_counts,
        delivered_events,
    })
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn relative_spread(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let minimum = values.iter().copied().reduce(f64::min).unwrap();
    let maximum = values.iter().copied().reduce(f64::max).unwrap();
    let center = median(values.to_vec());
    Some(if center == 0.0 {
        0.0
    } else {
        (maximum - minimum) / center
    })
}

fn main() {
    if let Err(error) = run() {
        eprintln!("pd14: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let options = options()?;
    let neurons = scaled_neurons(options.neuron_scale);
    let offsets = population_offsets(&neurons);
    let counts = projection_counts(options.neuron_scale, options.indegree_scale);
    let edge_count: usize = counts.iter().flatten().sum();
    let estimated_bytes = topology_memory(&neurons, &counts)?;
    let estimated_gib = estimated_bytes as f64 / 1024f64.powi(3);
    if estimated_gib > options.maximum_gib {
        return Err(format!(
            "estimated compact topology {estimated_gib:.3} GiB exceeds --maximum-gib {:.3}",
            options.maximum_gib
        )
        .into());
    }
    let build_started = Instant::now();
    let projections = if options.estimate_only {
        Vec::new()
    } else {
        build_topology(
            &neurons,
            &offsets,
            &counts,
            options.seed,
            options.indegree_scale,
        )?
    };
    let build_seconds = build_started.elapsed().as_secs_f64();
    let resident_bytes: usize = projections
        .iter()
        .map(|projection| projection.topology.resident_bytes())
        .sum();
    let mut elapsed = Vec::new();
    let mut last_result = SimulationResult {
        spike_counts: [0usize; 8],
        delivered_events: 0,
    };
    if !options.build_only && !options.estimate_only {
        for repetition in 0..options.warmups + options.repetitions {
            let started = Instant::now();
            let result = simulate(
                &neurons,
                &offsets,
                &projections,
                options.duration_ms,
                options.seed,
                options.indegree_scale,
            )?;
            let seconds = started.elapsed().as_secs_f64();
            if repetition >= options.warmups {
                if !elapsed.is_empty() && result != last_result {
                    return Err("repeated simulation changed deterministic result".into());
                }
                last_result = result;
                elapsed.push(seconds);
            }
        }
    }
    let rates: Vec<f64> = last_result
        .spike_counts
        .iter()
        .zip(neurons)
        .map(|(&spikes, count)| spikes as f64 / count as f64 / (options.duration_ms * 0.001))
        .collect();
    let report = json!({
        "schema": "b2-pd14-benchmark-v1",
        "model": "Potjans-Diesmann-2014",
        "numeric_profile": "f64-state/f32-static-weight/u16-delay",
        "neuron_scale": options.neuron_scale,
        "indegree_scale": options.indegree_scale,
        "duration_ms": options.duration_ms,
        "seed": options.seed,
        "population_names": POPULATIONS,
        "population_neurons": neurons,
        "neuron_count": offsets[8],
        "projection_count": projections.len(),
        "synapse_count": edge_count,
        "estimated_topology_bytes": estimated_bytes,
        "resident_topology_bytes": resident_bytes,
        "topology_build_seconds": build_seconds,
        "topology_edges_per_second": if build_seconds == 0.0 || options.estimate_only { None } else { Some(edge_count as f64 / build_seconds) },
        "warmups": options.warmups,
        "repetitions": options.repetitions,
        "simulation_seconds": elapsed,
        "simulation_median_seconds": if elapsed.is_empty() { None } else { Some(median(elapsed.clone())) },
        "simulation_relative_spread": relative_spread(&elapsed),
        "delivered_events": last_result.delivered_events,
        "delivered_events_per_second": if elapsed.is_empty() { None } else { Some(last_result.delivered_events as f64 / median(elapsed.clone())) },
        "spike_counts": last_result.spike_counts,
        "population_rates_hz": rates,
        "reference_rates_hz": FULL_REFERENCE_RATES,
    });
    let text = serde_json::to_string_pretty(&report)? + "\n";
    if let Some(path) = options.output {
        fs::write(path, &text)?;
    }
    print!("{text}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_full_scale_shape_is_preserved() {
        let counts = projection_counts(1.0, 1.0);
        assert_eq!(FULL_NEURONS.iter().sum::<usize>(), 77_169);
        assert_eq!(counts.iter().flatten().sum::<usize>(), 298_880_968);
        assert_eq!(counts[0][0], 45_499_805);
        assert_eq!(counts[6][7], 10_827_677);
    }
}
