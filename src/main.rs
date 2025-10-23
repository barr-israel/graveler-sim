use std::thread;

use rand_core::{RngCore, SeedableRng};
use simd_rand::portable::{SimdRandX8, Xoshiro256PlusX8, Xoshiro256PlusX8Seed};

fn main() {
    let max = par_roll();
    println!("{max}");
}

const MASK: u64 = !((1 << 25) - 1); // we don't want the last 25 bits
fn double_roll(rng: &mut Xoshiro256PlusX8) -> u32 {
    let roll = rng.next_u64x8();
    let roll2 = rng.next_u64x8();
    // if both bits are 1, that roll is a 1 out of 4
    let res = roll & roll2;
    // res contains 2 sets of 256 bits, split the sets and mask the 25 bits we don't want
    u32::max(
        res[0].count_ones()
            + res[1].count_ones()
            + res[2].count_ones()
            + (res[3] & MASK).count_ones(),
        res[4].count_ones()
            + res[5].count_ones()
            + res[6].count_ones()
            + (res[7] & MASK).count_ones(),
    )
}
fn thread_roll(rolls: u32) -> u8 {
    // seeding the generator
    let mut seed: Xoshiro256PlusX8Seed = Default::default();
    rand::thread_rng().fill_bytes(&mut *seed);
    let mut rng = Xoshiro256PlusX8::from_seed(seed);
    (0..rolls).map(|_| double_roll(&mut rng)).max().unwrap() as u8
}
fn par_roll() -> u8 {
    let thread_count = std::env::args().nth(1).map_or_else(
        || {
            println!("Expected thread count parametr, using all threads by default");
            thread::available_parallelism().unwrap().get()
        },
        |param| param.parse().expect("Invalid thread count"),
    ) as u32;
    let per_thread = 500_000_000 / thread_count + 1;
    let threads: Vec<thread::JoinHandle<u8>> = (1..thread_count)
        .map(|_| thread::spawn(move || thread_roll(per_thread)))
        .collect();
    let local_result = thread_roll(per_thread);
    threads
        .into_iter()
        .map(|t| t.join().unwrap())
        .max()
        .unwrap_or(0)
        .max(local_result)
}
