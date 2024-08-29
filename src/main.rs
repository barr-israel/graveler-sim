use rand_core::{RngCore, SeedableRng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use simd_rand::portable::{SimdRandX8, Xoshiro256PlusX8, Xoshiro256PlusX8Seed};

fn main() {
    let max = par_roll();
    println!("{max}");
}

const MASK: u64 = !((1 << 25) - 1); // we don't want the last 25 bits
fn double_coin_roll(rng: &mut Xoshiro256PlusX8) -> u32 {
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
fn thread_roll() -> u8 {
    // seeding the generator
    let mut seed: Xoshiro256PlusX8Seed = Default::default();
    rand::thread_rng().fill_bytes(&mut *seed);
    let mut rng = Xoshiro256PlusX8::from_seed(seed);
    // = to make it round up and not miss the last roll, /2 because each function call contains 2 sets rolls
    (0..=(1_000_000_000 / 2) / rayon::current_num_threads())
        .map(|_| double_coin_roll(&mut rng))
        .max()
        .unwrap() as u8
}
#[allow(dead_code)]
fn par_roll() -> u8 {
    // splits the work into threads, half the available threads was found to be best(blame
    // hyper-threading)
    rayon::ThreadPoolBuilder::new()
        .num_threads(std::thread::available_parallelism().unwrap().get() / 2)
        .build_global()
        .unwrap();
    (0..rayon::current_num_threads())
        .into_par_iter()
        .map(|_| thread_roll())
        .max()
        .unwrap()
}
#[allow(dead_code)]
fn roll() -> u8 {
    // seeding the generator
    let mut seed: Xoshiro256PlusX8Seed = Default::default();
    rand::thread_rng().fill_bytes(&mut *seed);
    let mut rng = Xoshiro256PlusX8::from_seed(seed);
    // = to make it round up and not miss the last roll, /2 because each function call contains 2 sets rolls
    (0..1_000_000_000 / 2)
        .map(|_| double_coin_roll(&mut rng))
        .max()
        .unwrap() as u8
}
