const C1: u32 = 0xcc9e2d51;
const C2: u32 = 0x1b873593;
const R1: u32 = 15;
const R2: u32 = 13;
const M: u32 = 5;
const N: u32 = 0xe6546b64;
const STEP: usize = 4;

#[inline]
pub(crate) fn hash32(key: &[u8]) -> u32 {
    hash32_with_seed(key, u32::default())
}

#[inline]
pub(crate) fn hash32_with_seed(key: &[u8], seed: u32) -> u32 {
    unsafe {
        let mut hash = seed;
        let mut data = key;

        let data_len = data.len();
        let data_len_remainder = data_len % STEP;
        let data_len_end = data_len - data_len_remainder;
        for i in (0..data_len_end).step_by(STEP) {
            let mut k = 0u32;

            for j in 0..STEP {
                k ^= u32::from(*data.get_unchecked(i + j)) << (8 * j as u32);
            }
            k = k.wrapping_mul(C1).rotate_left(R1).wrapping_mul(C2);

            hash ^= k;
            hash = hash.rotate_left(R2).wrapping_mul(M).wrapping_add(N);
        }

        // Handle the remaining bytes if any
        let mut k = 0u32;

        for i in 0..data_len_remainder {
            k ^= u32::from(*data.get_unchecked(data_len_end + i)) << (8 * i as u32);
        }

        k = k.wrapping_mul(C1).rotate_left(R1).wrapping_mul(C2);

        hash ^= k;
        hash ^= key.len() as u32;

        // Finalization mix - force all bits of a hash block to avalanche
        hash ^= hash >> 16;
        hash = hash.wrapping_mul(0x85ebca6b);
        hash ^= hash >> 13;
        hash = hash.wrapping_mul(0xc2b2ae35);
        hash ^= hash >> 16;

        hash
    }
}
