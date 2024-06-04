use gzp::par::compress::{ParCompress, ParCompressBuilder};
use gzp::par::decompress::{ParDecompress, ParDecompressBuilder};
use gzp::{deflate, Compression, ZWriter};
use std::env::args;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, Write};

#[derive(Debug, PartialEq)]
struct DiffEntry {
    // we use a relative index rather than absolute, so we can represent the values of this field with fewer bits than 64
    relative_index: u64,
    to: u8,
}

const CHUNK_SIZE: usize = 1024 * 64;

fn main() -> io::Result<()> {
    env_logger::init();
    let (action, file_a_path, file_b_path, output_path) = parse_args()?;

    match action.as_str() {
        "create_diff" => create_diff_file(&file_a_path, &file_b_path, &output_path),
        "recreate" => recreate_file(&file_a_path, &file_b_path, &output_path),
        _ => {
            log::error!("Invalid action: {}", action);
            std::process::exit(1);
        }
    }
}

fn parse_args() -> io::Result<(String, String, String, String)> {
    let args: Vec<String> = args().collect();
    if args.len() < 5 {
        log::error!(
            "Usage: {} <action> <file_a> <file_b> <output_file>",
            args[0]
        );
        log::error!("  action: 'create_diff' or 'recreate'");
        std::process::exit(1);
    }

    Ok((
        args[1].clone(),
        args[2].clone(),
        args[3].clone(),
        args[4].clone(),
    ))
}

/// Create a diff file between file_a & file_b, so file_b can be restored. Restoration is supported only for the second param file_b. Diff file
/// is compressed using zlib.
fn create_diff_file(file_a_path: &str, file_b_path: &str, diff_file_path: &str) -> io::Result<()> {
    let (file_a, file_b) = open_input_files(file_a_path, file_b_path)?;
    let diff_file = File::create(diff_file_path)?;
    let mut zlib_writer: ParCompress<deflate::Mgzip> = ParCompressBuilder::new()
        .compression_level(Compression::fast())
        .from_writer(diff_file);

    compare_files_and_write_diff(file_a, file_b, &mut zlib_writer)?;

    zlib_writer.finish().unwrap();

    log::info!("Compressed diff file created: {}", diff_file_path);
    Ok(())
}

/// This will create a diff file so that file_b can be recreated. file_a recreation is not supported.
fn compare_files_and_write_diff<R: Read + Seek, W: Write>(
    file_a: R,
    file_b: R,
    diff_writer: &mut W,
) -> io::Result<()> {
    let mut reader_a = BufReader::new(file_a);
    let mut reader_b = BufReader::new(file_b);
    let mut buf_a = [0u8; CHUNK_SIZE];
    let mut buf_b = [0u8; CHUNK_SIZE];
    let mut position: u64 = 0;
    let mut prev_index: u64 = 0;

    let mut diff_entries = Vec::with_capacity(CHUNK_SIZE);

    loop {
        // Read from fila_a and file_b into buffers buf_a and buf_b
        let (bytes_a, _) = (
            reader_a.read(&mut buf_a).unwrap_or(0),
            reader_b.read(&mut buf_b).unwrap_or(0),
        );

        // We're done with reading both files, so break loop. file_a and file_b are always the same size so we need to check only one of them
        if bytes_a == 0 {
            break;
        }

        for i in 0..bytes_a {
            let a_val = buf_a.get(i);
            let b_val = buf_b.get(i);

            // mismatch byte between file_a and file_b
            if a_val != b_val {
                let current_index = position + i as u64;
                let delta = current_index - prev_index;
                let diff_entry = DiffEntry {
                    relative_index: delta,
                    to: b_val.map(|v| *v).unwrap_or(0),
                };
                prev_index = current_index;
                diff_entries.push(diff_entry);

                if diff_entries.len() == CHUNK_SIZE {
                    // Write all accumulated diff entries and clear the buffer
                    for diff_entry in &diff_entries {
                        write_diff_entry(diff_writer, diff_entry)?;
                    }
                    diff_entries.clear();
                }
            }
        }
        position += bytes_a as u64;
    }

    // Write any remaining diff entries and clear the buffer
    for diff_entry in &diff_entries {
        write_diff_entry(diff_writer, diff_entry)?;
    }

    // Flush buffered writer before returning
    diff_writer.flush()?;

    Ok(())
}

fn recreate_file(file_a_path: &str, diff_file_path: &str, output_path: &str) -> io::Result<()> {
    let (mut file_a, diff_file) = open_input_files(file_a_path, diff_file_path)?;
    let diff_file = BufReader::new(diff_file); // Wrap the File in a BufReader
    let diff_file: ParDecompress<deflate::Mgzip> =
        ParDecompressBuilder::new().from_reader(diff_file);

    let output_file = File::create(output_path)?;
    recreate_file_inner(&mut file_a, diff_file, output_file)?;
    log::info!(
        "Output file recreated from compressed diff file: {}",
        output_path
    );
    Ok(())
}

// Recreate file_b from file_a + diff_fill
fn recreate_file_inner<R: Read + Seek, G: Read, W: Write>(
    file_a: &mut R,
    diff_file: G,
    mut output_file: W,
) -> io::Result<()> {
    let mut reader_a = BufReader::new(file_a);
    let mut diff_reader = BufReader::new(diff_file);
    let mut writer = BufWriter::new(&mut output_file);

    let mut buf_a = [0u8; CHUNK_SIZE];
    let mut current_position: u64 = 0;
    let mut diff_entry = read_diff_entry(&mut diff_reader);

    let mut output_buffer: Vec<u8> = Vec::with_capacity(CHUNK_SIZE);
    loop {
        let bytes_a = reader_a.read(&mut buf_a).unwrap_or(0);

        // File_a content exhausted
        if bytes_a == 0 {
            break;
        }

        output_buffer.clear();

        for i in 0..bytes_a {
            let mut next_entry = None;
            if let Some(ref mut entry) = diff_entry {
                if current_position as u64 == entry.relative_index {
                    // Apply the diff entry
                    output_buffer.push(entry.to);
                    next_entry = read_diff_entry(&mut diff_reader);
                    if let Some(ref mut next_e) = next_entry {
                        next_e.relative_index += entry.relative_index;
                    }
                } else {
                    // Write the byte from file_a
                    output_buffer.push(buf_a[i]);
                }
            } else {
                // Write the byte from file_a
                output_buffer.push(buf_a[i]);
            }

            current_position += 1;

            if let Some(_) = next_entry {
                diff_entry = next_entry;
            }
        }

        // Write the buffer to the output file
        writer.write_all(&output_buffer)?;
    }

    // Flush the buffered writer before returning
    writer.flush()?;

    Ok(())
}

fn read_diff_entry<R: Read>(diff_reader: &mut R) -> Option<DiffEntry> {
    let index = read_varint(diff_reader).ok()?;
    let mut buf = [0u8; 1];
    if diff_reader.read_exact(&mut buf).is_ok() {
        let to = buf[0];
        Some(DiffEntry {
            relative_index: index,
            to,
        })
    } else {
        None
    }
}

/// Reads a variable-length integer (varint) from the given reader and returns
/// it as a u64 value.
///
/// The varint encoding uses the least significant 7 bits of each byte to store
/// the integer value, with the most significant bit used as a continuation flag.
/// The continuation flag is set to 1 for all bytes except the last one, which
/// signals the end of the varint. This encoding is efficient for small integer
/// values, as it uses fewer bytes compared to a fixed-size integer.
fn read_varint<R: Read>(reader: &mut R) -> io::Result<u64> {
    let mut value: u64 = 0;
    let mut shift: u64 = 0;
    let mut buf = [0u8; 1];

    loop {
        reader.read_exact(&mut buf)?;
        let byte = buf[0];

        value |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    Ok(value)
}

fn write_diff_entry<W: Write>(diff_file: &mut W, diff_entry: &DiffEntry) -> io::Result<()> {
    write_varint(diff_entry.relative_index, diff_file)?;
    diff_file.write_all(&[diff_entry.to])
}

/// Writes a u64 value as a variable-length integer to the given writer.
///
/// The varint encoding uses the least significant 7 bits of each byte to store
/// the integer value, with the most significant bit used as a continuation flag.
/// The continuation flag is set to 1 for all bytes except the last one, which
/// signals the end of the varint. This encoding is efficient for small integer
/// values, as it uses fewer bytes compared to a fixed-size integer.
fn write_varint<W: Write>(mut value: u64, writer: &mut W) -> io::Result<()> {
    while value >= 0x80 {
        writer.write_all(&[(value & 0x7F) as u8 | 0x80])?;
        value >>= 7;
    }
    writer.write_all(&[value as u8])
}

fn open_input_files(file_a_path: &str, file_b_path: &str) -> io::Result<(File, File)> {
    let file_a = File::open(file_a_path)?;
    let file_b = File::open(file_b_path)?;
    Ok((file_a, file_b))
}

#[cfg(test)]
mod tests {

    use super::*;
    use std::io::Cursor;

    fn create_diff(file_a_content: &[u8], file_b_content: &[u8]) -> io::Result<Vec<u8>> {
        let file_a = Cursor::new(file_a_content);
        let file_b = Cursor::new(file_b_content);
        let mut diff_file = Cursor::new(Vec::new());

        compare_files_and_write_diff(file_a, file_b, &mut diff_file)?;
        Ok(diff_file.into_inner())
    }

    fn test_recreation(
        file_a_content: &[u8],
        file_b_content: &[u8],
        diff_file_content: &[u8],
    ) -> io::Result<()> {
        let mut file_a = Cursor::new(file_a_content);
        let diff_file = Cursor::new(diff_file_content);
        let mut recreated_file_b = Cursor::new(Vec::new());

        recreate_file_inner(&mut file_a, diff_file, &mut recreated_file_b)?;

        assert_eq!(recreated_file_b.into_inner(), file_b_content);
        Ok(())
    }

    #[test]
    fn file_a_and_file_b_are_the_same() {
        let file_a_content = b"hello world";
        let file_b_content = b"hello world";
        let diff_file_content = create_diff(file_a_content, file_b_content).unwrap();
        test_recreation(file_a_content, file_b_content, &diff_file_content).unwrap();
    }

    #[test]
    fn file_a_and_file_b_are_different() {
        let file_a_content = b"hello";
        let file_b_content = b"world";
        let diff_file_content = create_diff(file_a_content, file_b_content).unwrap();
        test_recreation(file_a_content, file_b_content, &diff_file_content).unwrap();
    }

    #[test]
    fn test_write_varint() {
        let mut buffer = Vec::new();
        let value: u64 = 12345;
        write_varint(value, &mut buffer).unwrap();

        assert_eq!(buffer, vec![0xB9, 0x60]);
    }

    #[test]
    fn test_read_varint() {
        let mut buffer = Cursor::new(vec![0xB9, 0x60]);
        let value = read_varint(&mut buffer).unwrap();

        assert_eq!(value, 12345);
    }

    #[test]
    fn test_read_write_varint() {
        let test_values = vec![0, 1, 127, 128, 16383, 16384, 2097151, 2097152, u64::MAX];

        for value in test_values {
            let mut buffer = Vec::new();
            write_varint(value, &mut buffer).unwrap();

            let mut buffer = Cursor::new(buffer);
            let read_value = read_varint(&mut buffer).unwrap();

            assert_eq!(value, read_value);
        }
    }
}
