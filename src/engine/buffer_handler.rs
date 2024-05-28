use flate2::read::MultiGzDecoder;
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::path::Path;
use zstd::stream::read::Decoder as ZstdDecoder;

pub fn create_buffered_input(input_filename: &str) -> Box<dyn BufRead> {
    // Handler for different (or no) compression types

    let input = File::open(input_filename).expect("Could not open the input file.");

    let input_format = Path::new(&input_filename)
        .extension()
        .and_then(|ext| ext.to_str())
        .expect("Failed to get the file extension.");

    match input_format {
        "gz" => {
            let gz_decoder = MultiGzDecoder::new(input);
            let reader = io::BufReader::new(gz_decoder);
            Box::new(reader)
        }
        "zst" => {
            let zstd_decoder = ZstdDecoder::new(input).unwrap();
            let reader = io::BufReader::new(zstd_decoder);
            Box::new(reader)
        }
        "vw" => {
            let reader = io::BufReader::new(input);
            Box::new(reader)
        }
        _ => {
            panic!("Please specify a valid input format (.vw, .zst, .gz)");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::{self, Read, Write};
    use tempfile::Builder as TempFileBuilder;
    use tempfile::NamedTempFile;
    use zstd::stream::Encoder as ZstdEncoder;

    fn create_temp_file_with_contents(
        extension: &str,
        contents: &[u8],
    ) -> io::Result<NamedTempFile> {
        let temp_file = TempFileBuilder::new()
            .suffix(&format!(".{}", extension))
            .tempfile()?;
        temp_file.as_file().write_all(contents)?;
        Ok(temp_file)
    }

    fn create_gzipped_temp_file(contents: &[u8]) -> io::Result<NamedTempFile> {
        let temp_file = TempFileBuilder::new().suffix(".gz").tempfile()?;
        let gz = GzEncoder::new(Vec::new(), Compression::default());
        let mut gz_writer = io::BufWriter::new(gz);
        gz_writer.write_all(contents)?;
        let gz = gz_writer.into_inner()?.finish()?;
        temp_file.as_file().write_all(&gz)?;
        Ok(temp_file)
    }

    fn create_zstd_temp_file(contents: &[u8]) -> io::Result<NamedTempFile> {
        let temp_file = TempFileBuilder::new().suffix(".zst").tempfile()?;
        let mut zstd_encoder = ZstdEncoder::new(Vec::new(), 1)?;
        zstd_encoder.write_all(contents)?;
        let encoded_data = zstd_encoder.finish()?;
        temp_file.as_file().write_all(&encoded_data)?;
        Ok(temp_file)
    }

    // Test for uncompressed file ("vw" extension)
    #[test]
    fn test_uncompressed_file() {
        let contents = b"Sample text for uncompressed file.";
        let temp_file =
            create_temp_file_with_contents("vw", contents).expect("Failed to create temp file");
        let mut reader = create_buffered_input(temp_file.path().to_str().unwrap());

        let mut buffer = Vec::new();
        reader
            .read_to_end(&mut buffer)
            .expect("Failed to read from the reader");
        assert_eq!(
            buffer, contents,
            "Contents did not match for uncompressed file."
        );
    }

    // Test for gzipped files ("gz" extension)
    #[test]
    fn test_gz_compressed_file() {
        let contents = b"Sample text for gzipped file.";
        let temp_file =
            create_gzipped_temp_file(contents).expect("Failed to create gzipped temp file");
        let mut reader = create_buffered_input(temp_file.path().to_str().unwrap());

        let mut buffer = Vec::new();
        reader
            .read_to_end(&mut buffer)
            .expect("Failed to read from the reader");
        assert_eq!(buffer, contents, "Contents did not match for gzipped file.");
    }

    // Test for zstd compressed files ("zst" extension)
    #[test]
    fn test_zstd_compressed_file() {
        let contents = b"Sample text for zstd compressed file.";
        let temp_file = create_zstd_temp_file(contents).expect("Failed to create zstd temp file");
        let mut reader = create_buffered_input(temp_file.path().to_str().unwrap());

        let mut buffer = Vec::new();
        reader
            .read_to_end(&mut buffer)
            .expect("Failed to read from the reader");
        assert_eq!(
            buffer, contents,
            "Contents did not match for zstd compressed file."
        );
    }

    // Test for unsupported file format
    #[test]
    #[should_panic(expected = "Please specify a valid input format (.vw, .zst, .gz)")]
    fn test_unsupported_file_format() {
        let contents = b"Some content";
        let temp_file =
            create_temp_file_with_contents("txt", contents).expect("Failed to create temp file");
        let _reader = create_buffered_input(temp_file.path().to_str().unwrap());
    }
}
